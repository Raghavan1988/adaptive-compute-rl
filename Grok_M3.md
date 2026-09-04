# Grok task — Run the real M3 value-of-compute probe with Qwen2.5-1.5B

**Objective:** Produce the first *scientific* M3 result — train the supervised
value-of-compute probe on real Qwen2.5-1.5B hidden states from GSM8K, and answer
**Question 2: can frozen hidden states predict the value of continuing to reason
better than simple baselines?**

This is an execution + analysis task, not a build task. The M3 pipeline is already
implemented, tested (105 tests pass), and verified end-to-end — **but only on synthetic
decodable data**, which is a smoke test and produces a meaningless verdict. Your job is
to run it for real and report what the data says, honestly, either way.

Read [`AGENTS.md`](./AGENTS.md) (authoritative contract), [`CLAUDE.md`](./CLAUDE.md),
and the M3 section of [`PLAN.md`](./PLAN.md) before starting. This depends on M1/M2
having a real verdict first (`Grok.md`): if the oracle does not beat fixed budgets on
the real model, **stop and reconsider** before spending compute on the probe (M2 exit).

---

## Background (what M3 measures)

At each *decision point* — the hidden state after `stop_budget` reasoning tokens — the
probe predicts the **value of continuing**, two explicit, distinct ways:

- `value_of_compute` (regression): Δ = P(correct | continue) − P(correct | stop).
- `fixes_incorrect` (binary): 1 iff P(correct | stop) < τ ≤ P(correct | continue).

Both are population quantities estimated from the counterfactual samples, and both are
kept distinct from "P(the current answer is correct)" (a probe predicting current
correctness is solving a *different* task). The probe is deliberately linear
(logistic / ridge) so a positive result is *decodability*, not evidence the model
"knows" its own value of compute — describe it that way.

**The claim to test:** the hidden-state probe beats the input-only difficulty baseline
(and the prior) on the held-out TEST split. If it does not, report that — internal
state may not carry the signal, and the hypothesis weakens before M4.

---

## Preconditions

- Working dir: `/home/raghavan/adaptive-compute-rl`, base conda env (Python 3.11).
- Deps already installed. **`numpy` must stay `<2`** — do not upgrade it.
- Hardware: 16 GB RTX 4090 Laptop GPU. Qwen2.5-1.5B in bf16 fits comfortably.
- Sanity check before running:
  ```bash
  python -m pytest -q          # expect 105 passed
  ruff check src scripts tests # expect clean
  ```

---

## Steps

### 1. Generate probe data on ALL splits (not just test)

A probe needs TRAIN/VAL/TEST hidden states. The M1 sweep runs test-only by default;
M3 uses `--splits train,val,test`. Pin the model revision (AGENTS.md §9) and cap the
train subsample at run time so the cap does not leak into the shared data config:

```bash
python scripts/generate_fixed_budgets.py \
  --config configs/experiment/gsm8k_m3.yaml \
  --splits train,val,test \
  --set model.revision=<commit-hash> \
  --set data.max_train_examples=400   # start modest; val is carved from train (10%)
```

`configs/experiment/gsm8k_m3.yaml` stores **four layers** (`[-1,-9,-17,-25]`) so the
probe can do a layer-wise analysis. This is more hidden-state storage than M1 — watch
disk. Note the printed `run_id`.

### 2. Train + evaluate the probe

```bash
python scripts/train_probe.py \
  --run-dir results/<run_id> \
  --config configs/experiment/gsm8k_m3.yaml
```

This writes `probe_results.json`, per-example `probe_predictions.jsonl`, and layer-wise
+ probe-vs-baseline plots, and prints the Question 2 verdict for both targets. It fits
on train, selects the layer + L2 strength on val, and scores test exactly once.

### 3. Report

From `probe_results.json` (never hand-typed numbers), report per target:

- selected layer + alpha, and the TEST metric (AUROC for `fixes_incorrect`, R² for
  `value_of_compute`) for the hidden-state probe, the input-only baseline, and the prior;
- `hidden_state_beats_input_baseline` and `decodability_margin` — the Q2 verdict;
- the layer-wise val curve (which layer is most decodable) and the per-stop-budget
  breakdown (does decodability change with how much reasoning has already happened);
- honest caveats: base rates, tiny test splits, any target with one class only (AUROC
  is `nan` there — say so rather than papering over it).

---

## Must hold (binding invariants)

- **No test-set training (AGENTS.md §4.2).** The scaler and probe fit on TRAIN only;
  the layer and alpha are chosen on VAL; TEST is scored once. The pipeline enforces
  this structurally — do not add a step that tunes on test.
- **Split integrity.** Splits are derived from `example_id` and asserted disjoint. Do
  not merge splits or relabel ids.
- **Preserve failures (§4.4).** Keep incorrect / malformed / degenerate trajectories —
  they define the targets. Do not filter them out to make the probe look better.
- **Decodability, not mechanism (CLAUDE.md).** A good probe shows the value of compute
  is *linearly readable* off the hidden state. It does not show introspection,
  metacognition, or causation. Word the report accordingly.
- **Report the null honestly.** If the hidden state does not beat the input-only
  baseline, that is a result — write it up; do not tune until it looks positive.

---

## Follow-up (not required for the first verdict)

Two baselines named in PLAN.md M3 — **entropy-based** and **verbalized confidence** —
are not yet collected: they need generation-side signals (per-step token logits for
predictive entropy; a "how confident are you (0–1)?" elicitation for verbalized
confidence). Adding them is a separate generation pass; `probes/baselines.py` and
`probes/train.py` are written so a new feature matrix drops in without reshaping the
pipeline. Until then, Q2 is answered against the input-only baseline, and
`probe_results.json` records that scope limit in `baselines_not_yet_available`.
