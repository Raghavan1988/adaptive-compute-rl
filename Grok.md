# Grok task — Run the real M1 sweep with Qwen2.5-1.5B

**Objective:** Produce the first *scientific* M1 result — run the counterfactual
fixed-budget sweep with the real SLM (`Qwen/Qwen2.5-1.5B-Instruct`) on a GSM8K
sample, and answer **Question 1: does additional reasoning have heterogeneous
value across examples?**

This is an execution + analysis task, not a build task. The M1 pipeline is already
implemented, tested (74 tests pass), and verified end-to-end — **but only on a tiny
random model**, which is a smoke test and produces a meaningless verdict. Your job
is to run it for real and report what the data says.

Read [`AGENTS.md`](./AGENTS.md) (authoritative contract) and [`CLAUDE.md`](./CLAUDE.md)
before starting. The invariants in the "Must hold" section below are binding.

---

## Background (what M1 measures)

For each test example, the sweep generates an answer at several **fixed reasoning
budgets** (e.g. 0/direct, 128, 256, 512 tokens), scores each with rule-based exact
match, and records per-example × per-budget rows plus a decision-point hidden state.
The analysis then categorizes, per example, how accuracy changes from the cheapest to
the most expensive budget:

- **helped** — more compute raised accuracy
- **hurt** — more compute *lowered* accuracy (a correct→wrong flip; we make **no
  monotonicity assumption** and keep these)
- **unchanged** — no change (split into already-correct / never-correct)

**Value of compute is "heterogeneous" iff examples do not all fall in one bucket.**
If it is *not* heterogeneous, the M1 exit criterion says to flag it — the project
hypothesis is weak and should be reconsidered before M2. Report the verdict honestly
either way.

---

## Preconditions

- Working dir: `/home/raghavan/adaptive-compute-rl`, base conda env (Python 3.11).
- Deps already installed (editable `pip install -e ".[dev]"`). **`numpy` must stay
  `<2`** in this env — do not upgrade it (breaks scipy/numba/etc.).
- Hardware: 16 GB RTX 4090 Laptop GPU. Qwen2.5-1.5B in bf16 fits comfortably.
- Model weights will download from HuggingFace on first run (needs network once).
- Sanity check before running:
  ```bash
  python -m pytest -q          # expect 74 passed
  ruff check src scripts tests # expect clean
  ```

---

## Steps

### 1. Create a real experiment config (do NOT edit the smoke config)

`configs/experiment/gsm8k_smoke.yaml` has deliberately tiny budgets/counts and is
not a reportable result. Create a **new** file
`configs/experiment/gsm8k_m1.yaml` instead:

```yaml
name: gsm8k_m1
seed: 0

model: qwen2.5-1.5b
data: gsm8k

generation:
  max_reasoning_budget: 512      # must be >= max(fixed_budgets); validation enforces this
  reasoning_increment: 64
  decision_interval: 64
  fixed_budgets: [0, 128, 256, 512]   # 0 = direct answer, no reasoning
  temperature: 0.7
  top_p: 0.95
  do_sample: true
  num_samples: 5                 # counterfactual samples per (example, budget); >1 estimates P(correct|budget)

representation:
  layers: [-1]
  token_position: last
  pooling: null

reward:
  task_reward_correct: 1.0
  task_reward_incorrect: 0.0
  lambda_compute_sweep: [0.0, 0.0001, 0.0005, 0.001]   # lambda is never a single value
  compute_proxy: reasoning_tokens

output_dir: results
notes: "M1 real run: Qwen2.5-1.5B, GSM8K test subsample, counterfactual fixed-budget sweep."
```

Cap the test subsample at run time (keep it out of the shared data config so the cap
doesn't leak into other experiments) with `--set data.max_test_examples=N`.

### 2. Pin the model revision (reproducibility, AGENTS.md §9)

Real runs must pin the exact model commit. Fetch the current main revision of
`Qwen/Qwen2.5-1.5B-Instruct` from the HF Hub and set it in
`configs/model/qwen2.5-1.5b.yaml` (`revision: <commit-hash>`), **or** pass
`--set model.revision=<commit-hash>` on the command line. Record which hash you used
— it is written into `run_record.json` automatically.

### 3. Pilot run first (sanity check, ~50 examples)

Confirm the real model produces sane output and a non-degenerate signal before
committing to the full run:

```bash
python scripts/generate_fixed_budgets.py \
  --config configs/experiment/gsm8k_m1.yaml \
  --set data.max_test_examples=50 \
  --set model.revision=<commit-hash>
```

Then summarize (the script prints the run dir; use it below):

```bash
python scripts/summarize_fixed_budgets.py --run-dir results/<run_id>
```

**What to check in the pilot:**
- Budget-0 accuracy is plausibly low and accuracy rises with budget (a sane model
  should get *some* GSM8K right with reasoning). If budget-512 accuracy is ~0, the
  prompt/answer-extraction is likely broken for this model — stop and inspect
  `fixed_budget_runs.jsonl` (look at `prediction` vs `ground_truth`,
  `finished_naturally`, `forced_answer`).
- `reasoning_tokens` never exceeds the budget (budget enforcement).
- Some examples are wrong at every budget and some flip — i.e. there is spread.

### 4. Full run

Once the pilot looks sane, scale the sample up (e.g. 200–300 examples). Mind the
cost note below — generation is **unbatched** (one sequence at a time), so cost is
roughly `n_examples × n_budgets × num_samples × per-generation-time`.

```bash
python scripts/generate_fixed_budgets.py \
  --config configs/experiment/gsm8k_m1.yaml \
  --set data.max_test_examples=200 \
  --set model.revision=<commit-hash>
```

### 5. Summarize + plot (generated from files, never hand-typed)

```bash
python scripts/summarize_fixed_budgets.py --run-dir results/<run_id>
```

Produces `summary.json` and `accuracy_vs_compute.png` in the run dir and prints the
heterogeneity verdict.

### 6. Interpret and report

Read `summary.json`. Report:
- `accuracy_by_budget` — the accuracy-vs-compute curve (accuracy + mean reasoning
  tokens per budget).
- `value_of_compute.counts` (helped / hurt / unchanged / unchanged_correct /
  unchanged_wrong), the fractions, `nonmonotone_examples`, and `mean_delta`.
- **`value_of_compute.heterogeneous`** — the headline verdict for Question 1.
- Whether any examples were **hurt** by more compute (correct→wrong) — concrete
  evidence for the no-monotonicity stance.

If `heterogeneous` is `False`, say so plainly and flag it per the M1 exit criterion.
Do not spin a weak result as strong.

---

## Must hold (binding invariants — AGENTS.md §4, §7, §25)

- **No test-set training.** This sweep only *evaluates* on the test split (no fitting
  of probes/policies/thresholds). That is allowed. Do not tune anything on test.
- **Preserve failures.** Every trajectory — right, wrong, malformed, truncated — is a
  result and must stay in `fixed_budget_runs.jsonl`. Do not filter them out.
- **Same examples across budgets.** The identical test examples must be run at every
  budget (the orchestrator already does this; don't subsample per-budget).
- **No monotonicity assumption.** Keep and report correct→wrong flips
  (`nonmonotone_examples`, `hurt`).
- **`λ` is never universal.** Keep `lambda_compute_sweep` a list. `reward_task` and
  `reward_compute` stay separate logged fields (already the case).
- **Compute proxy is `reasoning_tokens`** — never call it FLOPs.
- **Plots/tables come from result files**, never hand-typed numbers.
- **Machine-readable output**, config, and `run_record.json` (run_id, git_commit,
  model + revision, dataset/split, seed, budgets, reward config) are recorded per run
  — the pipeline does this; verify `run_record.json` exists and the revision is pinned.

---

## Cost / performance notes

- Generation is **unbatched** — `evaluation/fixed_budget_eval.py` loops
  `example × budget × sample` and calls `generate_at_budget` one sequence at a time.
  Expect the 200-example × 4-budget × 5-sample run to take on the order of an hour+ on
  the laptop 4090. Start with the 50-example pilot to time it, then extrapolate.
- If it is too slow: reduce `num_samples` (fewer counterfactual draws) or
  `max_test_examples` first, before touching budgets. Batching generation is a
  possible optimization but is **out of scope** for this task — do not refactor the
  generation loop; just run what exists.
- bf16 on `device: auto` should land on CUDA. Confirm GPU is used (check `nvidia-smi`
  during the run); if it silently ran on CPU it will be far slower.

---

## Known interpretation caveats (call these out in your report)

- **Sampling noise vs. real value.** With `do_sample: true` and `num_samples=5`,
  per-example accuracy is fractional (0, .2, .4, …). The default `unchanged_tol=0.0`
  in `summarize_counterfactuals` means *any* nonzero min→max delta counts as
  helped/hurt — so a single lucky/unlucky sample can flip a bucket. Note this when
  reporting; a small `helped`/`hurt` count at 5 samples may be sampling noise, not
  signal. (The summarize script currently hard-codes `tol=0.0` with no CLI flag —
  adding an optional `--unchanged-tol` flag is a reasonable small refinement if you
  want a robustness check, but is optional.)
- The verdict compares only the **cheapest vs most expensive** budget for the
  helped/hurt/unchanged buckets; `nonmonotone_examples` uses the full budget ladder.
  Both come from the same `summary.json`.
- The tiny-model smoke run reported `HETEROGENEOUS value: False` — that was expected
  garbage from a random model and tells you nothing. Ignore it.

---

## Deliverables (definition of done)

1. `configs/experiment/gsm8k_m1.yaml` committed.
2. Model revision pinned and recorded.
3. A completed run dir under `results/<run_id>/` containing
   `fixed_budget_runs.jsonl`, `hidden_states/`, `run_record.json`, `summary.json`,
   and `accuracy_vs_compute.png`.
4. A short written finding: the accuracy-vs-compute curve, the heterogeneity verdict
   for Question 1, the helped/hurt/unchanged/non-monotone counts, and — if value is
   not heterogeneous — an explicit flag that the hypothesis looks weak.

---

## File reference

| Purpose | Path |
|---|---|
| Sweep entry point | `scripts/generate_fixed_budgets.py` |
| Summary + plot entry point | `scripts/summarize_fixed_budgets.py` |
| Sweep orchestrator | `src/when_to_think/evaluation/fixed_budget_eval.py` |
| Budget-forced generation | `src/when_to_think/generation/fixed_budgets.py` |
| Counterfactual analysis | `src/when_to_think/evaluation/counterfactual.py` |
| Accuracy-vs-compute plot | `src/when_to_think/evaluation/plots.py` |
| Config schema | `src/when_to_think/config.py` |
| Milestone tracker | `PLAN.md` (M1 section) |
