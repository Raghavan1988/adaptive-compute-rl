# PLAN.md

Concrete implementation plan for `when-to-think`. This translates the README
milestones and `AGENTS.md` research sequence into a trackable build order.

**Guiding rule:** each milestone must run end-to-end and produce machine-readable
output before the next begins. Do not build the RL stack before fixed-budget and
oracle baselines work. See [`AGENTS.md`](./AGENTS.md) for the binding constraints
and [`CLAUDE.md`](./CLAUDE.md) for the operational summary.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.

---

## Current status

**M0 (Infrastructure), M1 (Counterfactual compute dataset), and M2 (Oracle allocation)
are complete** as of 2026-09-01. **M3 (Supervised value-of-compute probe) is built and
tested end-to-end on synthetic data** as of 2026-09-03; its scientific verdict awaits a
real run. Next up: run M3 with the real SLM, then **M4** (RL adaptive policy).

- M0 pipeline runs from config: `python scripts/evaluate.py --config configs/experiment/gsm8k_smoke.yaml`
  writes `eval.jsonl` (per-example scores + reward sweep), sharded hidden states, and
  a `run_record.json` to `results/<run_id>/`.
- M1 sweep runs from config: `python scripts/generate_fixed_budgets.py --config ...`
  writes `fixed_budget_runs.jsonl` (per-example × per-budget records) + hidden states;
  `python scripts/summarize_fixed_budgets.py --run-dir results/<run_id>` writes
  `summary.json` and `accuracy_vs_compute.png` and prints the heterogeneity verdict.
- M2 oracle builds from an M1 run: `python scripts/build_oracle.py --run-dir results/<run_id>`
  writes `oracle_summary.json`, per-example `oracle_allocation.jsonl`, and
  `oracle_frontier.png`, and prints whether the oracle beats fixed budgets at matched
  accuracy (the M2 exit gate).
- M3 probe builds from a sweep covering all splits:
  `python scripts/generate_fixed_budgets.py --config configs/experiment/gsm8k_m3.yaml --splits train,val,test`
  then `python scripts/train_probe.py --run-dir results/<run_id> --config configs/experiment/gsm8k_m3.yaml`
  writes `probe_results.json`, per-example `probe_predictions.jsonl`, and layer-wise +
  probe-vs-baseline plots, and prints the Question 2 verdict (does internal state beat the
  input-only baseline?). Fits on train, tunes on val, scores test once.
- 105 tests passing (`pytest`); lint clean (`ruff check src scripts tests`).
- **The pipelines are verified end-to-end but not yet with the real model** — M0/M1 on a
  tiny random model, M2 on a synthetic heterogeneous run. The scientific answers (is
  value-of-compute heterogeneous? does the oracle beat fixed budgets?) require running
  M1's sweep and then M2 with the actual SLM (Qwen2.5-1.5B); see `Grok.md` for the M1 run
  task.

Concrete decisions locked in during M0 (see AGENTS.md §25 — these are
research-significant and should not change silently):

- **Dataset id:** `openai/gsm8k` (the bare `gsm8k` alias was dropped by `datasets>=3`;
  same corpus).
- **Default model:** `Qwen/Qwen2.5-1.5B-Instruct`, frozen. Fits the 16 GB laptop 4090
  comfortably; model is a config field, so larger models are a config change.
- **Compute proxy:** `reasoning_tokens` (generated token count) — never called FLOPs.
- **Reward:** `R = R_task − λ·C`; `λ` is a **sweep** (`reward.lambda_compute_sweep`),
  `reward_task` and `reward_compute` logged separately.
- **M0 eval method name:** `single_pass_fixed_budget` (one reasoning pass; STOP/CONTINUE
  arrives in M4).
- **Env:** work in the base conda env; `numpy` pinned `<2` so the rest of that env keeps
  working (see `pip install -e ".[dev]"`).

---

## The four questions this plan must answer

1. Does additional reasoning have **heterogeneous value** across examples?
2. Can hidden states predict that value better than **simple confidence baselines**?
3. Can a learned **stop/continue policy** exploit the signal?
4. Does the policy improve the **accuracy-compute Pareto frontier** over fixed budgets?

Everything below exists to answer these cleanly. Nothing else is required until
they are answered.

---

## M0 — Infrastructure

Goal: load a model + benchmark and generate scored, reproducible answers with
selective hidden-state extraction.

- [x] Scaffold repo layout (`src/when_to_think/{data,models,generation,representations,probes,policies,rewards,evaluation,utils}`, `scripts/`, `tests/`, `configs/`, `results/`, `artifacts/`).
- [x] `pyproject.toml` with pinned, minimal deps.
- [x] Config loading (`configs/` + CLI). No experimental constant hard-coded in source.
- [x] Model + tokenizer loader (model name configurable, SLM frozen).
- [x] GSM8K loader with explicit, non-overlapping train/val/test splits.
- [x] Deterministic answer extraction + exact-match reward (rule-based).
- [x] Selective hidden-state extraction: record layer index, token position,
      reasoning step, model revision, pooling method. Sharded/streamed storage.
- [x] Run-record writer: `run_id, timestamp, git_commit, model_name, model_revision,
      tokenizer_name, dataset, split, seed, generation_config, reasoning_budget,
      decision_interval, reward_config, lambda_compute`.
- [x] Seed control: Python / NumPy / PyTorch / data sampling.

**Tests (required):** answer extraction (valid + malformed), reward calculation
(correct/incorrect task reward, compute penalty applied exactly once), budget
enforcement (never silently exceeds max), split integrity (no overlap). ✅ All present.

**Built:** `config.py` (typed config + CLI overrides), `models/loader.py` (frozen SLM),
`data/gsm8k.py` (disjoint splits), `rewards/{answer_extraction,reward}.py`,
`representations/extraction.py` (selective extraction + sharded writer),
`generation/generate.py` (single-pass, budget-enforced), `utils/{seeding,run_record}.py`,
`evaluation/evaluate.py`, and `scripts/evaluate.py`.

**Exit:** one config-driven command produces a scored JSONL with token counts,
latency, and extracted hidden states for a GSM8K sample.
✅ Done — `python scripts/evaluate.py --config configs/experiment/gsm8k_smoke.yaml`
writes `eval.jsonl` (per-example score + reward sweep), sharded hidden states, and
`run_record.json`. Verified end-to-end on a tiny model; run with the real SLM
(Qwen2.5-1.5B) for an actual result.

---

## M1 — Counterfactual compute dataset

Goal: quantify whether more reasoning helps, hurts, or does nothing — per example.

- [x] `scripts/generate_fixed_budgets.py`: run the same examples at multiple fixed
      budgets (e.g. 0/direct, 128, 256, 512, 1024 tokens), same test examples across budgets.
- [x] Per-example, per-budget records: answer, correct, reasoning_tokens, latency,
      hidden states at candidate decision points.
- [x] Aggregate the heterogeneity of value: fraction where more reasoning helps /
      hurts / is neutral (respect **no monotonicity** — track correct→wrong flips).
- [x] First accuracy-vs-compute plot, generated from the result files.

**Tests (required):** budget-forced generation (budget 0 = direct answer, budget never
exceeded, decision-point hidden state present), counterfactual categorization
(helped/hurt/unchanged), heterogeneity verdict, and non-monotonicity detection even when
endpoints agree. ✅ All present (`test_fixed_budgets.py`, `test_counterfactual.py`).

**Built:** `generation/fixed_budgets.py` (two-phase budget-forced generation:
reasoning capped at the budget, then a separate answer-elicitation phase, so
`reasoning_tokens` is a clean compute proxy independent of answer length),
`evaluation/fixed_budget_eval.py` (sweep over the *same* test examples × budgets ×
samples), `evaluation/counterfactual.py` (per-example value-of-compute categorization +
heterogeneity/non-monotonicity), `evaluation/plots.py` (accuracy-vs-compute, generated
from `summary.json`), and thin `scripts/{generate_fixed_budgets,summarize_fixed_budgets}.py`.

**Exit:** Question 1 answered with data. If value is *not* heterogeneous, flag it —
the hypothesis is weak and should be reconsidered before M2+.
⚠️ Pipeline complete and verified end-to-end, but only on a tiny random model (a smoke
test). The scientific verdict is **not yet obtained** — run the sweep with Qwen2.5-1.5B
on a real GSM8K sample to answer Question 1. The summary script already prints the
heterogeneity verdict and warns when value is homogeneous.

---

## M2 — Oracle allocation

Goal: upper bound on the value of adaptive allocation.

- [x] `scripts/build_oracle.py`: per example, pick the cheapest budget that is correct.
- [x] Oracle Pareto frontier plot over fixed-budget baselines.
- [x] Quantify maximum available compute savings at matched accuracy.

**Tests (required):** oracle construction on toy trajectories:
`STOP correct / CONTINUE correct → STOP`; `STOP wrong / CONTINUE correct → CONTINUE`
if gain exceeds cost; `STOP wrong / CONTINUE wrong → cheaper action`. ✅ All present
(`test_oracle.py`), plus frontier-monotonicity, matched-accuracy savings, and the
"oracle doesn't beat fixed" case.

**Built:** `evaluation/oracle.py` — the oracle is an *omniscient* per-example allocator
(the upper bound, not a deployable method): given penalty `λ` it picks per example
`argmax_b [acc(e,b) − λ·tokens(e,b)]`, cheaper budget winning ties. `λ=0` is the
accuracy-max oracle ("cheapest budget that is correct"); sweeping `λ` traces the exact
Pareto frontier (data-derived breakpoints, no arbitrary grid). Savings are reported at
**matched accuracy** (§4.1) and correct→wrong flips are handled by the argmax with **no
monotonicity assumption** (§4.5). Plus `evaluation/plots.py::plot_oracle_frontier` and
thin `scripts/build_oracle.py` (writes `oracle_summary.json`, per-example
`oracle_allocation.jsonl`, `oracle_frontier.png`).

**Exit:** if the oracle does not meaningfully beat fixed budgets, **stop and
reconsider** before building probes or RL.
⚠️ Pipeline complete and verified end-to-end (script + oracle + plot produce all
outputs; hand-checked on a synthetic heterogeneous run). The **scientific verdict is not
yet obtained** — run `build_oracle.py` on a *real* M1 run (Qwen2.5-1.5B; see `Grok.md`)
before trusting `oracle_dominates_fixed`. The script prints the verdict and warns when
the oracle fails to beat fixed budgets.

---

## M3 — Supervised value-of-compute probe

Goal: can frozen hidden states predict the marginal value of continuing?

- [x] `scripts/train_probe.py`: train on train split, tune on val, report test separately.
      Strict split discipline — the scaler and probe fit on TRAIN only, layer/alpha
      chosen by VAL, TEST scored once (`probes/train.py`).
- [x] Targets: Δ_t = P(correct|CONTINUE) − P(correct|STOP) (regression), and the binary
      "continuing fixes an incorrect answer" label. Both are population quantities from the
      counterfactual samples, kept explicitly distinct from "P(current answer correct)"
      (logged as `p_stop` / `current_correct`, never used as the target). `continue_mode`
      selects next-budget (marginal) vs to-max value (`probes/dataset.py`).
- [x] Probe types: logistic (binary) + ridge/linear (regression), pure-numpy and
      deterministic (`probes/models.py`). MLP not added — not justified yet.
- [~] Baselines: **input-only difficulty** and **prior/base-rate** built and compared
      (`probes/baselines.py`). **Entropy** and **verbalized confidence** are deferred —
      they need signals the sweep does not yet log (token logits; a confidence
      elicitation). The probe-vs-baseline harness accepts any extra feature matrix so they
      drop in once collected. Until then Q2 is answered vs the input-only baseline, and the
      scope limit is reported in `probe_results.json` (`baselines_not_yet_available`).
- [x] Layer-wise analysis (per candidate layer, selected on val) and decision-point
      analysis (test metrics per stop-budget) — `per_stop_budget_test`, layer-wise plot.

**Tests (required):** hidden-state reader round-trip; target definitions (delta +
fixes-incorrect); split derivation + leakage guard; probe models (ridge/logistic,
train-only scaler, determinism); metrics (AUROC incl. ties/degenerate, R², Brier);
and an end-to-end run where a hidden state encoding the target beats the input-only
baseline and prior, with `train/val/test` sizes and no test leakage. ✅ All present
(`test_hidden_reader.py`, `test_probe_dataset.py`, `test_probe_models.py`,
`test_probe_metrics.py`, `test_probe_train.py`).

**Built:** `representations/reader.py` (join hidden states back to outcome rows),
`probes/{dataset,models,metrics,baselines,train,plots}.py`, `scripts/train_probe.py`,
`configs/experiment/gsm8k_m3.yaml`, and a `--splits` option on the fixed-budget sweep
so TRAIN/VAL probe data is generated alongside TEST (default stays test-only, so the M1
run is unchanged). Config gained a typed `ProbeConfig`.

**Exit:** Question 2 answered. Report train/val/test separately; describe results
as *decodability*, not mechanism.
⚠️ Pipeline complete and verified end-to-end, but only on synthetic decodable data (a
smoke test, like M0–M2). The scientific verdict is **not yet obtained** — run the M3
sweep + probe with Qwen2.5-1.5B on real GSM8K (see `Grok_M3.md`) before trusting
`hidden_state_beats_input_baseline`. The script prints the verdict and warns when the
hidden state is not more decodable than the input alone.

---

## M4 — RL adaptive policy

Goal: a STOP/CONTINUE policy on hidden states that beats baselines at matched compute.

- [ ] STOP/CONTINUE environment:
      - STOP → terminate, score final answer, apply accumulated compute cost.
      - CONTINUE → grant a fixed reasoning increment, update state, accrue cost, next decision.
      - Always enforce a max compute budget.
- [ ] Policy architecture separable from the frozen base SLM.
- [ ] Reward `R = R_task − λ·C`, with `λ` swept — never a single hard-coded value.
- [ ] `scripts/train_policy.py` (thin entry point).
- [ ] Collapse diagnostics logged every run (see `CLAUDE.md` / `AGENTS.md` §16).
- [ ] Reward-hacking checks (`AGENTS.md` §17).
- [ ] Compare against **all** baselines at matched / explicitly-reported compute.

**Tests (required):** reward calculation and STOP/CONTINUE transition semantics;
budget enforcement in the env.

**Exit:** Questions 3 and 4 answered — the headline accuracy-vs-compute Pareto
curve with uncertainty across seeds / bootstrap CIs.

---

## M5 — Analysis

- [ ] Error taxonomy (where the policy makes expensive mistakes).
- [ ] Calibration: ECE, Brier, reliability diagrams (if policy predicts value/probability).
- [ ] Layer and decision-interval ablations.
- [ ] Ablations: internal state vs output confidence; internal state vs input-only;
      correctness-prediction vs value-of-compute prediction; `λ` sweep.
- [ ] Cross-dataset generalization (add one harder/shifted benchmark, e.g. MATH).
- [ ] Final figures + research write-up.

---

## Headline deliverable

A reproducible **Accuracy vs. Average reasoning compute** Pareto curve showing the
adaptive policy dominating or improving on fixed-budget baselines over meaningful
parts of the frontier, with uncertainty bands, generated entirely from
machine-readable result files.

---

## Stretch (only after M0–M5 are solid)

RETRIEVE action · larger-model escalation · tool use · multi-action policy ·
causal interventions on value-of-compute representations. Out of scope until the
core stop/continue result is established (`AGENTS.md` §2).
