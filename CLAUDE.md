# CLAUDE.md

Operational guide for Claude Code in the `when-to-think` repository.

> **Read [`AGENTS.md`](./AGENTS.md) first.** It is the authoritative contract for
> working here. This file is a short operational layer on top of it, not a
> replacement. When the two disagree, `AGENTS.md` wins.

---

## What this project is

A research codebase investigating a single question:

> Can an RL policy conditioned on an SLM's internal representations learn when to
> stop or continue reasoning, achieving a better accuracy-compute trade-off than
> fixed reasoning budgets?

The action space is exactly `{STOP, CONTINUE}`. See [`README.md`](./README.md)
for the full scientific framing and [`PLAN.md`](./PLAN.md) for the concrete
build order.

**Correct experimental methodology beats feature velocity.** The best version of
this repo is the smallest system that convincingly answers the question above.

---

## Non-negotiables (the ones that bite)

These are the invariants most likely to be violated by well-meaning changes.
The full list is in `AGENTS.md` §4 and §25.

- **No test-set training.** Never fit probes, policies, thresholds, calibration,
  or hyperparameters on the held-out test split.
- **Matched compute comparisons.** Never compare an adaptive method to a fixed
  method at different compute and attribute the gap to routing.
- **Preserve failures.** Do not drop incorrect / malformed / timed-out / collapsed
  trajectories — they are results.
- **No monotonicity assumption.** More tokens can turn a correct answer wrong.
- **Exact reward semantics.** Rule-based reward for verifiable tasks; no LLM judge
  unless explicitly required. Keep `reward_task` and `reward_compute` as separate
  logged fields.
- **`λ` is never universal.** Support sweeps over the compute penalty; never
  hard-code one value as canonical.
- **Precise claims.** A probe shows *decodability*, not causality, introspection,
  or metacognition. Use terms like `value_of_compute`, `hidden_state_signal`.

Treat changes to reward definition, answer extraction, task metric, data split,
reasoning budget, oracle definition, hidden-state position, baseline
implementation, or policy action semantics as **research-significant**: call the
change out explicitly, update tests, and version/invalidate old results.

---

## Where code goes

```text
src/when_to_think/   # all reusable logic (see AGENTS.md §10)
  data/ generation/ representations/ probes/ policies/ rewards/ evaluation/ utils/
scripts/             # thin entry points only — no core logic inline
tests/               # research-critical behavior is tested
configs/             # experiments run from config, not edited constants
notebooks/           # exploration & plotting only, never the sole implementation
results/ artifacts/  # machine-readable outputs; figures/tables generated from them
```

- Scripts are thin: parse config → call into `when_to_think` → write results.
- Experimental quantities (model, dataset, budgets, decision interval, layers,
  lr, `λ`, rollouts, seed) are **configurable**, never hard-coded constants.
- Every major evaluation writes a machine-readable per-example file (JSONL/Parquet).
  Plots and tables are generated from those files — never type numbers by hand.

---

## Build order

Follow `PLAN.md` / README milestones M0→M5. Do **not** build a sophisticated RL
stack before fixed-budget and oracle baselines run end-to-end. When torn between
a fancier feature and a cleaner central comparison, choose the cleaner comparison.

---

## Definition of done for a research change

- works end-to-end
- research-critical logic has tests (answer extraction, reward, budget
  enforcement, oracle construction, split integrity)
- config is recorded (run_id, git_commit, model, dataset/split, seed, budgets,
  reward config, `λ`, etc. — see `AGENTS.md` §9)
- output is machine-readable
- relevant baselines still run
- docs updated when semantics change

---

## Conventions

- **Names describe scientific meaning:** `value_of_compute`, `premature_stop_rate`,
  `compute_penalty` — not `magic_score`, `smart_router`, `best_policy_final_v3`.
- **Comments explain *why*** (assumptions, reward semantics, pitfalls), not what
  the Python already says.
- **Seeds:** Python, NumPy, PyTorch, and data sampling where applicable; document
  any remaining nondeterminism.
- **Dependencies:** minimal and pinned where reproducibility matters; no framework
  for a small utility.
- **Compute proxy:** if token count is the proxy, name it as such — never call it
  `FLOPs` unless FLOPs are actually measured.

---

## RL diagnostics (once M4 starts)

Always log: fraction STOP, fraction CONTINUE, mean reasoning tokens, accuracy,
mean reward, reward components separately, and action distribution by reasoning
step. Enforce a max compute budget. Investigate any collapse toward ~100% STOP or
~100% CONTINUE — rising aggregate reward is **not** proof of learning. Check for
reward hacking (parser quirks, premature termination of malformed answers,
degenerate EOS use).
