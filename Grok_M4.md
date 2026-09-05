# Grok task — Run the real M4 STOP/CONTINUE policy with Qwen2.5-1.5B

**Objective:** Produce the first *scientific* M4 result — train the RL STOP/CONTINUE
policy on real Qwen2.5-1.5B trajectories from GSM8K, and answer **Questions 3 and 4:
can a learned stop/continue policy exploit the hidden-state signal, and does it improve
the accuracy-compute Pareto frontier over fixed budgets at matched compute?**

This is an execution + analysis task, not a build task. The M4 pipeline is implemented,
tested (126 tests pass), and verified end-to-end — **but only on synthetic separable
data**, where the policy beats fixed budgets *by construction*. Your job is to run it for
real and report what the data says, honestly, either way.

Read [`AGENTS.md`](./AGENTS.md), [`CLAUDE.md`](./CLAUDE.md), and the M4 section of
[`PLAN.md`](./PLAN.md) first. This depends on M1/M2/M3 having real verdicts: if the oracle
does not beat fixed budgets (M2) or the hidden state is not decodable (M3), **stop and
reconsider** before training a policy.

---

## Background (what M4 measures)

The policy decides, at each decision point, whether to STOP (take the current answer) or
CONTINUE (reason another increment), conditioned only on the frozen decision-point hidden
state (+ a progress feature). It is trained by REINFORCE to maximize `R = R_task − λ·C`,
one policy per `λ` in the sweep, and is compared against fixed budgets and the omniscient
oracle **at matched compute**.

**Environment (research-significant — read before running).** M4 is offline over coherent
trajectory checkpoints: one reasoning rollout per example to the budget cap, checkpointed
every `decision_interval` tokens, with the provisional answer scored at each. This is
faithful because the policy never changes what the SLM generates — CONTINUE just reveals
the next checkpoint of the same rollout. If you need a *fully online* rollout environment
(re-generating under the policy's own stochasticity), that is a larger build; flag it
rather than silently swapping it in.

---

## Preconditions

- Working dir: `/home/raghavan/adaptive-compute-rl`, base conda env (Python 3.11).
- Deps installed. **`numpy` must stay `<2`.**
- Hardware: 16 GB RTX 4090 Laptop GPU. Qwen2.5-1.5B in bf16 fits.
- Sanity check:
  ```bash
  python -m pytest -q          # expect 126 passed
  ruff check src scripts tests # expect clean
  ```

---

## Steps

### 1. Generate coherent trajectories on all splits

```bash
python scripts/generate_trajectories.py \
  --config configs/experiment/gsm8k_m4.yaml \
  --splits train,val,test \
  --set model.revision=<commit-hash> \
  --set data.max_train_examples=300
```

Each example gets one rollout to `max_reasoning_budget` (512), checkpointed every
`decision_interval` (64) tokens. Watch generation time — this is the expensive step
(K answer elicitations per example). Note the printed `run_id`.

### 2. Train + evaluate the policy across the lambda sweep

```bash
python scripts/train_policy.py \
  --run-dir results/<run_id> \
  --config configs/experiment/gsm8k_m4.yaml
```

Writes `policy_results.json`, per-episode `policy_episodes.jsonl`, and the headline
`policy_frontier.png`. Trains on TRAIN, evaluates greedily on TEST.

### 3. Report

From `policy_results.json` (never hand-typed numbers):

- the **headline frontier**: adaptive policy vs fixed budgets vs oracle on
  accuracy-vs-compute, with the adaptive points' bootstrap CIs;
- `best_accuracy_gain_at_matched_compute` and `adaptive_beats_fixed` — the Q4 verdict;
- **collapse diagnostics for every λ**: fraction STOP/CONTINUE, mean reasoning tokens,
  action distribution by step, and the `collapse` flags. A λ that collapsed to
  ~always-STOP or ~always-CONTINUE is a fixed budget in disguise — say so;
- reward components (`mean_reward_task` vs `mean_reward_compute`) separately (§17).

---

## Must hold (binding invariants)

- **No test-set training (§4.2).** The policy and its feature standardizer are fit on
  TRAIN trajectories; TEST is only rolled out. Do not train or select on test.
- **Matched compute (§4.1).** Report the policy against fixed/oracle at the *same*
  compute. Never credit the policy for spending more/less than a baseline.
- **Budget enforced (§15).** The environment forces STOP at the cap; do not raise the cap
  mid-run to chase accuracy.
- **Rising reward is not learning (§16).** Inspect collapse every run. If a λ collapses,
  the "improvement" is an artifact — report it, do not bury it.
- **Reward hacking (§17).** Keep `reward_task` and `reward_compute` separate; check the
  policy is not exploiting the parser (e.g. stopping on unparseable answers). Preserve
  malformed / wrong trajectories (§4.4).
- **Precise claims.** The policy conditioning on hidden states shows the signal is
  *exploitable*, not that the model is introspective (CLAUDE.md).

---

## Follow-up (not required for the first verdict)

- **Seeds / CIs across runs.** The headline curve currently bootstraps over examples for
  one seed. For the paper, repeat training across seeds and report seed variance too.
- **Val-based model selection.** VAL trajectories are generated but unused by the sweep;
  use them to pick iterations / early-stop if training is unstable.
- **Larger MLP policy.** `policy.hidden_sizes` accepts widths (e.g. `[64]`) if the linear
  policy underfits — justify the change (§14) and re-report.
