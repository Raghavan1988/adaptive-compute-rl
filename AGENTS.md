# AGENTS.md

This file defines instructions for AI coding agents working in the `when-to-think` repository.

The project is a research codebase. Correct experimental methodology is more important than rapidly adding features.

---

## 1. Project Mission

The repository investigates one research question:

> **Can an RL policy conditioned on an SLM's internal representations learn when to stop or continue reasoning, achieving a better accuracy-compute trade-off than fixed reasoning budgets?**

All implementation decisions should make this question easier to answer cleanly.

Do not broaden the research question without an explicit human instruction.

---

## 2. Scope Guardrail

The initial action space is exactly:

```text
STOP
CONTINUE
```

Do not add the following to the core experiment unless explicitly requested:

- retrieval
- web search
- tools
- larger-model escalation
- multi-agent orchestration
- arbitrary environment actions

The project should first establish adaptive reasoning allocation in the two-action setting.

---

## 3. Research Sequence

Prefer work in this order:

1. deterministic task evaluation
2. fixed-budget inference
3. counterfactual dataset generation
4. oracle allocation
5. supervised hidden-state probes
6. non-RL adaptive baselines
7. RL stop/continue policy
8. representation and error analysis
9. optional extensions

Do not start by building a sophisticated RL stack before fixed-budget and oracle baselines work end-to-end.

---

## 4. Scientific Invariants

Agents must preserve the following invariants.

### 4.1 Same evaluation data

Methods being compared must use the same test examples unless the experiment explicitly studies distribution shift.

### 4.2 No test-set training

Do not train:

- probes
- policies
- thresholds
- calibration mappings
- hyperparameters

on the held-out test set.

### 4.3 Matched compute comparisons

When claiming one policy is better than another, compare methods at matched or explicitly reported compute.

Do not compare:

```text
adaptive policy at 150 tokens
```

to:

```text
fixed policy at 1024 tokens
```

and attribute the difference solely to routing.

### 4.4 Preserve failed trajectories

Do not silently remove:

- incorrect generations
- malformed answers
- cases where more reasoning hurts
- policy collapse
- timeouts

These are part of the result.

### 4.5 No monotonicity assumption

Do not assume:

```text
more tokens => higher correctness
```

The model may change a correct answer into an incorrect one.

### 4.6 Exact reward semantics

For automatically verifiable tasks, use deterministic rule-based reward whenever possible.

Do not replace exact-match evaluation with an LLM judge unless the experiment explicitly requires it.

---

## 5. Claims Discipline

Use precise language in code, comments, plots, and documentation.

A probe that predicts a label from a hidden state demonstrates **decodability**.

It does not, by itself, prove:

- causal mechanism
- introspection
- self-awareness
- metacognition
- that the base model naturally uses the decoded feature

Prefer terms such as:

- `hidden-state signal`
- `representation probe`
- `value-of-compute estimate`
- `adaptive compute policy`

Avoid stronger anthropomorphic claims without direct evidence.

---

## 6. Primary Experimental Quantity

The important quantity is not simply model confidence.

The project studies the **marginal value of additional computation**.

A useful conceptual target is:

\[
\Delta_t =
P(\text{correct} \mid \text{CONTINUE at } t)
-
P(\text{correct} \mid \text{STOP at } t)
\]

When constructing labels from sampled counterfactual runs, keep the empirical definition explicit.

Do not casually rename:

```text
probability current answer is correct
```

to:

```text
value of continuing
```

They are different quantities.

---

## 7. Reward Design

The default conceptual reward is:

\[
R = R_{\text{task}} - \lambda C
\]

where:

- `R_task` is correctness or another explicit task reward.
- `C` is compute.
- `lambda` is a configurable compute penalty.

Never hard-code one value of `lambda` as if it were universally correct.

Experiments should support sweeps over compute penalties.

If using token count as the compute proxy, name it clearly. Do not call token count `FLOPs` unless FLOPs are actually estimated or measured.

---

## 8. Baselines Are Required

Do not present the RL policy without the following categories of baselines.

### Fixed-budget

At least several fixed reasoning budgets.

### Simple adaptive

At minimum consider:

- random stopping at matched compute
- entropy-based rule
- verbalized confidence
- input-only difficulty predictor
- hidden-state correctness probe
- hidden-state value-of-compute probe

### Oracle

Use counterfactual fixed-budget outcomes to estimate an upper bound where possible.

A new training method is not useful merely because it beats the weakest baseline.

---

## 9. Reproducibility

Every experiment must be reproducible from configuration.

Prefer configuration files or command-line arguments over editing constants in source files.

Record:

```text
run_id
timestamp
git_commit
model_name
model_revision
tokenizer_name
dataset_name
dataset_split
seed
generation_config
reasoning_budget
decision_interval
reward_config
lambda_compute
training_config
```

Seed:

- Python
- NumPy
- PyTorch
- data sampling

where applicable.

When true determinism is impossible, document the source.

---

## 10. Code Organization

Keep reusable logic under:

```text
src/when_to_think/
```

Modules (✅ = implemented in M0; the rest arrive with their milestone):

```text
config.py         # ✅ typed experiment config: YAML composition + CLI overrides
data/             # ✅ dataset loading + disjoint splits (gsm8k.py)
models/           # ✅ frozen SLM + tokenizer loader (loader.py)
generation/       # ✅ budget-enforced reasoning generation (generate.py)
representations/  # ✅ selective hidden-state extraction + sharded storage
rewards/          # ✅ answer extraction + task reward / compute penalty
evaluation/       # ✅ scored, machine-readable evaluation (evaluate.py)
utils/            # ✅ seeding, run records
probes/           # ⬜ M3
policies/         # ⬜ M4
```

Scripts under `scripts/` should be thin entry points. See `PLAN.md` for milestone
status.

Bad:

```python
# scripts/train_policy.py
# 900 lines containing datasets, model wrappers, rewards, training, plots, and eval
```

Better:

```python
from when_to_think.policies import train_policy
from when_to_think.config import load_config

cfg = load_config(...)
train_policy(cfg)
```

---

## 11. Notebooks

Notebooks are allowed for:

- exploration
- plotting
- inspecting trajectories
- debugging small examples

They must not become the only implementation of:

- data generation
- reward computation
- training
- core evaluation
- metric calculation

If an experiment matters to the final result, move it into tested Python modules or scripts.

---

## 12. Testing Expectations

Add tests for research-critical behavior.

At minimum test:

### Answer extraction

Correctly parse valid and invalid benchmark outputs.

### Reward calculation

Verify:

- correct answer receives expected task reward
- incorrect answer receives expected task reward
- compute penalty is applied exactly once
- STOP and CONTINUE transitions behave as specified

### Budget enforcement

A generation configured for a given maximum reasoning budget must not silently exceed it.

### Oracle construction

Use toy trajectories where the expected oracle action is obvious.

Examples:

```text
STOP correct, CONTINUE correct
=> oracle should prefer STOP when CONTINUE costs more
```

```text
STOP wrong, CONTINUE correct
=> oracle should prefer CONTINUE if the correctness gain exceeds its cost
```

```text
STOP wrong, CONTINUE wrong
=> oracle should prefer the cheaper action
```

### Dataset split integrity

Prevent accidental overlap where feasible.

---

## 13. Hidden-State Extraction

When extracting representations, record:

- layer index
- token position
- reasoning step
- model revision
- pooling method, if any

Do not use vague names such as:

```text
model_features
```

Prefer:

```text
layer_16_last_token_residual
```

or a structured representation descriptor.

Avoid storing gigantic hidden-state dumps by default. Support selective layer extraction and streaming or sharded storage.

---

## 14. Probe Training

The base SLM should remain frozen during the initial probe experiments unless explicitly studying fine-tuning.

Start with simple models:

1. logistic / linear probe
2. small MLP only if justified

A more complex probe can increase decodability without demonstrating that the representation is clean or readily usable.

Report train/validation/test performance separately.

---

## 15. RL Implementation Guidance

RL is the main eventual training contribution, but simplicity matters.

The environment should expose a clear transition:

```text
state h_t
  |
policy
 / \
STOP CONTINUE
```

### STOP

- terminate the episode
- score the current final answer
- apply accumulated compute cost

### CONTINUE

- grant another fixed reasoning increment
- update the model state
- accrue compute cost
- present the next decision state

Always enforce a maximum compute budget to prevent infinite continuation.

Keep the policy architecture separable from the base SLM when possible so that comparisons are interpretable.

---

## 16. RL Collapse Diagnostics

Always log:

- fraction STOP
- fraction CONTINUE
- mean reasoning tokens
- accuracy
- mean reward
- reward components separately
- action distribution by reasoning step

Immediately investigate if the policy converges toward:

```text
~100% STOP
```

or:

```text
~100% CONTINUE until max budget
```

Do not interpret collapse as successful learning merely because aggregate reward increases.

---

## 17. Reward Hacking Checks

Whenever the learned policy improves reward, verify whether it also improves the intended task-quality/compute trade-off.

Check for:

- exploiting answer parser quirks
- prematurely terminating malformed answers
- generating formatting that receives accidental credit
- gaming length penalties
- dataset-specific shortcuts
- degenerate use of EOS or stop tokens

Keep task reward and compute penalty visible as separate logged fields.

---

## 18. Evaluation Output

Every major evaluation should produce a machine-readable result file.

Prefer JSONL, JSON, Parquet, or CSV with per-example data including:

```text
example_id
question
ground_truth
method
seed
reasoning_tokens
latency
prediction
correct
reward_task
reward_compute
reward_total
actions
```

Aggregate tables and figures should be generated from these files.

Do not manually type headline result numbers into plots.

---

## 19. Headline Figure

The project's most important figure is:

```text
Accuracy
   ^
   |
   |               adaptive / oracle
   |            *
   |        *
   |    *       fixed budgets
   |  o    o      o
   +--------------------------> Compute
```

The plotting code must be reproducible.

Whenever possible, include uncertainty across seeds or bootstrap confidence intervals.

---

## 20. Naming

Prefer names that describe scientific meaning.

Good:

```python
value_of_compute
continue_benefit
compute_penalty
fixed_budget
oracle_action
premature_stop_rate
```

Avoid:

```python
magic_score
quality
smart_router
confidence2
best_policy_final_v3
```

---

## 21. Configuration Over Constants

Experimental quantities must be configurable, including:

- model
- dataset
- maximum reasoning budget
- reasoning increment
- decision interval
- hidden-state layers
- policy architecture
- learning rate
- compute penalty
- number of rollouts
- random seed

A user should be able to rerun an experiment without editing source.

---

## 22. Performance Optimizations

Correctness comes before throughput.

When optimizing:

1. establish a reference implementation
2. add correctness tests
3. optimize
4. compare outputs against the reference

Useful optimizations may include:

- batched generation
- KV-cache reuse
- selective hidden-state extraction
- sharded rollouts
- mixed precision

Do not change generation semantics silently for speed.

---

## 23. Dependencies

Keep dependencies minimal.

Before adding a package:

- check whether an existing dependency already provides the functionality
- prefer widely used, maintained libraries
- pin versions where reproducibility matters

Do not introduce a full framework for a small utility function.

---

## 24. Comments and Documentation

Comments should explain:

- why an experimental decision exists
- assumptions
- non-obvious reward semantics
- potential scientific pitfalls

Do not write comments that merely translate Python into English.

Bad:

```python
# Increment count by one.
count += 1
```

Good:

```python
# Charge compute at the moment another reasoning block is requested so STOP
# never pays for tokens that were not generated.
compute_cost += block_tokens
```

---

## 25. Changes to Scientific Semantics

Treat changes to any of the following as research-significant:

- reward definition
- answer extraction
- task metric
- data split
- reasoning budget
- oracle definition
- hidden-state position
- baseline implementation
- policy action semantics

When modifying them:

1. call out the change explicitly
2. update tests
3. invalidate or clearly version old results if necessary
4. do not silently mix results generated under different semantics

---

## 26. What Not to Optimize For

Do not optimize the codebase for:

- impressive architecture diagrams
- maximum number of supported benchmarks
- maximum number of RL algorithms
- arbitrary agent capabilities
- a positive result at all costs

Optimize for:

- a clean experiment
- correct baselines
- reproducible evidence
- interpretable failures
- answering the research question

---

## 27. Definition of Done for a Research Change

A substantial experimental change is done only when:

- the implementation works end-to-end
- research-critical logic has tests
- configuration is recorded
- output is machine-readable
- relevant baselines can still run
- the README or experiment docs are updated when semantics change

---

## 28. Decision Rule for Agents

When unsure between:

> adding a sophisticated feature

and

> making the central comparison cleaner,

choose the cleaner comparison.

The best version of `when-to-think` is not the largest system. It is the smallest system that convincingly answers whether an SLM can **learn when additional reasoning is worth its cost**.
