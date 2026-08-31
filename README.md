# When to Think

**Learning adaptive test-time compute policies for small language models.**

Small language models (SLMs) commonly use the same reasoning budget for every query. This is inefficient: some questions can be answered immediately, while others benefit substantially from additional reasoning.

`when-to-think` studies whether an SLM can learn, from its own internal representations, **when additional reasoning is worth the compute cost**.

---

## Research Question

> **Can an RL policy conditioned on an SLM's internal representations learn when to stop or continue reasoning, achieving a better accuracy-compute trade-off than fixed reasoning budgets?**

This is the singular research question for the project.

The initial project deliberately focuses on only two actions:

- **STOP** — produce the final answer now.
- **CONTINUE** — spend another reasoning step or token budget before deciding again.

Retrieval, tool use, and escalation to a larger model are intentionally out of scope for the initial study. They may be added only after the core stop/continue hypothesis is established.

---

## Motivation

A fixed reasoning budget creates two kinds of waste:

1. **Overthinking easy problems**  
   The model spends tokens without improving correctness.

2. **Underthinking hard problems**  
   The model stops before additional reasoning could improve the answer.

The central hypothesis is that the model's hidden state contains information about the **marginal value of additional computation**.

Rather than asking only:

> "Is the model currently confident?"

we want to ask:

> "Will more reasoning improve the expected answer enough to justify its cost?"

That distinction is important. A model can be uncertain about both:

- a problem where more reasoning would help, and
- a problem where more reasoning would not help.

An adaptive compute policy should distinguish between them.

---

## Core Formulation

At reasoning step \(t\), the model has an internal representation:

\[
h_t
\]

A policy observes that representation and chooses:

\[
a_t \in \{\text{STOP}, \text{CONTINUE}\}
\]

The learned policy is:

\[
\pi_\theta(a_t \mid h_t)
\]

The objective is to maximize answer quality while penalizing unnecessary computation:

\[
R = R_{\text{task}} - \lambda C
\]

where:

- \(R_{\text{task}}\) is the task reward, initially exact-answer correctness.
- \(C\) is inference compute, initially measured using generated reasoning tokens or reasoning steps.
- \(\lambda\) controls the quality-versus-compute trade-off.

For an exact-match task, a simple reward is:

\[
R =
\mathbb{1}[\text{correct}]
-
\lambda \cdot \text{reasoning tokens}
\]

The project should report results across multiple values of \(\lambda\), not just one hand-picked operating point.

---

## Hypothesis

> Internal representations contain a predictive signal for the marginal value of additional reasoning, and an RL policy can exploit that signal to allocate test-time compute more efficiently than fixed-budget inference.

A successful system should move the **accuracy-compute Pareto frontier**, rather than merely reduce token usage by sacrificing accuracy.

---

## Project Scope

### Phase 1: Establish the phenomenon

Before training RL, determine whether additional reasoning has heterogeneous value across examples.

For every evaluation question, run the same base SLM with multiple fixed reasoning budgets, for example:

- 0 / direct answer
- 128 tokens
- 256 tokens
- 512 tokens
- 1024 tokens

Record:

- final answer
- correctness
- reasoning tokens
- latency
- hidden states at candidate decision points

This creates counterfactual evidence for whether additional computation helped each example.

### Phase 2: Build an oracle

Using the fixed-budget runs, construct an oracle that selects the cheapest reasoning budget that achieves a correct answer.

The oracle answers:

> If we knew the outcome of every possible reasoning budget ahead of time, how much compute could we save?

This gives an upper bound on the value of adaptive allocation.

If the oracle does not meaningfully outperform fixed budgets on the accuracy-compute frontier, the project hypothesis is weak and should be reconsidered before moving to RL.

### Phase 3: Probe the internal signal

Freeze the SLM.

Train lightweight predictors using hidden representations to estimate whether another reasoning step is valuable.

Useful targets include:

\[
\Delta_t =
P(\text{correct} \mid \text{continue at } t)
-
P(\text{correct} \mid \text{stop at } t)
\]

or a simpler binary label:

\[
y_t =
\mathbb{1}[\text{continuing changes an incorrect result into a correct result}]
\]

This phase is a diagnostic and strong supervised baseline. It is **not** the final contribution.

### Phase 4: Learn the RL policy

Train a stop/continue policy whose observation includes the SLM's internal representation.

The policy repeatedly decides:

```text
question
   |
   v
SLM reasoning
   |
   v
hidden state h_t
   |
   v
policy
 /   \
STOP  CONTINUE
 |       |
answer   more reasoning
```

The policy should learn to spend additional compute only when its expected benefit exceeds its cost.

### Phase 5: Analyze what the policy learned

After training, analyze:

- At which layers is value-of-compute most predictable?
- Does the RL policy correlate with the supervised value-of-compute probe?
- Does the policy stop because it predicts correctness, or because it predicts that further reasoning is unproductive?
- Which problem types receive additional compute?
- Where does the policy make expensive mistakes?

This makes internal representations an interpretability hook while keeping **RL-based adaptive inference** as the project's main contribution.

---

## Initial Tasks and Models

Start with automatically verifiable reasoning tasks.

Recommended initial benchmark:

- **GSM8K**

Then expand to one harder or distribution-shifted benchmark after the pipeline is stable.

Potential later benchmarks:

- MATH / competition-style mathematics
- arithmetic or symbolic reasoning datasets
- logic reasoning datasets

Avoid open-ended QA in the first experiment. Exact-match or rule-based rewards make the scientific result much easier to interpret.

For the SLM, prefer a model small enough that many counterfactual generations and RL rollouts are affordable.

The exact model should be configurable rather than hard-coded.

---

## Baselines

The project must compare against strong, simple alternatives.

### Fixed compute

- Direct answer / minimum budget
- Fixed 128-token reasoning
- Fixed 256-token reasoning
- Fixed 512-token reasoning
- Fixed 1024-token reasoning

### Adaptive non-RL baselines

- Random stopping matched for average compute
- Entropy-based stopping
- Verbalized confidence
- Input-only difficulty classifier
- Hidden-state correctness probe
- Hidden-state value-of-compute probe

### Upper bound

- Oracle budget allocation using counterfactual outcomes

The RL policy is valuable only if it improves on these baselines at comparable compute.

---

## Primary Evaluation

The headline result is **not a single accuracy number**.

The main artifact should be an:

## Accuracy vs. Compute Pareto Curve

For each method, report:

- task accuracy
- average generated reasoning tokens
- average number of reasoning steps
- latency
- stop rate
- compute-normalized performance

Plot:

\[
\text{Accuracy} \quad \text{vs.} \quad \text{Average reasoning compute}
\]

A successful method should dominate or improve upon fixed-budget baselines over meaningful parts of the frontier.

---

## Secondary Metrics

### Policy metrics

- STOP / CONTINUE accuracy relative to oracle action
- precision and recall for beneficial CONTINUE decisions
- unnecessary-continue rate
- premature-stop rate

### Calibration

If the policy predicts expected value or probability of benefit:

- Expected Calibration Error (ECE)
- Brier score
- reliability diagrams

### Representation analysis

- probe performance by layer
- probe performance by reasoning step
- correlation between predicted value and realized value
- cross-dataset generalization

### Systems metrics

- wall-clock latency
- tokens generated
- throughput where practical

---

## Key Ablations

At minimum, test:

1. **Internal state vs. output confidence**  
   Does the hidden representation contain useful information beyond entropy or verbalized confidence?

2. **Internal state vs. input-only features**  
   Is the policy exploiting model state rather than simply learning question difficulty?

3. **Correctness prediction vs. value-of-compute prediction**  
   Does explicitly predicting whether more reasoning helps outperform merely predicting whether the current answer is correct?

4. **Layer ablation**  
   Which model layers contain the strongest routing signal?

5. **Compute penalty sweep**  
   How does behavior change as \(\lambda\) varies?

6. **Decision interval**  
   Does deciding every 32/64/128 tokens change performance?

---

## Failure Modes to Watch

### Always continue

If compute is too cheap in the reward, the policy may learn to use the maximum budget on every question.

### Always stop

If compute is penalized too strongly, the policy may collapse to immediate answers.

### Reward hacking

The policy may optimize superficial properties correlated with the evaluator rather than genuinely better reasoning.

### Difficulty classification masquerading as introspection

A policy may infer difficulty from the question text without using meaningful information from internal representations.

Input-only controls are therefore required.

### No causal value from hidden states

A probe can decode information that the policy itself does not naturally use. Probe success should not automatically be described as a causal mechanism.

### Non-monotonic reasoning

More reasoning is not guaranteed to improve an answer. A model may move from correct to incorrect at a larger budget. Preserve these cases instead of assuming correctness is monotonic in compute.

---

## Research Integrity Rules

This project distinguishes clearly between:

- **decodability** — information can be predicted from hidden states,
- **policy usefulness** — that information improves compute allocation,
- **causality** — manipulating a representation changes model behavior.

A linear probe demonstrates decodability, not automatically mechanism or causality.

Likewise, a better adaptive policy demonstrates improved inference allocation, not necessarily that the model possesses human-like metacognition or self-awareness.

Avoid those stronger claims unless directly supported by experiments.

---

## Proposed Repository Structure

```text
when-to-think/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── configs/
│   ├── model/
│   ├── data/
│   └── experiment/
├── src/
│   └── when_to_think/
│       ├── data/
│       ├── models/
│       ├── generation/
│       ├── representations/
│       ├── probes/
│       ├── policies/
│       ├── rewards/
│       ├── evaluation/
│       └── utils/
├── scripts/
│   ├── generate_fixed_budgets.py
│   ├── build_oracle.py
│   ├── train_probe.py
│   ├── train_policy.py
│   └── evaluate.py
├── tests/
├── notebooks/
├── artifacts/
│   ├── figures/
│   └── tables/
└── results/
```

Notebooks should be used for exploration and visualization only. Core experimental logic belongs in `src/` and executable scripts.

---

## Experiment Record

Every run should preserve:

```text
run_id
git_commit
model_name
dataset
dataset_split
random_seed
reasoning_budget
decision_interval
reward_definition
lambda_compute
training_config
evaluation_metrics
```

Do not rely on notebook state or manually edited spreadsheets as the source of truth.

---

## Minimum Viable Research Result

The first complete version of the project is successful if it can answer all four questions:

1. **Does additional reasoning have heterogeneous value across examples?**
2. **Can hidden states predict that value better than simple confidence baselines?**
3. **Can a learned stop/continue policy exploit the signal?**
4. **Does the policy improve the accuracy-compute Pareto frontier over fixed reasoning budgets?**

Everything else is optional until those questions are answered.

---

## Milestones

### M0 — Infrastructure

- Load model and benchmark.
- Generate exact-match answers.
- Record token counts and latency.
- Extract selected hidden states.
- Add deterministic evaluation and tests.

### M1 — Counterfactual compute dataset

- Run multiple fixed budgets.
- Measure how often additional reasoning helps, hurts, or changes nothing.
- Produce the first accuracy-vs-compute plot.

### M2 — Oracle allocation

- Construct per-example optimal allocations.
- Plot the oracle Pareto frontier.
- Quantify the maximum available compute savings.

### M3 — Supervised value-of-compute probe

- Train hidden-state probes.
- Compare against entropy, verbal confidence, and input-only baselines.
- Run layer-wise analysis.

### M4 — RL adaptive policy

- Implement STOP / CONTINUE environment.
- Train the policy.
- Sweep compute penalties.
- Compare with all baselines at matched compute.

### M5 — Analysis

- Error taxonomy.
- Calibration.
- Layer and step ablations.
- Cross-dataset generalization.
- Final figures and research write-up.

### Stretch

Only after M0-M5 are solid:

- RETRIEVE action
- escalation to a larger model
- tool use
- multi-action policy
- causal interventions on value-of-compute representations

---

## Definition of Success

The strongest possible result would support a statement of the form:

> An SLM's internal representations predict the marginal benefit of additional reasoning. A policy trained on those representations learns to allocate test-time compute adaptively and achieves higher accuracy than fixed-budget inference at the same average compute, or equivalent accuracy with lower compute.

A negative result is also scientifically useful if carefully established, for example:

> Hidden-state probes predict correctness but do not reliably predict the marginal value of additional reasoning beyond input difficulty and output confidence.

The project should optimize for a trustworthy answer to the research question, not for forcing a positive result.

---

## Current Non-Goals

Until the core result is established, do **not** expand the project into:

- general-purpose agent routing
- RAG
- web search
- tool selection
- multi-model orchestration
- arbitrary external APIs
- large-model escalation
- broad mechanistic interpretability claims

Those extensions may be valuable later, but they weaken the initial experiment if introduced too early.

---

## License

Choose a license once model, dataset, and dependency licenses have been reviewed.
