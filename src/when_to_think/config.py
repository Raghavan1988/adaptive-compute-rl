"""Typed experiment configuration (AGENTS.md §9, §21).

Every experimental quantity is configurable here rather than hard-coded in source,
so a run is reproducible from a YAML file plus optional CLI overrides. The schema
is intentionally small: it covers what M0-M2 need (model, data, generation,
representation, reward). Probe/policy sub-configs are added when M3/M4 arrive,
rather than shipping dead fields now.

Design choices that encode research invariants:
- `RewardConfig.lambda_compute_sweep` is a LIST, never a scalar: `lambda` is never
  universal (AGENTS.md §7), so the schema forces a sweep.
- `compute_proxy` names what compute is measured in (default "reasoning_tokens").
  It must not be called "FLOPs" unless FLOPs are actually measured (AGENTS.md §7).
- Unknown config keys are rejected, not ignored, so a typo cannot silently change
  (or fail to change) an experiment.
- `val_fraction` is carved from the TRAIN split only; the test split is never
  touched for tuning (AGENTS.md §4.2).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Base SLM + tokenizer. The SLM is frozen in initial experiments (AGENTS.md §14)."""

    name: str
    # Pin `revision` for real runs so results are reproducible (AGENTS.md §9). Left
    # null in samples to avoid asserting a commit that may not exist locally.
    revision: str | None = None
    tokenizer_name: str | None = None  # defaults to `name` in __post_init__
    dtype: str = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = False
    frozen: bool = True

    def __post_init__(self) -> None:
        if self.tokenizer_name is None:
            self.tokenizer_name = self.name


@dataclass
class DataConfig:
    """Benchmark + split policy. Val is carved from train; test is held out."""

    dataset_name: str
    dataset_config: str | None = None
    train_split: str = "train"
    test_split: str = "test"
    # Validation is carved deterministically from the train split so the held-out
    # test split is never used for tuning (AGENTS.md §4.2).
    val_fraction: float = 0.1
    max_train_examples: int | None = None
    max_test_examples: int | None = None
    sampling_seed: int = 0


@dataclass
class GenerationConfig:
    """Fixed-budget and incremental generation settings.

    Budgets are in generated reasoning tokens. `fixed_budgets` is the M1 sweep;
    `max_reasoning_budget` is the hard cap the STOP/CONTINUE environment enforces
    so continuation cannot run forever (AGENTS.md §15).
    """

    max_reasoning_budget: int = 512
    reasoning_increment: int = 64  # tokens granted per CONTINUE
    decision_interval: int = 64  # tokens between STOP/CONTINUE decision points
    fixed_budgets: list[int] = field(default_factory=lambda: [0, 128, 256, 512])
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    # Counterfactual samples per (example, budget). >1 estimates P(correct | budget)
    # rather than a single draw (README Phase 1).
    num_samples: int = 1
    # Answer-elicitation protocol for fixed-budget runs (research-significant, §25).
    # After the reasoning budget, if the model has not already produced a clean final
    # answer, `answer_cue` is appended and up to `answer_max_tokens` are generated
    # (greedily) to force one. This makes "budget = 0 / direct answer" and truncated
    # budgets yield a scorable answer instead of a dangling reasoning trace. The cue
    # primes the '####' marker that answer extraction reads with highest precedence.
    answer_cue: str = "\n#### "
    answer_max_tokens: int = 24


@dataclass
class RepresentationConfig:
    """Selective hidden-state extraction (AGENTS.md §13).

    Record enough to reconstruct exactly which vector was probed: layer, token
    position, and pooling. Negative layer indices count from the last layer.
    """

    layers: list[int] = field(default_factory=lambda: [-1])
    token_position: str = "last"  # e.g. last generated reasoning token
    pooling: str | None = None  # None = raw token vector (no pooling)


@dataclass
class RewardConfig:
    """Reward = R_task - lambda * C (AGENTS.md §7).

    `lambda_compute_sweep` is a list on purpose: lambda is never universal, so the
    schema requires a sweep. Task reward and compute penalty stay separate fields
    so they can be logged independently (AGENTS.md §17).
    """

    task_reward_correct: float = 1.0
    task_reward_incorrect: float = 0.0
    lambda_compute_sweep: list[float] = field(
        default_factory=lambda: [0.0, 1e-4, 5e-4, 1e-3]
    )
    # Name the compute proxy honestly. Do NOT rename to "flops" unless FLOPs are
    # actually measured (AGENTS.md §7).
    compute_proxy: str = "reasoning_tokens"


@dataclass
class ProbeConfig:
    """Supervised value-of-compute probe (M3).

    The probe reads a frozen hidden state at a decision point and predicts the *value
    of continuing* — two explicit, distinct targets, evaluated separately:
    ``value_of_compute`` (Δ accuracy, regression) and ``fixes_incorrect`` (binary).
    These are research-significant definitions (AGENTS.md §25): changing them
    invalidates prior probe results.

    Selection discipline (AGENTS.md §4.2): ``layers`` and the regularization grids are
    tuned on the VAL split only; the test split is scored once for the chosen model.
    """

    # Candidate layers to probe; must be a subset of the layers stored by the run.
    layers: list[int] = field(default_factory=lambda: [-1])
    # "next" = value of one more budget increment; "max" = value of continuing to the cap.
    continue_mode: str = "next"
    # P(correct) threshold defining "incorrect" for the binary fixes-it target.
    correct_threshold: float = 0.5
    # Regularization grids searched on val (never on test).
    ridge_alphas: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    logreg_alphas: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    logreg_lr: float = 0.1
    logreg_max_iter: int = 500
    standardize: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment definition. `seed` seeds Python/NumPy/PyTorch/sampling."""

    name: str
    model: ModelConfig
    data: DataConfig
    seed: int = 0
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    output_dir: str = "results"
    notes: str = ""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_SCALAR_FIELDS = {"name", "seed", "output_dir", "notes"}


def _from_mapping(cls: type, mapping: dict[str, Any]):
    """Construct a flat dataclass from a dict, rejecting unknown keys.

    Rejecting unknown keys (rather than ignoring them) means a mistyped config key
    fails loudly instead of silently leaving an experiment quantity at its default.
    """
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(mapping).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(mapping) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)} (known: {sorted(known)})")
    return cls(**mapping)


def _resolve_refs(raw: dict[str, Any], configs_root: Path) -> dict[str, Any]:
    """Resolve string references for `model`/`data` into their sub-config files.

    An experiment file may either inline a sub-config as a dict, or name a file:
        model: qwen2.5-1.5b   ->   configs/model/qwen2.5-1.5b.yaml
        data: gsm8k           ->   configs/data/gsm8k.yaml
    """
    raw = copy.deepcopy(raw)
    for key, subdir in (("model", "model"), ("data", "data")):
        value = raw.get(key)
        if isinstance(value, str):
            ref_path = configs_root / subdir / f"{value}.yaml"
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Config '{key}: {value}' references missing file {ref_path}"
                )
            raw[key] = yaml.safe_load(ref_path.read_text())
    return raw


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `dotted.key=value` overrides to a raw config dict.

    Values are parsed as YAML so `512` -> int, `[0, 128]` -> list, `true` -> bool.
    Applied AFTER ref resolution so nested keys (e.g. `model.dtype`) can be set.
    """
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key, _, value = item.partition("=")
        parsed = yaml.safe_load(value)
        parts = key.split(".")
        node = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(
                    f"Cannot descend into non-mapping at {part!r} in override {item!r}"
                )
        node[parts[-1]] = parsed
    return raw


def _build_experiment(raw: dict[str, Any]) -> ExperimentConfig:
    raw = dict(raw)
    if "model" not in raw or "data" not in raw:
        raise ValueError("Experiment config must define both 'model' and 'data'")

    model = _from_mapping(ModelConfig, raw.pop("model"))
    data = _from_mapping(DataConfig, raw.pop("data"))
    generation = _from_mapping(GenerationConfig, raw.pop("generation", {}))
    representation = _from_mapping(RepresentationConfig, raw.pop("representation", {}))
    reward = _from_mapping(RewardConfig, raw.pop("reward", {}))
    probe = _from_mapping(ProbeConfig, raw.pop("probe", {}))

    unknown = set(raw) - _SCALAR_FIELDS
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    if "name" not in raw:
        raise ValueError("Experiment config must define 'name'")

    cfg = ExperimentConfig(
        model=model,
        data=data,
        generation=generation,
        representation=representation,
        reward=reward,
        probe=probe,
        **raw,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    """Fail fast on configs that violate research invariants or are internally broken."""
    g = cfg.generation
    if g.max_reasoning_budget <= 0:
        raise ValueError("generation.max_reasoning_budget must be positive")
    if g.reasoning_increment <= 0:
        raise ValueError("generation.reasoning_increment must be positive")
    if g.decision_interval <= 0:
        raise ValueError("generation.decision_interval must be positive")
    if any(b < 0 for b in g.fixed_budgets):
        raise ValueError("generation.fixed_budgets must be non-negative token counts")
    if max(g.fixed_budgets, default=0) > g.max_reasoning_budget:
        raise ValueError(
            "generation.fixed_budgets exceed max_reasoning_budget "
            f"({max(g.fixed_budgets)} > {g.max_reasoning_budget})"
        )
    if g.num_samples <= 0:
        raise ValueError("generation.num_samples must be positive")
    if g.answer_max_tokens <= 0:
        raise ValueError("generation.answer_max_tokens must be positive")

    d = cfg.data
    if not 0.0 <= d.val_fraction < 1.0:
        raise ValueError("data.val_fraction must be in [0, 1)")
    if d.train_split == d.test_split:
        raise ValueError("data.train_split and data.test_split must differ (no test-set training)")

    r = cfg.reward
    if not r.lambda_compute_sweep:
        raise ValueError(
            "reward.lambda_compute_sweep must be non-empty (lambda is never universal)"
        )
    if any(lam < 0 for lam in r.lambda_compute_sweep):
        raise ValueError("reward.lambda_compute_sweep values must be non-negative")

    p = cfg.probe
    if not p.layers:
        raise ValueError("probe.layers must be non-empty")
    if p.continue_mode not in ("next", "max"):
        raise ValueError("probe.continue_mode must be 'next' or 'max'")
    if not 0.0 < p.correct_threshold <= 1.0:
        raise ValueError("probe.correct_threshold must be in (0, 1]")
    if not p.ridge_alphas or any(a <= 0 for a in p.ridge_alphas):
        raise ValueError("probe.ridge_alphas must be non-empty and positive")
    if not p.logreg_alphas or any(a <= 0 for a in p.logreg_alphas):
        raise ValueError("probe.logreg_alphas must be non-empty and positive")
    if p.logreg_lr <= 0 or p.logreg_max_iter <= 0:
        raise ValueError("probe.logreg_lr and probe.logreg_max_iter must be positive")


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    configs_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load an experiment config from YAML, resolving refs and applying overrides.

    `configs_root` locates the `model/` and `data/` sub-config dirs; it defaults to
    the grandparent of `path` (i.e. `configs/` when the file is `configs/experiment/x.yaml`).
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    root = Path(configs_root) if configs_root is not None else path.parent.parent
    raw = _resolve_refs(raw, root)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _build_experiment(raw)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Fully-resolved config as a plain dict, for snapshotting into a run record."""
    return dataclasses.asdict(cfg)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write the resolved config to YAML (used when recording a run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))


# --------------------------------------------------------------------------- #
# CLI helpers (keep scripts thin: parse -> load_config -> call into package)
# --------------------------------------------------------------------------- #

def add_config_args(parser) -> None:
    """Register --config and repeatable --set KEY=VALUE on an argparse parser."""
    parser.add_argument("--config", required=True, type=Path, help="Path to experiment YAML")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set generation.max_reasoning_budget=256",
    )


def load_config_from_args(args) -> ExperimentConfig:
    """Build an ExperimentConfig from parsed args produced via `add_config_args`."""
    return load_config(args.config, overrides=args.overrides)
