"""Training-dynamics + metrics report data prep for eval_pareto_cv.py's
CV-verified Pareto configs, decoupled from any specific WandB project.

Generalizes the tail-half of notebook/momos2d_analysis.py (fetch named
WandB runs -> per-epoch mean/std across a group ->
`Report.training_overview` / `Report.metrics_vs_accuracy`), which
momos_hierarchical_analysis.py, vfold_vit_analysis.py, and
swapping_analysis.py all duplicate near-identically, so both the resnet and
mlp pareto-cv notebooks (and any future one) call into the same functions
instead of re-copying that pipeline again.
"""

from collections import defaultdict
import math
import os
import re
import sys
from typing import Sequence
import warnings

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import wandb
from wandb.apis.public import Run

from .fetch_log import extract_columns, merge_dfs
from .pareto import DEFAULT_METHOD_LABELS

# `build_reference_backbone` needs `model.build_backbone`/`utils.quant_bits`,
# which are bare (non-`src.`-prefixed) imports - they only resolve when
# `src/` itself is on sys.path (true when a script under `src/` is run
# directly, e.g. `python src/tune_quantization.py`, since Python adds the
# script's own directory to sys.path[0]). This module is instead imported
# the package way (`from src.view.training_analysis import ...`, `src`'s
# *parent* on sys.path), so patch `src/` on before those imports run.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from model import build_backbone  # noqa: E402
from quantizers import count_total_blocks, k_from_capacity  # noqa: E402
from utils.quant_bits import compute_quantization_bits  # noqa: E402

# `method` itself may contain underscores (e.g. "hierarchical_momos2d",
# "fold_momos"), so the greedy `+` backtracks until it finds the trailing
# `_t<digits>_f<digits>` - unambiguous for every known method name.
_RUN_NAME_RE = re.compile(r"^(?P<method>[a-zA-Z0-9_]+)_t(?P<trial_number>\d+)_f(?P<fold>\d+)$")

# Same list as notebook/momos2d_analysis.py's MOMOS2D_METRICS (note this is
# "quant/num_changed_weights", not fetch_log.py's differently-spelled/stale
# "quant/changed_weights"), plus the two metrics that weren't logged yet
# when momos2d_analysis.py was written: qbdm_complexity ("QuBD") and
# weight_entropy ("Entropy"). Both are already special-cased in
# Report._NORMALIZE_BY_MAX (src/view/report.py) so no report.py change is
# needed for them to normalize correctly.
PARETO_CV_METRICS: list[str] = [
    "metrics/bdm_complexity",
    "metrics/gzip_compression_rate",
    "metrics/bz2_compression_rate",
    "metrics/lzma_compression_rate",
    "metrics/weight_l2",
    "metrics/sparsity",
    "metrics/qbdm_complexity",  # QuBD
    "metrics/weight_entropy",  # Entropy
    "quant/distortion",
    "quant/num_changed_weights",
    "val/loss",
    "val/acc",
    "train/loss",
    "train/acc",
]

# Display titles for Report.metrics_vs_accuracy, shared by both notebooks.
PARETO_CV_METRICS_OVERVIEW: dict[str, str] = {
    "metrics/bdm_complexity": r"$\mathcal{\hat{K}}_\mathrm{BDM}$",
    "metrics/gzip_compression_rate": r"$\mathcal{r}_{Gzip}$",
    "metrics/bz2_compression_rate": r"$\mathcal{r}_\mathrm{BZ2}$",
    "metrics/lzma_compression_rate": r"$\mathcal{r}_\mathrm{LZMA}$",
    "metrics/weight_l2": r"$L_2$ Norm",
    "metrics/sparsity": "Sparsity",
    "metrics/qbdm_complexity": r"$\mathcal{\hat{K}}_\mathrm{QuBD}$",
    "metrics/weight_entropy": "Entropy",
    "quant/distortion": "Distortion",
    "quant/num_changed_weights": "# Changed Weights",
}


def load_pareto_cv_results(path: str) -> pd.DataFrame:
    """Read a `pareto_cv_results` YAML written by `src/eval_pareto_cv.py`
    (`_write_results`) into a flat DataFrame, one row per unique config.

    Dict-valued fields (`params`, `cv_metrics_mean`, `cv_metrics_std`,
    `per_fold`, ...) are kept as-is (object dtype); callers here only need
    `method`, `trial_number`, `compression_rate`, and the
    `cv_*_mean`/`cv_*_std` scalar columns.

    Only useful if `eval_pareto_cv.py` was actually run to completion with
    local disk access to write this file. If everything was logged straight
    to WandB instead (no local YAML ever produced), use
    `load_pareto_cv_results_from_wandb` instead - it reconstructs the same
    per-config summary directly from the project's per-fold runs.
    """
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    results = data.get("pareto_cv_results") or []
    if not results:
        raise ValueError(f"No 'pareto_cv_results' entries found in {path}")
    return pd.DataFrame(results)


def pareto_cv_run_names(method: str, trial_number: int, n_folds: int = 5) -> list[str]:
    """Reconstruct the exact WandB `display_name`s eval_pareto_cv.py logs for
    one config's folds (`src/eval_pareto_cv.py::_run_fold`):
    `f"{method}_t{trial_number}_f{fold_idx}"` for `fold_idx in range(n_folds)`.

    `method` must be the raw method string from the CV-results YAML's
    `method` field (e.g. "momos2d", "qat", "hierarchical_momos2d", "none")
    - never a display label from `pareto.DEFAULT_METHOD_LABELS` (e.g.
    "MoMos"), which only exists for plot/report legends and was never used
    to name a WandB run.
    """
    return [f"{method}_t{trial_number}_f{fold_idx}" for fold_idx in range(n_folds)]


def parse_pareto_cv_run_name(name: str) -> tuple[str, int, int] | None:
    """Reverse of `pareto_cv_run_names`: parse a WandB run's display name
    back into `(method, trial_number, fold_idx)`, or `None` if it doesn't
    match `eval_pareto_cv.py`'s `f"{method}_t{trial_number}_f{fold_idx}"`
    convention (e.g. an unrelated run logged into the same project).
    """
    m = _RUN_NAME_RE.match(name)
    if m is None:
        return None
    return m.group("method"), int(m.group("trial_number")), int(m.group("fold"))


def build_reference_backbone(model_name: str, dataset_name: str = "cifar10"):
    """Build a bare reference backbone for `compute_quantization_bits`'s
    `model` argument - the theoretical bit-accounting only needs the
    architecture (parameter shapes), not trained weights.

    Reads `src/configs/dataset/{dataset_name}.yaml` and
    `src/configs/model/{model_name}.yaml` directly rather than going through
    a full Hydra compose, so this has no dependency on the caller's own
    Hydra/CLI state.

    The model group is loaded from disk rather than synthesised as
    `{"name": model_name}`: timm-backed backbones need `model_id` (tinyvit ->
    `tiny_vit_5m_224`), which only lives in that yaml, so the synthesised
    version raised `ConfigAttributeError: Missing key model_id` for every
    model but mlp/resnet20. Those two carry just `name` plus training
    hyperparameters `build_backbone` ignores, so loading the file is a strict
    superset of the old behaviour. Falls back to the synthesised config if no
    such yaml exists.
    """
    dataset_cfg = OmegaConf.load(os.path.join(_SRC_DIR, "configs", "dataset", f"{dataset_name}.yaml"))
    model_path = os.path.join(_SRC_DIR, "configs", "model", f"{model_name}.yaml")
    model_cfg = (
        OmegaConf.load(model_path)
        if os.path.exists(model_path)
        else OmegaConf.create({"name": model_name})
    )
    return build_backbone(
        model_cfg,
        in_channels=dataset_cfg.in_channels,
        img_size=dataset_cfg.img_size,
        num_classes=dataset_cfg.num_classes,
    )


def format_group_label(method: str, quant_cfg: dict, backbone) -> str:
    """Compact, method-specific detail-run group label, so a group's
    hyperparameters are visible at a glance instead of just a bare method
    name:

        qat:                              "QAT{q}"                                    e.g. "QAT4"
        momos2d:                          "(rows, cols)\\n$k=..$"                     e.g. "(2, 4)\\n$k=12$"
        hierarchical_momos2d/fold_momos:  "(r, c)\\n$k=..$\\n(r', c')\\n$k'=..$"      e.g. "(2, 4)\\n$k=12$\\n(16, 32)\\n$k'=200$"

    Every piece gets its own line - shape, then count, per block - so
    nothing crowds/clashes even with a method-name line prepended on top
    (as `build_compression_breakdown` does for its bar labels).

    `k` is the *realized* integer size of the primary codebook
    (`quantizers.k_from_capacity(backbone, rows * cols, capacity)`) - not the
    raw `capacity` fraction - directly comparable across configs regardless
    of block shape. `k'` is the number of those motifs each *fold* keeps.
    This matches the notation notebook/complexity.py's
    `fold_momos_storage` is written in, where `v = r' * c'` is the fold
    count and `k'` the per-fold codebook; the secondary `(r', c')` line
    already carries `v`, so spending the second number on `k'` is what makes
    the label say something the shape does not. momos2d only ever has one
    block, so there's nothing to split beyond shape/count.

    Any other method (e.g. the dense "none" baseline, which has no
    rows/cols/capacity to show) falls back to just its method label
    (`DEFAULT_METHOD_LABELS`) - there's only ever one baseline config, so a
    trial-number disambiguator isn't needed.
    """
    if method == "qat":
        return f"QAT{int(quant_cfg['q'])}"

    if method == "momos2d":
        rows, cols = int(quant_cfg["rows"]), int(quant_cfg["cols"])
        k = k_from_capacity(backbone, rows * cols, float(quant_cfg["capacity"]))
        return f"({rows}, {cols})\n$k={k}$"

    if method in ("hierarchical_momos2d", "fold_momos"):
        primary, secondary = quant_cfg["primary"], quant_cfg["secondary"]
        p_rows, p_cols = int(primary["rows"]), int(primary["cols"])
        s_rows, s_cols = int(secondary["rows"]), int(secondary["cols"])
        k_p = k_from_capacity(backbone, p_rows * p_cols, float(primary["capacity"]))
        # `k'` is the per-fold codebook size the bit budget is actually
        # built from - `utils/quant_bits.py`'s `k_per_bucket`, a fraction of
        # the *primary* codebook clamped to the fold count. It is
        # deliberately NOT
        # `k_from_capacity(backbone, s_rows * s_cols, secondary.capacity)`:
        # that counts secondary-shaped blocks across the whole model, a
        # quantity no term of the hierarchical/fold_momos bit budget uses
        # (it overstates it ~15x on a 5M-param backbone), so the label would
        # contradict the compression-breakdown chart drawn beside it.
        n_folds = math.ceil(
            count_total_blocks(backbone, p_rows * p_cols) / (s_rows * s_cols)
        )
        k_prime = max(1, min(int(round(k_p * float(secondary["capacity"]))), n_folds))
        return f"({p_rows}, {p_cols})\n$k={k_p}$\n({s_rows}, {s_cols})\n$k'={k_prime}$"

    return DEFAULT_METHOD_LABELS.get(method, method)


def load_pareto_cv_results_from_wandb(
    entity: str,
    project: str,
    model_name: str,
    dataset_name: str = "cifar10",
    api: "wandb.Api | None" = None,
) -> pd.DataFrame:
    """Reconstruct the same per-config CV summary `load_pareto_cv_results`
    would read from a local YAML - but directly from `entity/project`'s
    per-fold runs, for when everything was logged straight to WandB and no
    local `eval_pareto_cv.py` output file exists at all.

    `eval_pareto_cv.py` never writes a combined per-config summary object to
    WandB itself - only one run per (method, trial_number, fold). So this:
    1. Fetches every run in the project and groups them by (method,
       trial_number), parsed from each run's display name via
       `parse_pareto_cv_run_name` (runs that don't match the naming
       convention are skipped, with a warning - e.g. unrelated runs sharing
       the project).
    2. Takes each fold-run's `val/acc` from `run.summary` (the same
       final-epoch value `eval_pareto_cv.py`'s own `res["val_acc"]` would
       have reported - cheaper than pulling full history just to build this
       summary, since only the detail-run step below needs per-epoch curves).
    3. Recomputes `compression_rate`/`bpp` via `compute_quantization_bits`
       against a freshly-built reference backbone (`build_reference_backbone`)
       and `run.config["quantization"]` - the fully-resolved per-trial
       quantization config, logged verbatim by `src/train.py`'s
       `logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))`
       call, so no Optuna search-space reconstruction is needed here at all.

    Returns a DataFrame with the same columns `load_pareto_cv_results`'s
    callers rely on: `trial_number`, `method`, `n_folds_completed`,
    `cv_val_acc_mean`, `cv_val_acc_std`, `compression_rate`, `bpp`, plus
    `quant_cfg` (the raw resolved quantization dict, e.g. for
    `format_group_label`).
    """
    api = api or wandb.Api()
    backbone_sample = build_reference_backbone(model_name, dataset_name)

    grouped: dict[tuple[str, int], list[Run]] = defaultdict(list)
    n_skipped = 0
    for run in api.runs(f"{entity}/{project}"):
        parsed = parse_pareto_cv_run_name(run.name)
        if parsed is None:
            n_skipped += 1
            continue
        method, trial_number, _fold = parsed
        grouped[(method, trial_number)].append(run)
    if n_skipped:
        warnings.warn(
            f"{n_skipped} run(s) in {entity}/{project} don't match eval_pareto_cv.py's "
            "naming convention - skipped."
        )

    rows = []
    for (method, trial_number), runs in grouped.items():
        val_accs = [run.summary.get("val/acc") for run in runs]
        val_accs = [float(v) for v in val_accs if v is not None]
        if not val_accs:
            warnings.warn(
                f"{method!r} trial {trial_number}: no run has a 'val/acc' summary value - skipping."
            )
            continue

        quant_cfg = dict(runs[0].config.get("quantization") or {})
        bit_stats = compute_quantization_bits(backbone_sample, quant_cfg)

        rows.append(
            {
                "trial_number": trial_number,
                "method": method,
                "n_folds_completed": len(val_accs),
                "cv_val_acc_mean": float(np.mean(val_accs)),
                "cv_val_acc_std": float(np.std(val_accs)),
                "compression_rate": float(bit_stats["compression_rate"]),
                "bpp": float(bit_stats["bpp"]),
                "quant_cfg": quant_cfg,
            }
        )

    if not rows:
        raise ValueError(
            f"No runs in {entity}/{project} matched eval_pareto_cv.py's naming convention "
            "with a usable 'val/acc' summary."
        )
    return pd.DataFrame(rows)


# Methods `compute_quantization_bits` returns a motif/index-or-mosaic/iota
# component split for - excludes "qat" (a single scalar q_bits, no split at
# all) and the dense "none" baseline (no compression to break down).
MOMOS_FAMILY_METHODS: frozenset[str] = frozenset(
    {"momos2d", "static_momos2d", "momos", "hierarchical_momos2d", "fold_momos"}
)

# compute_quantization_bits' raw "*_bits" keys -> one shared display name.
# momos2d/fold_momos's single index layer and hierarchical_momos2d's outer
# index layer both read as "Mosaic"; "Folds" is hierarchical_momos2d's extra
# per-fold index layer, absent for momos2d/fold_momos (src/utils/quant_bits.py).
_COMPONENT_DISPLAY_NAMES: dict[str, str] = {
    "motif_bits": "Motifs",
    "index_bits": "Mosaic",
    "mosaic_bits": "Mosaic",
    "iota_bits": "Folds",
}


def build_compression_breakdown(df: pd.DataFrame, backbone) -> pd.DataFrame:
    """Long-form (config, component) breakdown of each Pareto-optimal
    MoMos-family config's compressed size, for
    `pareto.plot_compression_breakdown`.

    Filters `df` (already carrying `method`, `quant_cfg`, `is_pareto_optimal`
    columns, as built by both pareto-cv notebooks) to Pareto-optimal rows in
    `MOMOS_FAMILY_METHODS` - qat has no component split, the dense "none"
    baseline has no compression, and dominated runs aren't part of this
    front's trade-off story. Calls the same `compute_quantization_bits` used
    to populate `compression_rate`/`bpp` in the first place, so results stay
    self-consistent with the rest of the report.

    `bar_label` prepends the method's display name to `format_group_label`'s
    output (rather than changing that shared, separately-tested function)
    specifically because hierarchical_momos2d and fold_momos render
    identically there for the same primary/secondary shape - this is the
    one place that distinction actually matters.

    Returns an empty DataFrame (columns `method, trial_number, bar_label,
    component, bits_fraction`) - never raises - if nothing survives the
    filter, so callers just check `.empty`.
    """
    columns = ["method", "trial_number", "bar_label", "component", "bits_fraction"]
    momos_df = df[df["is_pareto_optimal"] & df["method"].isin(MOMOS_FAMILY_METHODS)]
    if momos_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for row in momos_df.itertuples():
        stats = compute_quantization_bits(backbone, row.quant_cfg)
        dense_bits = stats["dense_bits"]
        bar_label = (
            f"{DEFAULT_METHOD_LABELS.get(row.method, row.method)}\n"
            f"{format_group_label(row.method, row.quant_cfg, backbone)}"
        )
        for key, display_name in _COMPONENT_DISPLAY_NAMES.items():
            if key not in stats:
                continue
            rows.append(
                {
                    "method": row.method,
                    "trial_number": int(row.trial_number),
                    "bar_label": bar_label,
                    "component": display_name,
                    "bits_fraction": stats[key] / dense_bits,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def fetch_named_runs(
    entity: str,
    project: str,
    run_names: Sequence[str],
    api: "wandb.Api | None" = None,
) -> dict[str, list[Run]]:
    """Fetch each of `run_names` from `entity/project` by exact
    `display_name`, one `api.runs(..., filters={"display_name": ...})` call
    per name (the same idiom every existing notebook/*_analysis.py uses for
    its "detail runs"). Tolerant of a run not existing yet (e.g. a
    still-running WandB project) or of a transient API error: each
    missing/erroring name is `warnings.warn`-ed and skipped, never raised,
    so a partially-populated project never aborts the report.

    Returns `{requested_name: [Run, ...]}` holding *every* run that matched
    each name, not just the first. A WandB `display_name` is not unique,
    and some sweeps in this repo deliberately reuse one name across folds -
    scripts/fold-momos-sweep-tinyvit.sh passes `wandb.name="cap${s_cap}"`
    for all 5 seeds/folds, so `cap0.25` is 5 distinct runs. Keeping only
    `matches[0]` silently collapsed such a group to a single run, which is
    what made every mean +/- std band in notebook/vfold_vit_analysis.py's
    report flat.

    A name that resolves to more than one run is warned about so the
    ambiguity stays visible: it is expected for a fold-sharing sweep, but a
    bug for `pareto_cv_run_names`' fold-unique names, where it means one
    fold was logged twice and would now be counted twice.

    Names that matched nothing are absent from the result, so `len(result)`
    counts *names resolved* - never runs returned.
    """
    api = api or wandb.Api()
    found: dict[str, list[Run]] = {}
    for name in run_names:
        try:
            matches = list(api.runs(f"{entity}/{project}", filters={"display_name": name}))
            if not matches:
                warnings.warn(f"No WandB run named {name!r} in {entity}/{project} - skipping.")
                continue
            if len(matches) > 1:
                warnings.warn(
                    f"{len(matches)} WandB runs share the display name {name!r} in "
                    f"{entity}/{project} - keeping all of them."
                )
            found[name] = matches
        except Exception as exc:  # noqa: BLE001 - network/API hiccups must not abort the report
            warnings.warn(
                f"Failed to fetch WandB run {name!r} from {entity}/{project}: {exc} - skipping."
            )
    return found


def fetch_grouped_runs(
    entity: str,
    project: str,
    groups: dict[str, Sequence[str]],
    api: "wandb.Api | None" = None,
) -> dict[str, list[Run]]:
    """Resolve `{group_label: [exact_run_name, ...]}` to `{group_label:
    [Run, ...]}` via `fetch_named_runs`, flattening its per-name lists so a
    group holds every run any of its names matched. A group that resolves
    to zero runs (e.g. every fold of a still-running config is missing) is
    dropped entirely, with a warning, rather than propagated as an empty
    entry - `build_runs_summary` assumes every group it receives has at
    least one run.

    The `n/N` counts in the warnings are *names* resolved, not runs: one
    name can legitimately match several runs (see `fetch_named_runs`), so a
    group may well hold more runs than it was given names.

    A group whose value is a bare `str` is treated as that one name. `str`
    is itself a `Sequence[str]`, so without this it would be iterated
    character by character and every character looked up as a run name.
    """
    api = api or wandb.Api()
    grouped: dict[str, list[Run]] = {}
    for label, names in groups.items():
        names = [names] if isinstance(names, str) else list(names)
        found = fetch_named_runs(entity, project, names, api=api)
        if not found:
            warnings.warn(
                f"Group {label!r}: 0/{len(names)} WandB run names resolved - dropping from report."
            )
            continue
        if len(found) < len(names):
            warnings.warn(
                f"Group {label!r}: only {len(found)}/{len(names)} WandB run names resolved."
            )
        grouped[label] = [run for runs in found.values() for run in runs]
    return grouped


def _run_metric_history(run: Run, metrics: Sequence[str], samples: int = 500) -> pd.DataFrame:
    """One run's epoch-indexed metric history: `run.history(samples=...)` ->
    `extract_columns` -> `merge_dfs` (the same three calls the `detail_runs`
    loop in notebook/momos2d_analysis.py makes per run), lifted into a
    single-run unit so `build_runs_summary` can call it per run. Returns an
    empty DataFrame (never raises) if the run has no history yet.

    Only requests columns this run's history actually has. Several metrics
    are method-specific (e.g. `quant/distortion`/`quant/num_changed_weights`
    are only logged by the MoMos-family projection callback -
    `src/utils/callbacks.py`'s `QuantizationCallback` - never by QAT or the
    "none" baseline), and `extract_columns` -> `df[["epoch", column]]` raises
    a hard `KeyError` for a column the DataFrame doesn't have at all.
    `aggregate_metric_across_runs` already tolerates a metric missing from a
    given frame; this is what keeps it from ever seeing one that's missing
    from the frame's *columns* in the first place.
    """
    history = run.history(samples=samples)
    if history.empty:
        return history
    available = [m for m in metrics if m in history.columns]
    frames = [history[["epoch"]].dropna().drop_duplicates()]
    frames += extract_columns(history, available)
    return merge_dfs(frames)


def aggregate_metric_across_runs(
    run_frames: Sequence[pd.DataFrame], metrics: Sequence[str]
) -> dict[str, tuple]:
    """Per-epoch mean/std of each metric across a group of runs' merged
    frames (from `_run_metric_history`), e.g. the fold-runs of one
    CV-verified config. Mirrors the `metrics_by_epoch` bucketing loop
    duplicated across every existing notebook/*_analysis.py: buckets every
    (epoch, value) pair by `round(epoch, 4)` (folds rarely land on exactly
    the same float epoch value, so rounding is what lets them merge into one
    bucket), then reduces each bucket to (mean, population-std).

    A metric absent from every frame gets `(array([]), array([]),
    array([]))` so `Report.metrics_vs_accuracy`'s emptiness check
    (`len(data[m][0]) == 0`) skips it cleanly instead of a KeyError.
    """
    metrics_by_epoch: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for frame in run_frames:
        for metric in metrics:
            if metric not in frame.columns:
                continue
            clean = frame[["epoch", metric]].dropna()
            for epoch, val in zip(clean["epoch"], clean[metric]):
                metrics_by_epoch[metric][round(epoch, 4)].append(val)

    result: dict[str, tuple] = {}
    for metric in metrics:
        bucket = metrics_by_epoch.get(metric)
        if not bucket:
            result[metric] = (np.array([]), np.array([]), np.array([]))
            continue
        sorted_epochs = sorted(bucket)
        means = [np.mean(bucket[e]) for e in sorted_epochs]
        stds = [np.std(bucket[e]) if len(bucket[e]) > 1 else 0.0 for e in sorted_epochs]
        result[metric] = (np.array(sorted_epochs), np.array(means), np.array(stds))
    return result


def build_runs_summary(
    grouped_runs: dict[str, list[Run]],
    metrics: Sequence[str],
    samples: int = 500,
) -> dict[str, dict[str, tuple]]:
    """`{group_label: [Run, ...]}` -> `{group_label: {metric: (epochs,
    means, stds)}}`, exactly the `run_data` shape `Report.training_overview`
    and `Report.metrics_vs_accuracy` require. Fetches + merges each run's
    history via `_run_metric_history`, then reduces per group via
    `aggregate_metric_across_runs`. A group whose runs all have empty
    histories (e.g. training just started) is warned about and dropped.
    """
    runs_summary: dict[str, dict[str, tuple]] = {}
    for label, runs in grouped_runs.items():
        frames = [
            f for f in (_run_metric_history(r, metrics, samples=samples) for r in runs) if not f.empty
        ]
        if not frames:
            warnings.warn(f"Group {label!r}: every run's history is empty - dropping from report.")
            continue
        runs_summary[label] = aggregate_metric_across_runs(frames, metrics)
    return runs_summary
