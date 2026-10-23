# %% [markdown]
# Read every executed-and-not-pruned trial from an Optuna quantization study
# and plot the full Accuracy vs. compression-rate scatter - similar to the
# Pareto-front preview in optuna-dashboard, but rendered through this repo's
# own `src.view` (Figure/Report) plotting stack and exported as a PDF.
#
# Unlike `tune_quantization.py`'s own `{study_name}_pareto.yaml` export (which
# only contains the Pareto-optimal trials), this reads directly from the
# study's sqlite `.db` so it reflects every COMPLETE trial, with the Pareto
# front highlighted among them.
# %%
import optuna
import pandas as pd
from optuna.trial import TrialState

from src.view import Report
from src.view.pareto import DEFAULT_METHOD_LABELS, plot_pareto_front

nn = "resnet"
title = ("Resnet20 Pareto Front",)
DB_PATH = f"outputs/{nn}_paretofront4.db"
STUDY_NAME = None
OUTPUT_PDF = f"{nn}_pareto_front.pdf"


def load_non_pruned_trials(db_path: str, study_name: str | None = None) -> pd.DataFrame:
    """Read every executed-and-not-pruned (COMPLETE) trial straight from an
    Optuna sqlite study - independent of tune_quantization.py's own
    Pareto-only YAML export, so it reflects the full trial history.

    "Not pruned" == optuna.trial.TrialState.COMPLETE. PRUNED/FAIL/RUNNING
    trials either were deliberately stopped early or never produced valid
    objective values, so they're excluded here (same filter
    `study.best_trials` and tune_quantization.py's exact-reuse cache both
    already apply).
    """
    storage_url = f"sqlite:///{db_path}"
    if study_name is None:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        if len(summaries) != 1:
            names = [s.study_name for s in summaries]
            raise ValueError(
                f"Expected exactly one study in {db_path}, found {names}; "
                "set STUDY_NAME explicitly."
            )
        study_name = summaries[0].study_name

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    pareto_numbers = {t.number for t in study.best_trials}  # Optuna's own Pareto front

    rows = []
    for t in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]):
        rows.append(
            {
                "trial_number": t.number,
                "method": t.user_attrs.get("method", "unknown"),
                "val_acc": t.values[0],
                "compression_rate": t.values[1],
                "bpp": t.user_attrs.get("bpp", float("nan")),
                "is_pareto_optimal": t.number in pareto_numbers,
            }
        )
    return pd.DataFrame(rows)


# %%

df = load_non_pruned_trials(DB_PATH, STUDY_NAME)

# Presentation-only cleanup, kept out of `load_non_pruned_trials` so that reader
# stays a faithful raw dump of the study. Trials reused via tune_quantization.py's
# exact-param cache never re-run `trial.set_user_attr("method", ...)`, so their
# `method` falls back to "unknown" - drop them, since which method they actually
# used can't be recovered from this db. Relabel the raw method names to
# shorter, presentation-friendly names for the plot legend.
n_unknown = int((df["method"] == "unknown").sum())
if n_unknown:
    print(f"Dropping {n_unknown} trial(s) with method == 'unknown' (missing user_attr)")
df = df[df["method"] != "unknown"].copy()
df["method"] = df["method"].replace(DEFAULT_METHOD_LABELS)

n_opt = int(df["is_pareto_optimal"].sum())
print(f"Loaded {len(df)} non-pruned (COMPLETE) trials from {DB_PATH}")
print(f"  {n_opt} Pareto-optimal, {len(df) - n_opt} dominated")
print(df.sort_values("compression_rate").to_string(index=False))

fig = plot_pareto_front(
    df,
    x="compression_rate",
    y="val_acc",
    group="method",
    optimal="is_pareto_optimal",
    x_label=r"$r_\mathrm{\phi}$",
    y_label="Validation Accuracy",
    title=title,
)

report = Report()
report.append_figures([fig])
report.save(OUTPUT_PDF)
print(f"Saved {report.output_dir / OUTPUT_PDF}")
