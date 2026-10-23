# %% [markdown]
# %%
import wandb
from src.view import Report
from src.view.weight_distribution import plot_weights_2d
import os
from tqdm import trange

project = "momos2d-collapse"  # "momos-reproduction"
api = wandb.Api()
runs = [
    ("model-pw0ti3vz", "momos"),
    ("model-qcci4krm", "momos"),
    # "model-y4jvu3xv",
    ("model-2fieryvz", "vfold"),
]
_config_cache = {}
for run_name, method in runs:
    report = Report()
    rows = cols = cap = None
    for i in trange(0, 20):
        # Deterministic local path so we can check for an existing download *before*
        # touching the network, instead of relying on wandb's default cache directory.
        root = os.path.join("artifacts", f"{run_name}-v{i}")
        ckpt_path = os.path.join(root, "model.ckpt")
        need_artifact = run_name not in _config_cache or not os.path.exists(ckpt_path)

        if need_artifact:
            artifact = api.artifact(
                f"danesinoo-university-of-copenhagen/momos-collapse/{run_name}:v{i}"
            )
            if run_name not in _config_cache:
                q_cfg = artifact.logged_by().config.get("quantization", {})
                _config_cache[run_name] = q_cfg
            if not os.path.exists(ckpt_path):
                artifact.download(root=root)

        q_cfg = _config_cache[run_name]

        if q_cfg["method"] == "momos2d":
            rows, cols, cap = (
                q_cfg["rows"],
                q_cfg["cols"],
                q_cfg["capacity"],
            )
        elif q_cfg["method"] == "hierarchical_momos2d":
            rows, cols, cap = (
                q_cfg["primary"]["rows"],
                q_cfg["primary"]["cols"],
                q_cfg["primary"]["capacity"],
            )
        if os.path.exists(ckpt_path):
            report.append_figures(plot_weights_2d(ckpt_path, rows, cols, i * 20))
        for figure in report.figures:
            figure.fontsize = 20

    if rows is not None:
        report.save(f"{method}_wa_r{rows}_c{cols}_cap{cap}.pdf")
