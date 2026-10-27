"""Weight-analysis report for the soft-MoMos MLP training-dynamics runs.

Mirrors notebook/weight_analysis.py but reads the local periodic checkpoints
written by src/train.py (periodic_checkpoint=true) instead of downloading wandb
artifacts.

It discovers the run dirs under outputs/cifar10_mlp/ that contain periodic
epoch checkpoints (checkpoints/epoch-epoch=N.ckpt), sorted by directory mtime.
The two newest such dirs are the soft-MoMos training-dynamics runs, in launch
order (first run = rows=1, cols=2; second run = rows=2, cols=1, matching
scripts/soft-momos-training-dynamics.sh). Once BOTH runs have reached their
final periodic checkpoint (internal epoch 399 == 400 training epochs), one
multipage PDF per run is built with plot_weights_2d for every 20-epoch
checkpoint.

stdout carries the delivery payload (lines starting with "PDF:"/"MEDIA:");
progress/incomplete diagnostics go to stderr, where the cron watchdog logs them.
"""

import matplotlib

matplotlib.use("Agg")

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.view.report import Report
from src.view.weight_distribution import plot_weights_2d

CHECKPOINT_ROOT = ROOT / "outputs" / "cifar10_mlp"
OUTPUT_DIR = ROOT / "assets"
# Launch order of scripts/soft-momos-training-dynamics.sh
SHAPES = [(1, 2), (2, 1)]
N_CHECKPOINTS = 400 // 20

EPOCH_RE = re.compile(r"epoch=(\d+)\.ckpt$")


def _epoch_of(path: Path) -> int:
    m = EPOCH_RE.search(path.name)
    assert m is not None, f"unexpected checkpoint name: {path.name}"
    return int(m.group(1))


def find_run_dirs() -> list[tuple[Path, list[Path]]]:
    """Run dirs containing periodic checkpoints, oldest-modified first."""
    out = []
    for d in CHECKPOINT_ROOT.iterdir():
        ckpts = sorted(d.glob("checkpoints/epoch-epoch=*.ckpt"))
        if ckpts:
            out.append((d.stat().st_mtime, d, ckpts))
    out.sort(key=lambda t: t[0])
    return [(d, ckpts) for _, d, ckpts in out]


def main() -> int:
    runs = find_run_dirs()
    if not runs:
        print("STATUS: no periodic checkpoints found yet.", file=sys.stderr)
        return 0

    incomplete = []
    for d, ckpts in runs:
        epochs = [_epoch_of(p) + 1 for p in ckpts]
        done = any(e == 400 for e in epochs)
        print(
            f"STATUS: {d.name}: {len(ckpts)}/{N_CHECKPOINTS} checkpoints, "
            f"latest epoch {max(epochs)}/400" + ("" if done else " (incomplete)"),
            file=sys.stderr,
        )
        if not done:
            incomplete.append(d)

    if len(runs) < 2 or incomplete:
        # Not finished (or extra incomplete dirs) -- nothing to deliver yet.
        return 0

    # The two newest completed run dirs, in launch order.
    run1, run2 = runs[-2:]
    OUTPUT_DIR.mkdir(exist_ok=True)

    for (d, ckpts), (rows, cols) in zip([run1, run2], SHAPES):
        ckpts = sorted(ckpts, key=_epoch_of)
        report = Report(output_dir=str(OUTPUT_DIR))
        for p in ckpts:
            epoch = _epoch_of(p) + 1
            report.append_figures(plot_weights_2d(str(p), rows, cols, epoch))
        name = f"soft_momos_wa_rows{rows}_cols{cols}_400ep.pdf"
        report.save(name)
        report.close()
        out = OUTPUT_DIR / name
        print(f"PDF: {out}")
        print(f"MEDIA: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())