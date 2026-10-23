# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.12.3.final.0 (momos-reproduction; uv)
#     language: python
#     name: Python 3.12.3.final.0 (momos-reproduction; uv)
# ---

# %% [markdown]
# # Data Exploration
# ## Data Import
# %%
from src.view.report import Report
import numpy as np
from itertools import product, combinations
from scipy.special import comb, stirling2
import math

report = Report()


# %%
def momos_complexity(n, cap, s, q=32):
    m = np.ceil(n / s)
    k = np.maximum(1, np.floor(m * cap))
    motif_complexity = k * s * q
    mosaic_complexity = m * np.ceil(np.log2(k))
    return motif_complexity + mosaic_complexity


# %%


def fold_momos_complexity(n, cap, s, cap_prime, v, q=32):
    m = np.ceil(n / s)
    k = np.maximum(1, np.floor(m * cap))
    k_prime = np.maximum(1, np.minimum(k * cap_prime, m / v))
    motif_complexity = k * s * q
    fold_complexity = v * k_prime * np.ceil(np.log2(k))
    mosaic_complexity = m * np.ceil(np.log2(k_prime))
    return motif_complexity + fold_complexity + mosaic_complexity


# %%
N = np.logspace(4, 8, 1000)
cap_primes = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
fig_momos = report.new_figure(figSize=(10, 7.95), fontsize=17)

fig_momos._ax().plot(
    N,
    momos_complexity(N, 0.001, 2) / N / 32,
    "--",
    label="MoMos baseline",
    color="black",
)

fig_momos.axhline(
    data={"FP32": 1},
    linestyle="-",
    color="black",
)

qats = {f"Q{qat}": qat / 32 for qat in [2, 4, 8, 16]}
fig_momos.axhline(
    data=qats,
    linestyle="--",
    color="grey",
)

models = {"ResNet20": 268346, "MLP": 986634, "TinyViT": 5.4e6}
fig_momos.axvline(
    models,
    linestyle="--",
    color="grey",
)

fig_momos.plot(
    {
        f"{cap:g}%": (N, fold_momos_complexity(N, 0.001, 2, cap, 128) / N / 32)
        for cap in cap_primes
    },
    exp_name=None,
    x_label="Number of parameters (n)",
    y_label="Bits Ratio vs FP32",
    logx=True,
    axis=None,
    grid=False,
)

fig_momos._ax()
# %%
N = np.logspace(22, 23.5, 1000)
cap_primes = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
fig_momos = report.new_figure(fontsize=15)

fig_momos._ax().plot(
    N,
    momos_complexity(N, 0.001, 2) / N / 32,
    "--",
    label="MoMos baseline",
    color="black",
)

fig_momos.axhline(
    data={"FP32": 1},
    linestyle="-",
    color="black",
)

fig_momos.plot(
    {
        f"{cap:g}%": (N, fold_momos_complexity(N, 0.001, 2, cap, 128) / N / 32)
        for cap in cap_primes
    },
    exp_name=None,
    x_label="Number of parameters (n)",
    y_label="Bits Ratio vs FP32",
    logx=True,
    axis=None,
    grid=False,
)

fig_momos._ax()
# %%

# Stacked-bar complexity breakdown for three hand-picked configs at the MLP
# parameter count (n = 986634, slightly below 1M - the value used in
# momos2d_analysis.py and the `models` dict above). Components: Z = motif
# store, I = fold/intermediate index, M = mosaic store. For plain MoMos there
# is no fold component (I), so its bar is just Z + M.
n_mlp = 986634  # MLP parameter count
q = 32
s = 2

# MoMos (2, 1): r=2, c=1, cap=0.001  -> s = r*c, cap = capacity
momos_parts = (lambda m, k: {"Z": k * s * q, "M": m * np.ceil(np.log2(k))})(
    np.ceil(n_mlp / s),
    np.maximum(1, np.floor(np.ceil(n_mlp / s) * 0.001)),
)

# MoMos (2, 2): r=2, c=2, cap=0.001  -> s = r*c, cap = capacity
s_22 = 2 * 2
momos_22_parts = (lambda m, k: {"Z": k * s_22 * q, "M": m * np.ceil(np.log2(k))})(
    np.ceil(n_mlp / s_22),
    np.maximum(1, np.floor(np.ceil(n_mlp / s_22) * 0.001)),
)

# V-Fold MoMos: r=2,c=1,cap=0.001 / r'=4,c'=32,cap'=0.25 -> v = r'*c'
v = 4 * 32
cap_prime = 0.25
_m = np.ceil(n_mlp / s)
_k = np.maximum(1, np.floor(_m * 0.001))
_kp = np.maximum(1, np.minimum(_k * cap_prime, _m / v))
vfold_parts = {
    "Z": _k * s * q,
    "I": v * _kp * np.ceil(np.log2(_k)),
    "M": _m * np.ceil(np.log2(_kp)),
}

columns = {
    "MoMos\n(2, 1)\ncap=0.001": momos_parts,
    "MoMos\n(2, 2)\ncap=0.001": momos_22_parts,
    "V-Fold MoMos\n(2, 1)\ncap=0.001\n(4, 32)\ncap'=0.25": vfold_parts,
}
component_colors = {"Z": "steelblue", "I": "darkseagreen", "M": "coral"}

fig_complexity = report.new_figure(fontsize=15)
ax = fig_complexity._ax()

seen = set()
for x, parts in enumerate(columns.values()):
    bottom = 0
    for part, bits in parts.items():
        ax.bar(
            x,
            bits / (n_mlp * q),
            bottom=bottom,
            color=component_colors[part],
            label=part if part not in seen else None,
        )
        seen.add(part)
        bottom += bits / (n_mlp * q)

# Indicate the Baseline (FP32, full = 1.0) as a reference line.
ax.axhline(
    1.0,
    linestyle="-",
    color="black",
    linewidth=2.0,
    label="Baseline" if "Baseline" not in seen else None,
)

# Human-readable legend names for the complexity components.
component_labels = {"Z": "Motifs", "I": "Folds", "M": "Mosaic", "Baseline": "Baseline"}

ax.set_xticks(range(len(columns)), columns.keys())
ax.set_ylabel("Bits Ratio vs FP32")
ax.set_yscale("log")
# Floor the log axis below the smallest component (Motifs ~1e-3) so the
# bottom slice is actually rendered instead of being clipped at the axis edge.
ax.set_ylim(bottom=1e-4)
ax.grid(True, which="both", axis="y", ls="-", color="grey", alpha=0.7, linewidth=1.0)
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(part) for part in ["Z", "I", "M", "Baseline"] if part in labels]
ax.legend(
    [handles[i] for i in order],
    [component_labels[labels[i]] for i in order],
)

fig_complexity._ax()
# %%

# Same data, normal (linear) y-scale - shows the absolute bit-budget split
# rather than the log-compressed view above.
fig_complexity_linear = report.new_figure(fontsize=15)
ax_lin = fig_complexity_linear._ax()

seen = set()
for x, parts in enumerate(columns.values()):
    bottom = 0
    for part, bits in parts.items():
        ax_lin.bar(
            x,
            bits / (n_mlp * q),
            bottom=bottom,
            color=component_colors[part],
            label=part if part not in seen else None,
        )
        seen.add(part)
        bottom += bits / (n_mlp * q)

# Indicate the Baseline (FP32, full = 1.0) as a reference line.
ax_lin.axhline(
    1.0,
    linestyle="-",
    color="black",
    linewidth=2.0,
    label="Baseline" if "Baseline" not in seen else None,
)

ax_lin.set_xticks(range(len(columns)), columns.keys())
ax_lin.set_ylabel("Bits Ratio vs FP32")
ax_lin.grid(True, axis="y", ls="-", color="grey", alpha=0.7, linewidth=1.0)
handles, labels = ax_lin.get_legend_handles_labels()
order = [labels.index(part) for part in ["Z", "I", "M", "Baseline"] if part in labels]
ax_lin.legend(
    [handles[i] for i in order],
    [component_labels[labels[i]] for i in order],
)

fig_complexity_linear._ax()
# %%
report.save("v_fold_momos_complexity.pdf")


# %%


def motifs(k, s, q):
    N = np.arange(q)
    S = list(product(N, repeat=s))
    Z = np.array([comb for comb in combinations(S, k)])
    return Z.reshape(-1, k, s)


def mosaic(k, m):
    K = np.arange(k)
    M = np.array(list(product(K, repeat=m)))
    return M.reshape(-1, m)


def reconstruct_nn(Z, M):
    return Z[:, M]


def H(k, s, q, m):
    k = min(k, m, q**s)
    Z = motifs(k, s, q)
    M = mosaic(k, m)
    all_reconstructions = reconstruct_nn(Z, M)
    flattened_reconstructions = all_reconstructions.reshape(-1, m, s)
    H = np.unique(flattened_reconstructions, axis=0)
    return H


def H_size(k, s, q, m):
    num_elements_S = q**s
    total_sum = 0

    for u in range(1, k + 1):
        combinations = comb(num_elements_S, u, exact=True)
        factorial = math.factorial(u)
        stirling = stirling2(m, u)
        term = combinations * factorial * stirling
        total_sum += term

    return total_sum


# %%

k = 3
s = 6
m = 4
q = 2

h = H(k, s, q, m)
len_h = H_size(k, s, q, m)

print(len(h) == len_h)


# %%


def fold_mosaic(folds, m):
    v = len(folds)
    M = np.array(list(product(*(folds[p % v] for p in range(m)))))
    return M.reshape(-1, m)


def fold_H(k, s, q, m, v, k_prime):
    assert m % v == 0
    k = min(k, m, q**s)
    k_prime = min(k_prime, k)
    Z = motifs(k, s, q)
    fold_choices = list(combinations(range(k), k_prime))
    reconstructions = []
    for folds in product(fold_choices, repeat=v):
        M = fold_mosaic(folds, m)
        reconstructions.append(reconstruct_nn(Z, M).reshape(-1, m, s))
    H = np.unique(np.concatenate(reconstructions), axis=0)
    return H


def fold_H_size(k, s, q, m, v, k_prime):
    assert m % v == 0
    N = q**s
    c = m // v

    def A(r):
        return sum(
            math.perm(r, t) * stirling2(c, t, exact=True)
            for t in range(1, min(r, k_prime, c) + 1)
        )

    def B(u):
        return sum(
            (-1) ** w * comb(u, w, exact=True) * A(u - w) ** v for w in range(u + 1)
        )

    return sum(comb(N, u, exact=True) * B(u) for u in range(1, k + 1))


# %%

k = 3
s = 2
q = 2
v = 2
m = 4
k_prime = 2

h = fold_H(k, s, q, m, v, k_prime)
len_h = fold_H_size(k, s, q, m, v, k_prime)

print(len(h) == len_h)

k = 3
s = 1
q = 3
v = 2
m = 6
k_prime = 2

h = fold_H(k, s, q, m, v, k_prime)
len_h = fold_H_size(k, s, q, m, v, k_prime)

print(len(h) == len_h)

print(fold_H_size(3, 2, 2, 4, 1, 3) == H_size(3, 2, 2, 4))
# %%

X = np.linspace(1, 6, 1000)


def f(x, a, b, eps):
    alpha = np.max([np.zeros_like(x), 1 - np.maximum(0, a / eps - x / eps)], axis=0)
    beta = np.max([np.zeros_like(x), 1 - np.maximum(0, x / eps - b / eps)], axis=0)
    return alpha + beta - 1


print(len(f(X, 2, 5, 0.5)))

fig = report.new_figure()
fig.plot(
    {
        "f": (X, f(X, 2, 5, 0.5)),
    },
    "F(x)",
    axis=None,
)
fig.show()


# %%


def momos_storage(n, cap, s, q=32):
    m = np.ceil(n / s)
    k = np.maximum(1, np.floor(m * cap))
    return {
        "Z": k * s * q,
        "M": m * np.ceil(np.log2(k)),
    }


def fold_momos_storage(n, cap, s, cap_prime, v, q=32):
    m = np.ceil(n / s)
    k = np.maximum(1, np.floor(m * cap))
    k_prime = np.maximum(1, np.minimum(k * cap_prime, m / v))
    return {
        "Z": k * s * q,
        "I": v * k_prime * np.ceil(np.log2(k)),
        "M": m * np.ceil(np.log2(k_prime)),
    }


# %%
n = 5.4e6
q = 32
s = 2
caps = [0.1, 0.01, 0.001]
fold_cap = 0.01
cap_prime = 1 / 16
folds = [32, 128, 512]

columns = {"FP32": {"W": n * q}}
columns |= {f"MoMos\ncap={cap:g}": momos_storage(n, cap, s, q) for cap in caps}
columns |= {
    f"V-Fold\nv={v}": fold_momos_storage(n, fold_cap, s, cap_prime, v, q) for v in folds
}

component_colors = {"W": "grey", "Z": "steelblue", "I": "darkseagreen", "M": "coral"}

fig_storage = report.new_figure(fontsize=15)
ax = fig_storage._ax()

seen = set()
for x, parts in enumerate(columns.values()):
    bottom = 0
    for part, bits in parts.items():
        ax.bar(
            x,
            bits / (n * q),
            bottom=bottom,
            color=component_colors[part],
            label=part if part not in seen else None,
        )
        seen.add(part)
        bottom += bits / (n * q)

ax.set_xticks(range(len(columns)), columns.keys())
ax.set_ylabel("Bits Ratio vs FP32")
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(part) for part in component_colors if part in labels]
ax.legend([handles[i] for i in order], [labels[i] for i in order])

fig_storage._ax()

fig_storage.show()
# %%
