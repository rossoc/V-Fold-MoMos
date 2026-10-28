#import "@template/setting:0.1.0": *
#import "@preview/lovelace:0.3.0": *

#show: doc => assignment(
  title: "Softly Constrained MoMos",
  doc,
)

#let cmt(body) = [#h(1fr) #text(size: 0.85em, fill: luma(110))[$triangle.stroked.r$ #body]]

#let report-figure(path, page, caption, label) = [
  #figure(image(path, page: page, width: 82%), caption: caption) #label
]

= Introduction <intro>

Many methods in the literature enforce and exploit redundancies in
over-parameterized neural networks to compress and simplify them.
For example, Mosaic-of-Motifs (MoMos) @bakhtiarifard_algorithmic_2026 partitions
the weights into small blocks and forces every block to be one of a few reusable
motifs.
V-Fold MoMos @rosso_neural_2026 applies MoMos to the mosaic: it partitions
the blocks and reduces the number of motifs each block can use.
This report adds another piece: Soft MoMos pulls the blocks towards the
motifs using the loss, instead of projecting them at the end of each epoch;
it increases accuracy and compression rate compared to MoMos and V-Fold MoMos.

= MoMos and V-Fold MoMos <momos>

Let $bold(w) in RR^n$ be the weights and $psi$ the slicing operator that cuts
them into $m$ blocks of shape $(r, c)$, with $s = r c$.
A MoMos network is

$ bold(w) = psi^(-1)(cal(Z)[cal(M)]), quad cal(Z) in RR^(s times k), quad cal(M) in [k]^m. $

The $k$ columns of $cal(Z)$ are the motifs, and the mosaic $cal(M)$ assigns one
motif to every block.
Storing a block costs $ceil(log_2 k)$ bits, whatever its size.
So larger blocks compress more, and the mosaic dominates the cost.

*V-Fold MoMos* assign each block position to a fold of $k' <= k$ motifs.
Indices into $k'$ motifs cost fewer bits than indices into $k$, so the mosaic
shrinks.

Both methods project the weights onto the corresponding hypothesis space at the
end of every epoch; indeed, they randomly sample the motifs and then they swap
each block with its closest motif under the $L_2$ norm.
As shown by @rosso_neural_2026, the blocks collapse to far fewer distinct
motifs than the hard constraint declares at the start of the training.
Thus, we can achieve higher compression rates with a postprocessing technique;
otherwise, we can preserve higher accuracy, changing the algorithm.

= Soft MoMos <soft-momos>

Soft MoMos applies a soft constraint to the network including the distortion to
the loss.
At the end of the training, for few epochs the network is
recomputed from its representation: mosaic and motifs and the gradient at the
level of the blocks is accumulated on the motifs.
This is similar to soft weight sharing @nowlan_simplifying_1992 @ullrich_soft_2017.
Since $k$ is fixed and the weights equal $cal(Z)[cal(M)]$ after the frozen
phase, the compression rate is known before training.

The training has three steps (@alg:soft).
+ *Init*: Sample $k$ blocks of the network at random as the motifs. This
  is the only randomness, together with the initialization of the network.
+ *Mosaic update*: At the start of every epoch, assign each block to its nearest
  motif; this gives the mosaic $cal(M)$.
  For each batch, add the distortion
  $D = 1 / n sum_i norm(z_(cal(M)_i) - psi(bold(w))_i)^2$ to the loss, as a
  regularizer.
  After the step, move every motif to the mean of its blocks. As noted in the
  thesis @rosso_neural_2026, the distortion is similar to $L_2$
  regularization, but much weaker in practice.
+ *Frozen phase*: In the last epochs (e.g. the last $25%$), fix $cal(M)$ and
  compute the network as $bold(w) = psi^(-1)(cal(Z)[cal(M)])$. The gradient of
  a motif is the sum of the gradients of its blocks, and only $cal(Z)$ is
  trained.

Both motifs and mosaic are learnt throughout the training, and so there is a
small overhead during the training, since the distortion is computed for every
batch.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  pseudocode-list(booktabs: true, title: [Soft MoMos Training])[
    + *Input:* weights $bold(w)$, block shape $(r, c)$, motifs $k$, epochs $T$,
      frozen fraction $p$ (`frozen_pct`), distortion scale $lambda$
    + $cal(Z) <- {0} union {" sample " k - 1 " blocks from " psi(bold(w))}$
    + *for* epoch in 1, ..., $(1 - p) dot T$ *do* #cmt[soft phase]
      + $cal(M)_i <- arg min_(j in [k]) norm(z_j - psi(bold(w))_i)$ for every
        block $i$
      + *for* batch in 1, ..., batches *do*
        + update $bold(w)$ with a gradient step on
          $cal(L)_"batch" + lambda dot D$
        + $z_j <- "mean" {psi(bold(w))_i : cal(M)_i = j}$ for every $j in [k] \\ {0}$ #cmt[motifs follow the blocks]
    + *for* epoch in 1, ..., $p dot T$ *do* #cmt[frozen phase: $cal(M)$ is fixed]
      + *for* batch in 1, ..., batches *do*
        + $bold(w) <- psi^(-1)(cal(Z)[cal(M)])$
        + $nabla z_j <- sum_(i : cal(M)_i = j) nabla_(psi(bold(w))_i) cal(L)_"batch"$
          for every $j in [k] \\ {0}$
        + update $cal(Z)$ with a gradient step #cmt[only $cal(Z)$ is trained]
    + *return* $cal(Z)$, $cal(M)$
  ],
) <alg:soft>

= Experiments <experiment>

We trained Soft MoMos for 100 epochs to find the best
- $"frozen_pct"$ per block size;
- scale of the distortion regularizer.

We also want the best block size for a fixed $k = 256$. We chose
$k = 256 = 2^8$ because it is the most convenient value in practice: it
corresponds to the type `uint8`.
For the MLP, this corresponds to the capacity
$(k dot s) \/ n = #calc.round(256 / 986634 * 100, digits: 2)% dot s$.
We fix $k$ because the original paper @bakhtiarifard_algorithmic_2026 points out
that, at a fixed capacity, a smaller block size is better.
Thus, a fixed $k$ means the capacity increases with the block size, and
the compression rate.

We searched block shapes $(r, c) in {1, 2} times {2, 3, 4, 5}$.
This report compares four configurations: QAT, Soft MoMos, MoMos, and V-Fold
MoMos (written Fold in the figures). Shapes are written $(r, c)$.

After the initial search, we repeated the training of the best configurations
for 200 epochs with 5-fold cross-validation.
Finally we report the compression ratio as
$r_phi = (n dot q) \/ (k s q + ceil(n \/ s) dot ceil(log_2 k))$, where $q$ is
the bit width of a weight.

= Results <result>

- Soft MoMos seems to beat the previous Pareto front on the MLP
  (@fig:pareto, @tab:pareto). The breakdown of the compressed size into motifs and mosaic is
  in @fig:breakdown.
- At the end of the training (at $"frozen_pct"$), the training accuracy drops
  while the validation accuracy rises (@fig:training).
- The validation accuracy stays low during the soft phase and jumps at the
  freeze (@fig:training2).
- Soft MoMos induces a good amount of sparsity (@fig:sparsity).
  This is reasonable, since it acts like an $L_2$ penalty on weights close to 0.
- BDM @zenil_block_2022 and QuBD @bakhtiarifard_characterizing_2026 show that
  complexity increases after projection onto the MoMos space (@fig:bdm,
  @fig:qbd). This seems to contradict the reduced overfitting.
- The entropy smoothly decreases after the projection (@fig:entropy),
  thus the increase in BDM and QuBD may not come from the entropy factor.
  We cannot explain this yet, and bugs are always possible.
- The distortion of Soft MoMos is orders of magnitude bigger than that of MoMos
  and V-Fold MoMos, and it has not converged by the end of the soft phase
  (@fig:distortion).

#figure(
  table(
    columns: 5,
    align: (left, center, right, right, center),
    stroke: none,
    inset: (x: 6pt, y: 3pt),
    table.hline(),
    table.header(
      table.cell(inset: (x: 6pt, y: 6pt))[*Method*],
      table.cell(inset: (x: 6pt, y: 6pt))[*Shape*],
      table.cell(inset: (x: 6pt, y: 6pt))[*$r_phi$*],
      table.cell(inset: (x: 6pt, y: 6pt))[*Test acc. (%)*],
      table.cell(inset: (x: 6pt, y: 6pt))[*Pareto*],
    ),
    table.hline(),
    table.cell(rowspan: 2, align: horizon)[QAT],
    [8 bit], [4.0], [58.0 ± 0.5], [✓],
    [4 bit], [8.0], [57.5 ± 0.4], [✓],
    table.hline(stroke: 0.5pt),
    table.cell(rowspan: 6, align: horizon)[MoMos],
    [(1,2)], [4.7], [57.5 ± 0.4], [],
    [(1,2)], [9.1], [54.4 ± 0.5], [],
    [(2,2)], [9.5], [54.4 ± 0.2], [],
    [(2,2)], [18.1], [50.4 ± 0.4], [],
    [(2,4)], [30.6], [47.9 ± 0.5], [],
    [(2,4)], [30.6], [47.6 ± 0.4], [],
    table.hline(stroke: 0.5pt),
    table.cell(rowspan: 4, align: horizon)[V-Fold MoMos],
    [(2,2)], [11.8], [53.2 ± 0.4], [],
    [(2,2)], [17.3], [51.4 ± 0.4], [],
    [(1,4)], [26.3], [49.3 ± 0.1], [✓],
    [(1,4)], [39.3], [41.1 ± 1.2], [✓],
    table.hline(stroke: 0.5pt),
    table.cell(rowspan: 5, align: horizon)[Soft MoMos],
    [(1,3)], [11.9], [55.8 ± 0.5], [✓],
    [(2,2)], [15.7], [54.6 ± 0.4], [✓],
    [(1,5)], [19.5], [52.8 ± 0.5], [✓],
    [(2,3)], [23.1], [52.5 ± 0.4], [✓],
    [(2,5)], [36.2], [48.6 ± 0.7], [✓],
    table.hline(),
  ),
  caption: [Values referred to @fig:pareto, test accuracy
    (mean $plus.minus$ std over 5 folds) and compression rate. V-Fold MoMos
    lists its primary block shape.],
) <tab:pareto>

#report-figure(
  "../assets/mlp_merged_cv_report.pdf",
  1,
  [CV-verified test accuracy against compression rate $r_phi$, for all methods.],
  <fig:pareto>,
)
#report-figure(
  "../assets/mlp_merged_cv_report.pdf",
  2,
  [Compressed size of the Pareto-optimal configurations, split into motifs and
    mosaic.],
  <fig:breakdown>,
)
#report-figure(
  "../assets/mlp_merged_cv_report.pdf",
  3,
  [Training overview of QAT, Soft (1,3), MoMos (1,2) and Fold (2,2); the shapes,
    hence the compression rates, differ.],
  <fig:training>,
)
#report-figure(
  "../assets/mlp_merged_cv_report.pdf",
  9,
  [Sparsity against validation loss over training.],
  <fig:sparsity>,
)
#report-figure(
  "../assets/mlp_merged_cv_report.pdf",
  12,
  [Distortion against validation loss over training.],
  <fig:distortion>,
)
#report-figure(
  "../assets/mlp_soft_momos_cv_report.pdf",
  3,
  [Training overview of 4 Soft MoMos experiments.],
  <fig:training2>,
)
#report-figure(
  "../assets/mlp_soft_momos_cv_report.pdf",
  4,
  [BDM complexity against validation loss, for four Soft MoMos shapes.],
  <fig:bdm>,
)
#report-figure(
  "../assets/mlp_soft_momos_cv_report.pdf",
  10,
  [QuBD complexity against validation loss, for four Soft MoMos shapes.],
  <fig:qbd>,
)
#report-figure(
  "../assets/mlp_soft_momos_cv_report.pdf",
  11,
  [Entropy against validation loss, for four Soft MoMos shapes.],
  <fig:entropy>,
)

= Discussion and Conclusion <conclusion>

We find this result interesting because it invites deeper thought about the
compression constraint enforced by MoMos. We do not claim that the proposed
algorithm is final, but it shows that the hard MoMos constraint allows a rich
hypothesis space. The MoMos algorithm in @bakhtiarifard_algorithmic_2026 has a
very strong constraint that is not fully exploited: we could either compress
more or preserve higher accuracy.

The regularization could be strengthened, perhaps with an $L_1$ norm. This
might reduce the distortion and stabilize training, avoiding the jump in
training and validation accuracy that the MoMos projection causes at the end.

It is also important to highlight that the best hyper-parameters were found
using optuna on a training of 100 epochs, and then the training was extended to
200 epochs. It means that the `frozen_pct` is biased on the searching space.
Therefore, an algorithm that is robust to the number of epochs could enhance the
performance.

We also highlight that the MLP model on CIFAR10 tends to overperform the quality
of the method.

#bibliography("biblio.bib")
