# ConStellaration UQ Surrogate: Decision Log

**Project:** Uncertainty-aware surrogate on `proxima-fusion/constellaration`, plus a study of where its predictions can be trusted.
**Budget:** ~2.5 weeks, 10 to 20 h/week.
**Started:**

## How to use this

Work top to bottom. Items marked `SETTLED` are already decided; the reasoning is recorded so you can defend it later or overturn it deliberately. Items marked `OPEN` need your call. `GATE` items are stop-and-check points.

Write your own reasoning in the blockquote under each open item, in your own words. That text is what you will use to answer "why did you do it that way" three weeks from now.

⚠️ marks the two decisions that fail silently if you get them wrong.

---

## Stage 0. Framing

---

**0.1 Project shape** `SETTLED`

**Decided:** Uncertainty-aware surrogate on ConStellaration, paired with a study of where its predictions can be trusted.

**Why:** The generative modelling lane is crowded (the paper's own Section 5 plus at least one published diffusion paper). The UQ lane is empty, and the paper's Appendix A.4 names uncertainty calibration as an open direction in print.

---

**0.2 Method family** `SETTLED`

**Decided:** Deep ensemble of mean-variance networks. No Gaussian processes, no Laplace approximation in the core.

**Why:** Ensembles give epistemic and aleatoric directly, without needing to subtract one from the other. GPs and Laplace go in future work, where they cost nothing.

---

**0.3 Scope discipline** `SETTLED`

**Decided:** A finished narrow project beats an unfinished ambitious one. Active learning is optional and gets cut first.

**Why:** Two and a half weeks. Everything in Stage 9 exists to be cut.

---

**0.4 Headline deliverables** `SETTLED`

**Decided:** Two. (1) Calibration across all three splits, binned by distance from the training region, with the interior-hole versus tail contrast as the main figure. (2) The deferral curve.

**Why:** Two clear results beat five vague ones. Every other output supports one of these.

**Updated:** Deliverable 1 was originally "the in-region versus out-of-region calibration gap." Reframed twice since: the three-condition design (3.8) replaced the single split, and the finding is now generalization behaviour under each split, reported whether or not a gap appears. Framing a result as "the gap is the finding" leaves you with nothing to report if the gap is small.

---

**0.5 Novelty claim** `SETTLED`

- [x] Decided

**Decided, use this wording:**

> As of August 2026 I found no published work studying uncertainty calibration or extrapolation reliability of surrogates on this dataset. The nearest work is Bayesian optimization using GP uncertainty on a 38-configuration subset, where the GP is optimization machinery rather than an object of study.

**Why narrow, not broad:** "no UQ work exists on this dataset" is falsifiable in one sentence, because the alpha-particle Bayesian optimization paper does carry GP uncertainty. Narrowing to calibration and extrapolation reliability holds cleanly and shows you actually looked.

---

### Search record (performed 2026-08-26)

**Channels, four:** topic search (dataset + UQ + calibration + surrogate), citation search on arXiv 2506.19583, method search (stellarator + deep ensemble / epistemic / calibration), exhaustive term search (dataset name + deferral + active learning + ensemble). Then the Semantic Scholar citation graph via API, and the HF dataset Community tab.

**Coverage:** ~14 distinct works touch the dataset. Four read in full text rather than judged by title.

**Note on method:** web search and Semantic Scholar barely overlapped. The citation graph missed three papers web search found (sheaf neural operators, alpha-particle BO, latent diffusion), and web search missed eight of the graph's eleven. Running one alone would have produced a false clear.

**The two near-misses, know these by name:**

1. **Bayesian optimization of stellarator alpha-particle confinement** (arXiv 2606.19523, Jun 2026). Uses 38 ConStellaration boundaries inside a 270-configuration set. Does use GP uncertainty via Bayesian optimization. But no calibration analysis, no distribution-shift or extrapolation study; the GP is optimizer machinery. They do test sensitivity to dataset composition, which is adjacent but a different question.
2. **RAMBO, Regime-Adaptive Bayesian Optimization** (arXiv 2601.20043, Jan 2026). Discusses miscalibrated GP uncertainty, but as a general argument about single GPs oversmoothing sharp transitions. Cites ConStellaration in its references without using it.

**Everything else** is physics optimization or generative work. The generative lane is confirmed crowded (at least three diffusion or generation papers). The forward-surrogate-with-UQ lane is empty.

**Incidental find, useful elsewhere:** two independent papers describe the low-aspect-ratio region as sparsely sampled and are actively generating there. See 3.4.

**HF Community tab:** two threads only. One Parquet bot, one unanswered question from a month ago asking whether the task is boundary-to-metrics or metrics-to-boundary. Nothing competing, and a mild signal that the forward direction is uncrowded.

⚠️ **The claim is dated on purpose.** If the write-up or interview lands materially later than August 2026, re-run the search before repeating the claim.

**My decision:**
> Narrow and dated. I searched four channels, found about fourteen works using the dataset, and read the four that could have collided. Two come close on method but neither studies calibration or extrapolation, and I can name both. A broad claim would get knocked down in one sentence by anyone who knows the field.

---

## Stage 1. Data

*Blocks everything downstream. Target: first afternoon.*

---

**1.1 Which subset** `SETTLED`

**Decided:** `default` only. Not the finite-beta subsets.

**Why:** The HF repo has twelve subsets (default, five finite-beta, and six matching `vmecpp_wout`), which sum to roughly 958k rows and explain the ~959k the viewer shows. Mixing vacuum and finite-beta rows without beta as an input would put genuinely different answers on identical inputs. That is the one real source of unobserved-confounder noise in this dataset and it is avoidable.

---

**1.2 Field period filter** `SETTLED`

- [x] Decided

**Decided:** NFP=3.

**Why:** Largest bucket by a wide margin (paper reports 15k, 20k, 68k, 27k, 28k at NFP 1 to 5). The toroidal mode index is in units of NFP, so the coefficient vector means something different at each value. Mixing them is two encodings in one feature vector.

**Blocks:** 1.3, 3.4, 3.6

**Knock-on effect:** field period is now a constant column, so anything that relied on varying it is dead. Killed as a split-axis fallback (3.4), as a manufactured-aleatoric mechanism (2.4), and as an optional extra shift (old 9.4, deleted).

**My decision:**
> NFP=3. It is the biggest bucket and it is what Proxima filtered to in their own example, so the Table 7 comparison stays clean. Mixing field periods would put two different encodings in one feature vector.

---

**1.3 Generation pathways** `SETTLED`

- [x] Decided

**Decided:** Match Appendix A.4 exactly. DESC and VMEC++ optimized boundaries only, ~23k points after their filters.

**Why:** Matching makes your in-region numbers directly comparable to their Table 7, which is the strongest positioning available to you. Adding the other pathways gives more data and a more honest mixture, but breaks comparability.

**Risk accepted:** the ~23k pool could be thin at the compact end you plan to hold out. That risk is already covered by the day 1-2 baseline gate, which checks histogram thickness before any ensemble is built, so it does not need a second mitigation here.

**Knock-on effect:** generation pathway is now a constant column, so it is dead as a split-axis fallback (3.4).

**My decision:**
> Match A.4. One variable changed, not two: same data slice as their baseline, plus the UQ layer on top. If I widen the pathways my numbers drift from Table 7 and I spend write-up space explaining why instead of on the actual result.

---

**1.4 Outlier trimming** `SETTLED` ⚠️

- [x] Decided

**Decided:** Match A.4's 0.05% per-metric tail trim, applied to the target metric only. Never to the split axis.

**Why:** Trimming tails on aspect ratio deletes exactly the compact configurations you plan to hold out. It destroys the extrapolation condition and you would not notice until the test set came back tiny and you blamed the histogram. Matching the trim on the target keeps Table 7 comparability.

**My decision:**
> Trim the target the way A.4 did so the numbers stay comparable, and never touch the split axis. Trimming the split axis would silently delete the test set and I would misread it as a thin histogram.

---

**1.5 Input vector** `SETTLED` ✅ VERIFIED 2026-08-26

**Decided:** The 80 coefficients as the paper defines them, with major radius fixed at 1.

**Why:** Poloidal and toroidal mode numbers capped at 4, stellarator symmetry zeroing certain entries, R(0,0) fixed at 1. The paper states 80 degrees of freedom explicitly. Confirm your loader returns exactly 80 columns before going further; if it does not, you have misread the symmetry conditions.

**Verified against their code, not inferred.** `_to_X` in `src/constellaration/generative_model/bootstrap_dataset.py`:

```python
x = np.concatenate([
    surface.r_cos.ravel()[surface.max_toroidal_mode + 1:],
    surface.z_sin.ravel()[surface.max_toroidal_mode + 1:],
])
```

`max_toroidal_mode = 4`, so each `(5, 9)` array ravels to 45 entries and drops indices `[0:5]`, which are `m=0, n=-4..0`. That is the four symmetry zeros plus R(0,0). 40 kept per array, 80 total.

**Verified empirically across all 182,221 usable rows:** `r_cos[m=0, n<0]` and `z_sin[m=0, n<0]` are exactly 0.0, and `Z(0,0)` is exactly 0.0. No exceptions, not "small", exactly zero.

**Surprise on R(0,0), resolved.** Stored R(0,0) is *not* exactly 1: range 0.894 to 1.016, std 0.0051, only 52.8% of rows within 1e-4 of 1.0. But it is 6 to 20 times less variable than the least-variable genuine coefficient (std 0.033 to 0.103), correlates weakly with aspect ratio (0.14), and their inverse function `_x_to_surface` hardcodes `r_cos[max_toroidal_mode] = 1.0` on reconstruction. So it is optimizer residue around a fixed convention, and dropping it is correct.

**Decision: drop R(0,0). 80 columns.**

---

**1.6 Failed rows** `SETTLED` ✅ VERIFIED 2026-08-26

**Decided:** Load `default` unfiltered to get the ~24k flagged failures. Use them only for the survivorship figure, never for training.

**Why:** `misc.has_neurips_2025_forward_model_error` is a real column, so survivorship is measurable rather than speculative. Watch for nulls: a null almost certainly means "this check does not apply to this generation pathway," not "no error." Also check whether `boundary.r_cos` is populated for flagged rows, since some may have failed at generation rather than simulation.

**Null convention confirmed.** Their loader `load_source_datasets_with_no_errors` does `.fillna(False)` across all five error flags then drops any row with a True. So nulls count as "no error", and the filter is all five flags, not just the NeurIPS one:

```python
errors_dframe = dframe[[
    "misc.has_optimize_boundary_omnigenity_vmec_error",
    "misc.has_optimize_boundary_omnigenity_desc_error",
    "misc.has_generate_qp_initialization_from_targets_error",
    "misc.has_generate_nae_initialization_from_targets_error",
    "misc.has_neurips_2025_forward_model_error",
]].fillna(False)
dframe = dframe[~errors_dframe.any(axis=1)]
```

⚠️ **THE NULL-ROW TRAP.** Exactly one row in 182,222 (file 3, index 60739) has `boundary.r_cos`, `boundary.z_sin`, `n_field_periods` and every metric set to `None`. It failed at boundary *generation*, so the solver never ran on it, so:

```
misc.has_optimize_boundary_omnigenity_desc_error : True
misc.has_neurips_2025_forward_model_error        : False   <- reads CLEAN
```

Filtering on `has_neurips_2025_forward_model_error` alone, which is the flag named throughout the docs, lets this row through. It then either throws on `np.array(None)` or silently produces NaNs that surface much later as an unexplained loss failure. It did in fact crash the first full-dataset script written for this project.

**Required:** drop rows with null `boundary.r_cos` / `z_sin` as an explicit standalone check, never as a side effect of trusting an error flag. It happens to be redundant on today's data (the five-flag filter already catches this row via the DESC flag) and it stays in anyway. The point is that the filter cannot let a null through, not that it happens not to.

---

**1.7 Boundary sign convention (z_sin canonicalization)** `SETTLED` 2026-08-26

- [x] Decided

**Decided:** Do NOT canonicalize. Use the 80 coefficients exactly as stored.

**What this is.** The boundary is a Fourier series in two angles, θ poloidal and φ toroidal:

```
R(θ,φ) = Σ r_cos[m,n] · cos(mθ − n·NFP·φ)
Z(θ,φ) = Σ z_sin[m,n] · sin(mθ − n·NFP·φ)
```

Negating every `z_sin` coefficient leaves R untouched and sends Z to −Z, which is the same shape reflected through the horizontal midplane. So two different 80-vectors can describe mirror-image shapes.

**The stored data is not canonicalized:** 89.4% of the pool has the reference coefficient `z_sin[m=0,n=1] < 0`, 10.6% has it positive.

**Their code does both things, in different places.** `_flip_z_sin_if_negative` in `_augment_dataset` forces one handedness, but that is generative-model-specific. Their surrogate feature extractor `_to_X` does **not** canonicalize. For a forward surrogate, `_to_X` is the precedent that applies.

**Why not canonicalize, primary reason:** `_to_X` does not canonicalize, and Table 7 comparability is the entire point of matching their pipeline. Canonicalizing would put our inputs in a different space from the precedent we measure against.

**Why not canonicalize, second and independent reason:** the two handedness classes are not interchangeable. The sign of the reference coefficient correlates with the target at −0.29 across the 27,022-row pool, and the classes have different aspect ratio, elongation and triangularity distributions. Folding them together discards a signal the model can currently use.

⚠️ **The original justification does not survive contact with their code. Corrected 2026-08-27.** This entry previously argued that mirroring flips the sign of rotational transform, so canonicalizing inputs without flipping targets would create identical inputs with opposite labels. Checked against both the repo (cloned, read directly) and the pool:

- Their canonicalizer keys on `z_sin[0, DATASET_MAX_TOROIDAL_MODE + 1]`, and that constant is 4, so the reference coefficient is `z_sin[m=0,n=1]`. Confirms the column used above is the right one. Its docstring reads "Flip theta sign."
- `forward_model.py:135-144` interpolates `equilibrium.iotaf` to normalized effective radius 1.0 and passes it straight through, divided by NFP at line 217. The stored metric is VMEC's **signed** edge iota. No absolute value anywhere in the forward model.
- All three benchmark problems apply `np.abs()` to the metric at constraint evaluation (`problems.py` lines 174, 248, 384). That is dead code if the value were always positive, written three times. Proxima themselves treat the sign as physically arbitrary.
- Their own optimizer does not. `augmented_lagrangian_runner.py` builds the same constraint from the raw signed value in all three problem branches. So their scorer enforces `|iota| ≥ bound` while their generator enforces `iota ≥ bound`. That inconsistency is theirs, not a subtlety we missed.
- In the pool, both handedness classes are overwhelmingly positive: 24,041 of 24,169 with the reference negative, 2,848 of 2,853 with it positive. Only 133 negative targets in 27,022, about 0.49%.

Read together: the generation optimizer selected for positive signed iota, which is why negatives are rare, and handedness does not determine the sign of the stored target. The most consistent reading is that VMEC's iota sign follows its own toroidal flux convention rather than boundary handedness, and the `np.abs()` in `problems.py` is defensive coding for cases where it might not.

⚠️ **Retracted supporting evidence.** This entry previously cited "the target takes both signs in the data (minimum −0.49 and −0.56 in the two sign groups)" as showing handedness is a real varying property. It shows no such thing. A negative minimum within each group is consistent with a 0.5% negative tail and says nothing about handedness. Do not reuse that sentence.

**Still open, deliberately parked:** whether a genuine mirrored *pair* carries opposite iota. Neither their code nor the class-level distributions settle it, because two rows in opposite handedness classes are not necessarily mirror images of one another. Testable via the pair search folded into 2.2. Nothing downstream depends on the answer.

**Side effect worth remembering:** `aspect_ratio_over_edge_rotational_transform` divides by the *signed* edge transform (`forward_model.py:214`), so it inherits the sign. Relevant only if that metric is ever promoted to a target.

**My decision:**
> Unchanged conclusion, rebuilt justification. Leave the coefficients alone because `_to_X` leaves them alone and comparability to Table 7 is the point, and because the two handedness classes carry genuinely different information. Drop the label-corruption argument entirely: it is not what their pipeline does, and it is the kind of sentence an interviewer would pull on.

---

**1.8 The filter chain** `SETTLED` ✅ VERIFIED 2026-08-26

- [x] Decided

**Decided:** the ordered row-dropping sequence from raw download to training pool.

| step | operation | rows left |
|---|---|---|
| 0 | load all three `data/*.parquet` files | 182,222 |
| 1 | drop rows where any of the 5 error flags is True (`fillna(False)` first) | 158,685 |
| 2 | keep `boundary.n_field_periods == 3` | 68,191 |
| 3 | keep rows where `desc` or `vmec` settings id is non-null | 27,050 |
| 4 | drop rows with null `boundary.r_cos` / `z_sin` | 27,050 |
| 5 | trim 0.05% off each tail of the **target metric only** | ~27,023 |

Steps 1 to 3 are Proxima's own filters, copied from their loader for Table 7 comparability. Step 4 is the null-row guard (1.6). Step 5 is 1.4, deliberately narrower than their all-twelve-metric trim, because trimming the split axis would delete the extrapolation region.

**Two counts reconcile exactly and confirm the chain is right:** 158,685 matches the paper's "~158k evaluated without errors", and 68,191 matches the documented "68k at NFP=3" (that figure is post-error-filter, which resolves an apparent discrepancy with the raw NFP=3 count of 82,043).

**Generation pathway has no single column.** It is read off which of four settings blocks has a non-null `.id`. This partitions cleanly: 182,221 of 182,222 rows have exactly one populated. Counts match the documented pathways: `desc` 87,727 (~88k), `nae_init` 48,851 (~49k), `qp_init` 29,886 (~30k), `vmec` 15,757 (~15k). Note these `.id` values identify *settings groups*, not rows: 17,671 desc rows share only 148 unique ids, so they cannot be used for deduplication.

**⚠️ Unresolved: 27,050 vs A.4's stated ~23k.** Ruled out by direct test: duplicates (only 8 identical boundaries in 27,050), the five-flag filter (removes 1 extra row over the NeurIPS flag alone), and the documented 0.05% per-metric trim (leaves 26,746; reaching 23k would need roughly a 1% trim). Two surviving explanations, not distinguishable from public information: A.4's appendix describes its filter chain incompletely, or the HF dataset grew after the NeurIPS submission.

**Decided:** accept 27,050 and document the discrepancy honestly in the write-up. A.4's training code is not published (same reason its NRMSE/SNR formulas are unrecoverable, see 5.3), so exact reproduction may not be achievable, and a 15% pool difference will not change which (axis, direction) pair wins the day 1-2 grid.

**My decision:**
> Match their filters where their code shows them, then stop. I can account for every step of my chain and two of my intermediate counts land exactly on numbers the paper reports, so the pipeline is right even though the final count is 15% above their stated figure. Forcing it down to 23k would mean inventing a filter they never documented, which is worse than a documented discrepancy.

---

## Stage 2. Noise floor

*Your stated priority 1. Target: one afternoon.*

---

**2.1 Pre-register the prediction** `SETTLED`

**Decided:** Write down before looking: aleatoric will sit at the numerical floor.

**Why:** Inputs are fully observed (no truncation), the solver is deterministic, fixed tolerance produces a deterministic high-frequency function rather than noise, and the four generation pathways shift the distribution over x without changing y given x. Predicting this and confirming it is a result. Hunting for noise that is not there is a wasted afternoon.

---

**2.2 Duplicate detection tolerance** `SETTLED` ✅ 2026-08-27 (mirror-pair search still open)

- [x] Decided (exact-match and tolerance-based passes both done)

**Tolerance-based result (2026-08-27), `scripts/stage_2_noise_floor.py`.** Tolerance set to **0.2 in z-scored 80-dimensional distance**, chosen from the nearest-neighbour distance distribution rather than in advance: p0.05 is 0.000000 and p0.1 is 0.1965, so 0.2 is where the exactly-coincident pairs stop and genuinely distinct shapes begin. A natural break, not a round number.

| | pairs | median &#124;dy&#124; | p90 | max | Table 7 RMSE |
|---|---|---|---|---|---|
| edge rotational transform | 28 | 0.000000 | 0.000376 | 0.003917 | 0.006 |
| log10 qi | 28 | 0.000000 | 0.000976 | 0.021100 | 0.051 |

**Verdict: no near-duplicate contradiction.** Median target difference among near-identical shapes is exactly zero to six decimals for both targets. Two pairs per target exceed the 2.3 floor bar, which is expected and not a concern: that bar is calibrated for an aggregate floor rather than individual pairs, and those pairs sit at the top of the 0 to 0.2 distance range where genuine shape difference applies. The decisive comparison is that the *worst* near-twin disagreement is below Table 7's RMSE for the same metric, so even the extreme case would not limit a model.

**Caveat:** only 28 of 27,050 rows have a neighbour that close, so tail statistics are thin. The result holds because the median is exactly zero rather than merely small.

**Options:** round coefficients to N decimals and group / exact match only / nearest-neighbour distance threshold

**Leaning:** start coarse, tighten until groups appear

**Why:** You are looking for identical inputs with different targets. Also check whether `plasma_config_id` repeats across rows, which would be the trivial explanation for anything you find.

**Exact-match result:** hashing `boundary.json` over the 27,050-row training pool gives 27,042 unique, so **8 exact duplicates**. Negligible, and far too few to explain anything. The tolerance-based version (round-and-group, or kNN threshold) is still worth one pass as part of 2.3, but the exact-duplicate explanation for target spread is now ruled out.

**Note:** there is no usable `plasma_config_id`. The four `*_optimization_settings.id` columns identify settings groups, not configurations (17,671 desc rows share 148 unique ids), so they cannot serve this purpose.

**Mirror-pair search, folded in here** `ADDED` 2026-08-27

Third question answered by the same index, at roughly ten extra lines. Settles the piece of 1.7 left open: does a genuine mirrored pair carry opposite iota?

- **Query set.** Copy X, negate columns 40:80. Negating every `z_sin` is the mirror, so this is the mirrored version of every row.
- **Scaling.** Fit the scaler on the original pool, then apply that same transform to the mirrored queries. Do not refit on the mirrored set or the two sides land in different spaces. Note that negating `z_sin` leaves its standard deviation unchanged, so this bites on the means.
- **Search.** Index the originals, query with the mirrors. Brute force, 27k x 80 is seconds. Same `NearestNeighbors` object as 2.3.
- **Exclude self-matches.** A shape with `z_sin` near zero is its own mirror and will match itself perfectly. Drop any query whose distance to its own unmirrored row is below tolerance.
- **Do not pick a tolerance blind.** Compare two distributions: each row's distance to its nearest *real* neighbour excluding itself, against each mirrored query's distance to its nearest real row. The first is the yardstick for how close two independently generated designs normally land. Overlapping distributions mean genuine pairs plausibly exist; mirrored distances sitting systematically further out mean the dataset never contains both handedness versions of any shape. **That comparison is the result.** Everything else is plumbing.
- **Verdict.** For pairs inside tolerance, scatter partner target against original. On `y_j = +y_i` the target is mirror-invariant. On `y_j = −y_i` it flips. No pairs at all is a clean answer too.

⚠️ **"No pairs exist" does not make canonicalizing safe.** It only means the training set holds no contradictory labels today. The model still meets both handednesses at test time, and folding them together still discards the −0.29 signal. All three outcomes leave the 1.7 decision standing.

**When:** here, inside 2.3's index build. Not as its own task, which is what would make it expensive. Run it standalone only if the sign claim is challenged in review, or if `aspect_ratio_over_edge_rotational_transform` is ever promoted to a target, since it inherits the sign.

**My decision:**
> Exact duplicates are a non-issue, 8 in 27k. Fold the tolerance-based check into the kNN work in 2.3 rather than treating it as its own step, since standardized-coefficient distance answers both questions at once. Same for the mirror-pair search: same index, same distance, third question answered nearly free. If it had needed its own afternoon I would have skipped it, since nothing downstream depends on the answer.

---

**2.3 kNN residual spread setup** `SETTLED`

- [x] Decided

**Decided:** Distance on standardized (z-scored) coefficients. k = 5 to 10, not tuned further.

**Why:** Coefficient magnitudes decay hard with mode number, so raw Euclidean distance is dominated by the low-order modes and your "neighbours" are only neighbours in a few dimensions. k is not worth optimising, this is a coarse sanity check, not a figure.

**Note:** unlike 3.11, full 80-dimensional distance is the right tool here. This check asks a general question about local consistency across the whole dataset, so it has no reason to be tied to one axis.

**My decision:**
> Standardize first, otherwise "nearest neighbour" just means "similar in the few low-order modes that happen to be biggest." k anywhere from 5 to 10 is fine, this is a sanity check and I am not going to tune it.

**Result (2026-08-27), `scripts/stage_2_noise_floor.py`.** k=10, brute force, distance on z-scored coefficients. Method: pair each shape with its single nearest neighbour, bin the pairs by distance, take the median target difference per bin, then fit a line through the closest three bins and read the intercept at zero distance. The intercept is the floor estimate; the slope is just local steepness of the physics and is not the quantity of interest.

**Pre-registered bar:** 20% of Table 7's RMSE. Independent errors combine in quadrature, so a floor at 20% of model error inflates total error by about 2%, which is invisible. Stating the bar as a tolerance on total error rather than a bare percentage is what makes it defensible.

| target | intercept | bar | verdict |
|---|---|---|---|
| edge rotational transform | −0.000003 | 0.0012 | AT FLOOR |
| log10 qi | 0.001834 | 0.0102 | AT FLOOR |

**Verdict: the 2.1 prediction is confirmed. Aleatoric sits at the numerical floor for both candidate targets.** The negative intercept on edge rotational transform is fit noise, meaning indistinguishable from zero.

**The single strongest line in the output:** the 28 pairs closer than distance 0.197 have a median target difference of **0.000000 for both targets**. Shapes that are effectively identical carry identical answers to six decimals. That is determinism measured rather than asserted, and it is the sentence worth quoting.

⚠️ **Binning matters more than expected, and the first pass got it wrong.** With uniform deciles, log10 qi came out ABOVE FLOOR at intercept 0.021, twice its bar. The cause was the bottom decile spanning distance 0 to 1.53 while genuinely close pairs live below 0.5, so the fit window sat at distances 1.1 to 2.4 and the intercept was a long extrapolation into unobserved territory. qi is about 12x steeper in shape than edge rotational transform (slope 0.053 vs 0.0044), and steeper functions curve more, so a linear extrapolation overshoots hardest there. Rebinning with quantile edges packed into the left tail (`BIN_QUANTILES = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]`) dropped the intercept 10x to 0.0018 and flipped the verdict. **Record this rather than hiding it:** the first answer was a binning artifact, diagnosed by noticing the fit window never came near zero.

⚠️ **High-dimensional caveat, belongs in the write-up.** Median nearest-neighbour distance is 3.84 in z-scored 80-dimensional space, which is far. In high dimensions everything is far from everything, so genuine near-twins barely exist outside the bottom percentile. This caps what the check can prove, and it is better stated by us than discovered by a reviewer.

---

**2.4 Manufactured aleatoric** `SETTLED`

- [x] Decided

**Decided:** Hide the high mode-number coefficients.

**Why:** Hiding inputs creates a known aleatoric component, so you can check whether the variance head recovers the right magnitude instead of asserting the decomposition works. You know the true map is exactly deterministic, which almost no real dataset lets you claim. Hiding coefficients is graded and tunable.

**Option removed:** "hide the field period" is no longer possible. NFP is fixed at 3 by 1.2, so there is nothing left to hide.

**Feeds:** 6.2

**My decision:**
> Drop the high mode-number coefficients from the input. The model then cannot distinguish shapes that genuinely differ, which is real aleatoric noise of a size I chose. That turns the decomposition into something I measured rather than something I asserted, and it is the reason the variance head is not decorative.

---

**2.5 GATE: does the data behave as expected?** `GATE` ✅ PASSED 2026-08-27

- [x] Passed

**Stop if:** duplicates exist with large target spread, or kNN residual spread in dense regions is far above the numerical floor.

**Then:** find out why before building anything. Something about the data is not what you think it is, and every downstream number would inherit that misunderstanding.

**Neither condition triggered.**

- **Duplicates with large target spread: no.** 28 near-identical pairs, median target difference exactly 0.000000 for both targets, worst case below Table 7's RMSE for the same metric (2.2).
- **Residual spread above the floor: no.** Intercepts of −0.000003 and 0.001834 against bars of 0.0012 and 0.0102 (2.3).

**What this buys downstream.** The aleatoric term now has a known true value of approximately zero, so when the ensemble's variance head reports one, there is a reference to check it against. Without this measurement a wrong variance head and a genuinely noisy dataset would look identical. It is also what makes the manufactured-aleatoric experiment in 2.4 clean: the injected noise will sit well above a natural floor that is at zero.

**Still open in Stage 2:** the mirror-pair search folded into 2.2. It answers the 1.7 leftover and nothing downstream depends on it.

---

## Stage 3. Target and split

---

**3.1 Primary target** `SETTLED` ✅ 2026-08-27

- [x] Decided

**Decided:** `edge_rotational_transform_over_n_field_periods`.

**Options:** `edge_rotational_transform_over_n_field_periods` / `vacuum_well` / `log10(qi)` / other

**Why:** Passes all four filters: requires the equilibrium solve (so deferral has stakes), tidy range with no heavy tail, appears as a constraint in all three benchmark problems, and is not defined as a max or min over a surface (so no kinks). It is the low-risk choice for week one, and swapping the target later is one line.

**Confirmed by the day 1-2 grid**, `scripts/day_1_2_grid.py`, so this is now evidence rather than a prior. On the winning split (aspect ratio, tail-low) it shows an out/in RMSE ratio of **3.02**, against 1.53 for log10 qi on the same split. The stronger gap means more signal for the calibration study to work with.

**In-region sanity anchor:** RMSE 0.0138 against Table 7's 0.006. About 2.4x, which is what one untuned MLP should give against a tuned ten-member ensemble. log10 qi lands at 0.143 against 0.051, a factor of 2.8. A consistent factor across two unrelated metrics is evidence the pipeline is right, since a loader or scaling bug would not produce that.

**Blocks:** everything downstream

**My decision:**
> Edge rotational transform, and now for a measured reason rather than a prior. It shows a 3x error gap on the winning split where qi shows 1.5x, so it gives the calibration work more to bite on. The in-region number also sits a consistent 2.4x above Table 7 across both targets, which tells me the shortfall is ensemble and tuning rather than a pipeline bug.

---

**3.2 Secondary target** `OPEN`

- [ ] Decided (defer this until end of week 2)

**Options:** `log10(qi)` / none

**Leaning:** log10(qi) if time allows

**Why:** It is the metric the dataset is named around, it spans decades so relative error is the natural cost, and A.4 reports `log_10_qi` so you have a comparison number. Reporting two targets lets you say something comparative: epistemic dominates on the smooth metric, the variance head earns its keep on the rough one.

**Evidence from the day 1-2 grid (2026-08-27):** viable, and more interesting than expected. log10 qi degrades on **both** tails (1.53 low, 1.75 high) where edge rotational transform degrades sharply on one only (3.02 low, 1.14 high). So the two targets fail differently, which makes the comparative claim in the Why above concrete rather than speculative. Its in-region RMSE of 0.143 against Table 7's 0.051 tracks the same 2.4 to 2.8x factor as the primary target, so nothing about it looks broken.

**Still deferred to the week 2 gate.** The evidence says it is worth doing, not that there will be time. Per 6.1 it gets its own ensemble, never its own N-sweep.

---

**3.3 Target transform and scaling** `SETTLED`

**Decided:** Log10 for qi. Z-score the target using training-set statistics only.

**Why:** qi spans more than two orders of magnitude and is strictly positive. A.4 did the same. Training-set statistics only, or you have leaked test information into the scaling.

---

**3.4 Split axis** `SETTLED` ✅ 2026-08-27

- [x] Decided

**Decided: aspect ratio.** Confirmed by the day 1-2 grid, `scripts/day_1_2_grid.py`, not by the reasoning below, which was a prior.

**max_elongation is disqualified on coverage, not on gap.** It does produce error gaps (1.91 tail-low, 1.26 tail-high on the primary target), but it cannot support the calibration figure:

| axis | cut | min bin count | achievable δ |
|---|---|---|---|
| aspect ratio | tail-low | 276 | 0.059 |
| aspect ratio | hole | 641 | 0.039 |
| max elongation | tail-low | 12 | 0.283 |
| max elongation | tail-high | **0** | ∞ |

Elongation has a pathological outlier tail: its tail-high held-out set reaches 37 standard deviations from the training region, against 2.13 for aspect ratio's tail-low. That spreads the held-out points so thinly that an equal-width distance bin comes out **empty**, and a bin with no points cannot yield a coverage rate at any tolerance. Aspect ratio clears both required bands comfortably; elongation clears neither.

**This is the disqualification rule from 3.7 firing exactly as designed.** Worth noting in the write-up: the axis with the second-largest gap was rejected for a reason that has nothing to do with gap size, which is what the two-condition pass criterion existed to catch.

**Options:** aspect ratio / max elongation / high mode-number spectral energy / PCA direction

**Leaning:** aspect ratio

**Why:** Physically legible, an explicit constraint in all three benchmarks, and measurable from the inputs alone so it is a shift in the questions rather than in the answers. Never split on the target, which would confound covariate shift with label shift.

**External corroboration (found during the 0.5 search, 2026-08-26).** The premise that the compact end is undersampled is no longer just our reading of how the data was generated. Two independent papers treat it as established: the domain-adaptive latent diffusion work (arXiv 2608.16938) states it targets "the sparsely sampled low-aspect-ratio regime," and "Data-Driven Generation of Compact Quasi-Isodynamic Stellarators" (2026) works the same region. Both are generative rather than UQ work, so they do not compete, they corroborate. Useful in two places: it makes the split-axis justification citable rather than inferred, and it makes the deferral curve more pointed, since researchers are actively generating candidates in exactly the region where a confidently-wrong surrogate is most dangerous. **Caveat:** both work at NFP=4 while we are at NFP=3, so this supports the shape of the design space, not our specific slice. Say it that way.

**Options removed:** generation pathway is a constant column after 1.3, field period after 1.2. Neither is a usable fallback any more.

**Input-measurability confirmed (2026-08-27, gate 3.5).** Aspect ratio is predictable from the 80 coefficients at R² 0.988, with the shortfall from 1.0 traced to model budget rather than missing information. The "measurable from the inputs alone" clause in the Why above is now checked rather than assumed, so the axis is safe from the label-shift confound. Full numbers and the threshold caveat are in 3.5.

**Early evidence for the compact end, from the same run.** Prediction error on aspect ratio itself tracks data density: lowest in the dense band around A 9.9 to 10.1, elevated at both the sparse upper tail and the compact end. Weak evidence, since it concerns a geometric quantity rather than a solver metric, but it is consistent with the sampling-density premise and costs nothing to mention.

**How it gets decided:** not from reasoning, from the day 1-2 grid. Aspect ratio and max elongation are checked together, both directions each, against both candidate targets. Winner is whichever (axis, direction) pair shows the strongest real in/out error gap AND clears the coverage threshold in both a mid-range band and a tail band, since the main figure needs the hole and the tail results both. If aspect ratio and max elongation tie, take aspect ratio for its tighter link to the paper's compactness narrative (Section 3.3). If nothing shows a gap, reconsider PCA direction or high mode-number spectral energy before going further.

**Blocks:** 3.5, 3.6, 3.7

**My decision:**
>

---

**3.5 GATE: is the split axis really a function of the inputs?** `GATE` ✅ PASSED WITH CAVEAT 2026-08-27

- [x] Passed

**Check:** predict `metrics.aspect_ratio` from the raw coefficients. It should come out near machine precision.

**If yes:** the leakage objection is dead and you can say so in one sentence.
**If no:** you have misunderstood how aspect ratio is defined here. Resolve before continuing, or pick a different axis.

**Run:** `scripts/gate_3_5_split_axis.py`, 27,050-row pool, 80/20 split, seed 0. Inputs are the 80 coefficients exactly as the surrogate will see them, so R(0,0) is excluded per 1.5.

| model | R² | RMSE |
|---|---|---|
| RidgeCV (linear) | 0.784 | 0.761 |
| HistGradientBoosting, 100 iters | 0.983 | 0.216 |
| HistGradientBoosting, 1000 iters | 0.988 | 0.179 |
| MLP, A.4 architecture (256x3, tanh) | 0.983 | 0.213 |

std of y is 1.639 for scale.

**Verdict: aspect ratio is input-measurable. Use it as the split axis.** Splitting on it is a shift in the questions asked, not in the answers, so the leakage objection is answered.

**Supporting diagnostics, all pointing the same way:**

- **Model class, not missing information.** Ridge at 0.784 versus 0.988 for nonlinear models. Aspect ratio is roughly R(0,0) over the minor radius, a ratio, and a linear model cannot represent division.
- **Not outliers.** The 0.05% trim drops 3 test points and moves R² by 0.00002.
- **Not the dropped R(0,0).** Correlation between ridge residual and R(0,0) is +0.075, essentially nothing.
- **Not converged.** Every increase in training budget raised the number (GBM 0.983 to 0.988 on iterations alone; the MLP early-stopped at 116 then 179 iterations, never approaching its cap). Budget-limited, not information-limited, which is the signature of a mapping that exists.
- **Error is diffuse, not structural.** Median absolute error 0.072, p90 0.265.

⚠️ **The caveat: it scored 0.988 against a pre-registered 0.99, so it technically missed.** The bar was set from the information floor implied by the dropped R(0,0). That was the wrong instrument twice over. First, an information floor bounds what *any* model could know; it says nothing about what a quick untuned model reaches on 21k points, so it conflated knowable with learnable. Second, the gate's question is qualitative, input-measurable versus solver output, and a single absolute R² cannot answer it without a reference for what each category looks like on this data.

**Decided: record the miss, do not move the bar.** The threshold stays 0.99 in the script and this entry documents why it was not met. A pre-registered miss that can be explained is stronger evidence of honest method than a threshold quietly relaxed to 0.98 after the fact.

**The better test, not run, noted for completeness.** Fit the same model on a solver-dependent target (`edge_rotational_transform_over_n_field_periods`) and compare. If aspect ratio is clearly better predicted, the objection dies by contrast rather than by an absolute number. Skipped as the gate had already served its purpose and the day 1-2 grid produces those numbers anyway.

**If the attribution question resurfaces:** one run with R(0,0) added back to the inputs cleanly separates "missing input" from "model budget". Not needed now.

**Independent support:** the dataset documentation already lists `aspect_ratio` among the metrics computable from the coefficients without solving. This gate corroborates a documented fact rather than discovering one, which is why 0.988 is sufficient.

**Incidental finding, feeds 3.4 and Stage 7.** Mean absolute error by aspect ratio decile is lowest where the data is densest (0.064 at A 9.92 to 10.14) and highest in the sparse upper tail (0.210 at A 10.14 to 12.05), with the compact end also elevated (0.096 at A 4.07 to 6.05). The extrapolation premise is visible in a purely geometric quantity, before any surrogate has been built.

---

**3.6 Split direction** `SETTLED` ✅ 2026-08-27

- [x] Decided

**Decided: low. Hold out the compact, low-aspect-ratio end.**

**Options:** hold out low aspect ratio (compact) / hold out high aspect ratio

**The grid settles it decisively.** Out/in RMSE ratio on the primary target, aspect ratio axis:

| direction | ratio | d_max | min bin | δ |
|---|---|---|---|---|
| **tail-low (compact)** | **3.02** | 2.13 | 276 | 0.059 |
| tail-high | 1.14 | 1.35 | 196 | 0.070 |

Holding out the compact end triples the error. Holding out the high end costs 14%, which is barely a shift at all. Both bands have adequate coverage, so this is a genuine difference in generalization rather than an artifact of test-set size.

**The leaning was right, and now it is measured.** Convenient rather than lucky: the compact end is both the commercially interesting frontier (Section 3.3's compactness versus coil complexity trade-off) and the direction that actually produces a gap. Had they disagreed, the evidence would have won.

**Corroboration lines up too.** The two independent papers in 3.4 describe the low-aspect-ratio region as sparsely sampled and are generating there. The 3.02 ratio is what that sparsity looks like from the surrogate's side, which strengthens the deferral argument: people are pushing into exactly the region where a confidently wrong surrogate does the most damage. Caveat unchanged, both work at NFP=4 while we are at NFP=3.

**Why:** Compact devices are the commercially interesting frontier, and the paper's Section 3.3 makes the compactness versus coil complexity trade-off explicit. But Table 6 shows aspect ratio targets were drawn uniformly on 4 to 12 while A was an upper-bound constraint during optimization, so achieved values will not be uniform.

**Updated:** direction is no longer decided separately from axis. Both directions for both candidate axes go into the same day 1-2 grid (see 3.4), so the direction falls out of the same evidence rather than being chosen first and checked second. The old fallback wording, "flip to the high end and change the story," is superseded: if aspect ratio fails in both directions the fallback is max elongation, not a narrative change.

**My decision:**
>

---

**3.7 Threshold and held-out fraction** `SETTLED` ✅ 2026-08-27

- [x] Decided

**Decided: 20% held out, 8 distance bins, δ = 0.06.**

At `test_fraction = 0.2` on the winning axis and direction, the thinnest equal-width distance bin holds **276** points, which supports δ = 0.059 by n = z²p(1−p)/δ² solved for δ. The mid-range band does better still, 641 points and δ = 0.039. Both required bands clear, and comfortably better than the δ = 0.10 the method note below hoped for.

**Do not tighten δ to 0.039 just because the hole allows it.** The figure compares the two bands, so the reportable tolerance is set by the weaker of them. Quote δ = 0.06 for both.

**Hole placement RESOLVED (2026-08-29): centre the hole at percentile 30, spanning p20 to p40.** `scripts/hole_placement.py`.

The geometry problem this run exposed was that the hole at the median reached only **0.20** standard deviations from the training region while the tail reached **2.13**, a factor of ten. The main figure's sharpest claim is "at matched distance, leaving the region costs more than filling a gap," and at the median that comparison existed only over 0 to 0.20, where the tail has barely begun to degrade. The three options were widen the hole, accept the weaker claim, or move the hole somewhere sparser.

**Moving it won.** p30 sits immediately above the tail's p0 to p20 without overlapping it, so the two experiments never share a held-out configuration and the held-out sizes stay matched at 5,404 each. It reaches **0.45**, more than doubling the comparison range, with 583 points in the thinnest distance bin and δ = 0.041. It is the widest reach available to a non-overlapping hole, so 0.45 is the ceiling on the matched-distance claim unless the held-out fraction changes.

**δ = 0.06 is unaffected.** It was always set by the tail's 276-point bin, and that band did not move.

⚠️ **The sweep revised a headline finding.** The day 1-2 result had the hole at the median costing nothing, 0.95 against the tail's 3.02, which read as clean interpolation versus failed extrapolation. That held for one placement only. Across centres the penalty is a **U in percentile**, lowest in the dense middle and rising at both ends: at 20% held out, 1.96 / 1.80 / 1.12 / 0.95 / 0.90 / 0.80 / 0.82 for centres p20 through p80; at 10% held out, which is the only run that reaches p90, 1.79 / 1.47 / 1.48 / 0.94 / 0.82 / 0.73 / 0.78 / 0.75 / 0.91. Ratio tracks reach almost monotonically, so gap **width** drives most of the penalty and sparsity matters because it makes gaps wide.

⚠️ **A 20% band cannot be centred above p80.** It runs off the end of the data and stops being an interior hole. Reaching p90 needs the 10% sweep, and p95 to p100 can host no interior hole at any width. That region is tail-high territory only.

⚠️ **Ratios below 1.0 are expected, not anomalous.** The denominator is a random in-region slice drawn from the whole training region including its thin compact end, while a hole in the dense middle contains none of those hard cases. A ratio of 0.80 means filling that gap was easier than the model's average job, not that removing data helped.

⚠️ **Two rows carry the U's upper arm and they are the weakest in either table.** The 10% sweep's p90 and tail-low bins hold about 130 points against roughly 310 elsewhere, at a single seed. Before leaning on the asymmetry, rerun p30 and p90 at seeds 0, 1, 2. Six fits, about six minutes.

**Options:** fixed percentile / fixed aspect ratio value / sized to hit a target test-set count

**Decided (method):** size it to hit the per-bin coverage requirement. Required points per bin follows the standard proportion formula, n = z² · p(1−p) / δ², with z = 1.96 for 95% confidence and a conservative p = 0.5 (worst-case variance, so the answer is valid regardless of which nominal coverage level you check). 8 to 10 distance bins.

**Still open:** δ itself, the tolerance on the coverage estimate. Roughly: δ = 0.15 needs ~44 points per bin, δ = 0.10 needs ~100, δ = 0.05 needs ~400. At 8 to 10 bins, δ = 0.10 means 800 to 1000 held-out points in total. Pick δ from the day 1-2 histogram once you can see what is actually achievable, not blind beforehand.

**Must clear twice:** independently for a mid-range band (interior hole) and a tail band. Failing either disqualifies the axis. This is a property of the histogram, not a partition, the three splits are separate experiments that each train on the full ~23k minus their own held-out region, so they never compete for data at the same time.

**Hole placement:** by percentile, centred on the median, not by axis value. Achieved aspect ratio is skewed because A was an upper-bound constraint during generation, so a value-space band can land somewhere sparse while a percentile band guarantees mass.

**My decision:**
> Work backwards from what the calibration figure needs instead of picking a threshold out of the air. The formula fixes the method now, the tolerance waits until I can see the histogram, because choosing δ blind risks finding out the tail cannot support it and redoing the decision anyway.

---

**3.8 Three-condition design** `SETTLED`

- [x] Decided

**Decided:** All three. Random (plain train/test, the easy baseline, in-domain only), interior hole (a contiguous gap carved from the middle of the split axis with training data on both sides), tail (the extreme end held out entirely). Same model and recipe for each.

**Why:** Same model, same recipe, three splits. Anyone can report the random split. The distance between the interior-hole result and the tail result is the actual contribution, and it separates "the model cannot fill gaps" from "the model cannot leave the region." This is the main figure.

**Why it matters beyond the project:** the two failure modes have different fixes. Interior-hole failure means sparse pockets can be backfilled with a few more optimization runs. Tail failure means no amount of nearby backfilling helps and every query near the frontier needs a real solver call. That is a different resourcing answer for an R&D team, and it is the one that ties the calibration result to the deferral curve.

**First measurement (2026-08-29), `scripts/distance_error.py`.** Three splits, one MLP each, per-point error binned by distance from the training region. This is deliverable one in single-model form; the ensemble later adds the uncertainty bands without changing the figure's shape.

| distance out | interior hole at p30 | tail-low |
|---|---|---|
| 0 (in region) | 0.0145 | 0.0138 |
| ~0.22 | 0.0243 to 0.0280 | 0.0303 |
| ~0.40 | 0.0325 to 0.0336 | 0.0376 |
| 1.83 | no points | 0.0577 |

RMSE on edge rotational transform per field period, whose pool std is 0.0786.

**At matched distance the tail costs about 15% more than the hole**, consistently at both 0.2 and 0.4 standard deviations out. So most of the headline gap, 3.02 against 1.80, is **distance rather than edge**: the tail split simply asks about configurations much further from the data. What survives after controlling for distance is the edge premium itself, and it is real but modest. That is a weaker claim than "the model interpolates for free and cannot extrapolate," and a considerably more defensible one.

**The far tail number is the one to quote in a pitch.** At 1.83 standard deviations out, RMSE is 73% of the target's own standard deviation. Predicting the dataset mean and ignoring the input entirely would score 100%. The surrogate is barely beating nothing out there while still returning a confident-looking number, which is the deferral argument in one line.

⚠️ **Two measurement choices that would otherwise flatter the result.** Distance is computed against the fit set, not the training mask; using the mask would include the in-region slice in its own reference set and hand those points distance 0 by construction rather than by measurement. And the in-region slice is a single anchor at distance 0, not part of the binned curve: it is half the evaluation set, so quantile binning over the combined set spent three of eight bins stacking it on the y axis.

**My decision:**
> Run all three. A random split alone just reproduces Table 7. The interesting number is the distance between the hole result and the tail result, because that separates "cannot interpolate into gaps" from "cannot extrapolate past the edge," and those two problems need different responses.

---

**3.9 Validation set placement** `SETTLED`

**Decided:** The validation set used for early stopping is drawn from in-region data only.

**Why:** Including out-of-region points in it leaks the extrapolation condition into training and quietly flatters the result.

---

**3.10 Input scaling** `SETTLED`

**Decided:** Fit the input scaler on training data only. Do not clip.

**Why:** Out-of-region inputs will fall outside the fitted range. That is correct and expected, and it is the whole point. Clipping would hide the shift from the model.

---

**3.11 Distance from the training region** `SETTLED`

**Decided:** Distance along the split axis to the nearest training point, normalized by a split-independent constant: the split axis's standard deviation over the full filtered dataset.

**Why not full 80-dimensional input distance:** raw Euclidean distance across the coefficients would be dominated by the low-order modes, and it would break the causal story. The whole design manipulates one axis deliberately, so degradation should be measured along that same axis. Full input distance moves for reasons unrelated to the manipulation.

**Why a dataset-level denominator, not "the training region's own width":** that was the original wording and it does not survive the three-condition design. The hole's training region spans nearly the whole axis while the tail's stops at the cutoff, so normalizing each to its own width puts the two figures on different scales and breaks exactly the comparison the main figure exists to make. A constant denominator means degradation at 0.5 in one split is directly comparable to 0.5 in the other.

**One formula covers all three splits:** tail is distance past the cutoff, interior hole is distance from the nearer edge peaking at the hole's centre, random is ~0 everywhere, which is why the random split gets no distance-binned figure.

**Expected shape:** the hole's x-range is short and the tail's is long. That is the geometry, not a defect, and it is what supports the claim "at matched distance, leaving the region costs more than filling a gap."

**My decision:**
> Measure distance along the split axis only, since that is the thing I actually manipulated. Normalize by a dataset-level constant so the hole figure and the tail figure land on the same scale, otherwise the main comparison is meaningless.

---

## Stage 4. Model

---

**4.1 Architecture** `SETTLED`

**Decided:** Follow Appendix A.4: three layers, 256 hidden units, tanh activations.

**Why:** Proxima found this by Bayesian optimization over depth, width, activation, and learning rate. Do not spend two days rediscovering it.

⚠️ **Comparability caveat, do not overstate this.** The paper never says whether A.4's ensemble is one shared network with a twelve-metric output or twelve separate single-metric ensembles built from the same architecture template. "The MLPs mapped coefficients to target key metrics" is genuinely ambiguous, and the A.4 training code is not in the public repo (checked directly). What is certain is our side: single-target, one mean-variance network per metric. So the claim is same-metric test performance, not a verified same-or-different architecture claim in either direction. Saying "I used their architecture" is fine; saying "I reproduced their model" is not.

---

**4.2 Ensemble size** `SETTLED`

- [x] Decided

**Decided:** 10, matching A.4.

**Why:** A.4 used ten. Five is a citation habit, not a principle, and spread estimated from five samples is noisy enough that per-point epistemic bounces around. On 80 inputs with small MLPs, ten costs almost nothing.

**My decision:**
> Ten, same as A.4. Epistemic is the spread across members, and five members give a spread estimate too noisy to read per point. Ten small MLPs is cheap.

---

**4.3 Diversity mechanism** `SETTLED`

- [x] Decided

**Decided:** Different init plus shuffle order only. No bootstrap resampling.

**Why:** Bootstrapping means each member sees roughly 63% of N, which changes the effective training size per member and confounds the epistemic-versus-N sweep in Stage 6.

**Blocks:** 6.1

**My decision:**
> Init and shuffle order only. Bootstrapping would mean each member trains on about 63% of N, so the N on the x-axis of the sweep would not be the N each member actually saw, and the curve would be uninterpretable.

---

**4.4 Loss and variance-collapse fix** `SETTLED` ✅ 2026-08-29

- [x] Decided

**Options:** plain Gaussian NLL / MSE warm-up plus variance floor / β-NLL / warm-up plus floor, with β-NLL in reserve

**Decided: MSE warm-up of 25 epochs, then Gaussian NLL, with the predicted variance floored at 1e-6 in z-scored space. β-NLL held in reserve behind a stated trigger.** Values live in `src/constellaration_uq/nets.py` as module constants alongside the rest of the frozen recipe (4.7).

**Why:** The NLL gradient on the mean is scaled by 1/σ², so points the model fits poorly get high predicted σ and are then down-weighted, which stalls learning exactly where it is most needed. Note this is not really "variance collapse" in the σ→0 sense; calling it that in an interview invites a correction. β-NLL (Seitzer et al. 2022) counteracts it directly if warm-up plus floor is not enough. If you go to β-NLL, β becomes another choice.

**Where 25 comes from.** The step 0 history in `results/hp_check.json`, for the frozen recipe. Its validation loss is within 3x of its best by epoch 5, 1.5x by epoch 18, 1.2x by epoch 32, and it stops at 75. So 25 lands where the mean is substantially learned, with two thirds of the epoch budget left for the NLL phase. There is a range of defensible answers here, roughly 20 to 40, and what matters is that the number is fixed, documented, and identical at every N in the sweep. Making it a fraction of the run would vary it with N and break the recipe freeze.

**Where 1e-6 comes from.** Training happens in z-scored space, where the target has unit variance, so 1e-6 is six orders of magnitude below the signal. It is safely above float32 resolution, so 1/variance cannot explode, and in physical units it is a standard deviation of 0.00008 on a target whose own spread is 0.0786. That is far below anything Stage 2 could resolve, which is the point: **the floor must be too small to manufacture an aleatoric term**, because a floor that binds would be read by the step 5 hidden-coefficient check as a working variance head.

**β-NLL trigger, so this is not an open-ended option.** Adopt it if either symptom appears in the first mean-variance ensemble: the predicted variance sitting pinned on its floor across most of the in-region data, or training destabilising once warm-up ends. Both are visible in one run. If neither appears, β-NLL stays unused and is reported as considered-not-needed rather than untried.

**No separate tuning run for these.** Step 5, the hidden-coefficient check, already asks the only question that matters here: does the variance head recover a magnitude that was deliberately injected. A second check for the same thing is scope this budget does not survive.

⚠️ **Reorder steps 2 to 5 because of this entry.** The original order built three mean-variance ensembles and then checked the variance head. If the loss is wrong that means redoing three. Run the random-split ensemble first, then the hidden-coefficient check on it, then the hole and tail ensembles. A bad loss then costs one ensemble instead of three, and the reordering costs nothing.

**My decision:**
> Warm-up 25 epochs, floor the variance at 1e-6, and write the β-NLL trigger down as an observable symptom rather than leaving it as a vague reserve. Then move the variance-head check to immediately after the first mean-variance ensemble, so if any of this is wrong I find out after one run rather than three.

---

**4.5 Variance parameterization** `SETTLED`

- [x] Decided

**Decided:** Predict log-variance, with a floor. Softplus held in reserve.

**Why:** Simplest, numerically stable, standard deep-ensemble choice, and it pairs directly with the MSE-warm-up-plus-variance-floor loss in 4.4 since the floor is a one-line clamp before exponentiating.

**Implementation note:** isolate it behind one small function, raw network output in, positive variance out. Everything else calls that function rather than exp or softplus directly, so a future swap is a one-function edit. Not a config field, this is architecture and the config object is scoped to things that vary per run (see 4.8).

**My decision:**
> Log-variance with a floor. It is the standard choice, exp keeps it positive automatically, and the floor drops straight in. Putting it behind one function means switching to softplus later costs one edit instead of a search through the codebase.

---

**4.6 MSE-only baseline** `SETTLED`

- [x] Decided

**Decided:** Train one, before the mean-variance ensemble. Same architecture minus the variance head, single output, MSE loss, ten members.

**Why:** Cheap, and reproducing A.4's Table 7 numbers in-region is both your sanity anchor and your comparability claim. If your MSE ensemble does not land near their R² and RMSE, your pipeline has a bug and you want to know before you build three splits on top of it.

**Two distinct reasons, worth keeping separate.** Comparability: A.4 trained with MSE, so their Table 7 RMSE comes from a model with no variance head. Comparing a mean-variance ensemble's RMSE against it compares two different training objectives, the plain MSE run is the like-for-like number. Debuggability: if it matches Table 7, the loader, NFP and pathway filters, trimming, scaling, and architecture are all verified in one run, so later degradation is attributable to the variance head or the NLL loss rather than the pipeline. Without it a bad result has two possible causes at once.

**Cost:** near zero. Same code path as the MSE warm-up phase the locked loss (4.4) already requires.

**Pass condition, pre-registered 2026-08-29 before the run.** Ensemble RMSE on the random split's held-out set below **0.0105**, and better than any individual member.

**Where that number comes from.** Table 7 reports 0.006 for this metric. The sklearn single MLP gave 0.0138 and the frozen PyTorch recipe gives 0.01149 for one network (4.10). Averaging ten members should improve on a single member comfortably, so 0.0105 is a modest bar rather than a stretch, and it still sits about 1.75x above Table 7, which is expected against a tuned ensemble. Missing it points at the pipeline, which is the only thing this run exists to test.

⚠️ **Registered before seeing the result on purpose.** Gate 3.5 set its bar from an argument that did not apply, missed it, and the miss was recorded rather than the bar moved. That only works if the bar exists first. Judging afterwards always produces a pass.

**RESULT 2026-08-29, `scripts/mse_ensemble.py`, ten members in 454s: MISSED by 0.2%.** Ensemble RMSE **0.01052** against the 0.0105 bar. The second condition passed cleanly: 0.01052 against a best member of 0.01223, a 14% improvement, so the ensemble is not being carried by one lucky member. Per-member RMSE spans 0.01223 to 0.01304 across ten seeds, with nothing pathological.

⚠️ **The bar was badly calibrated, and that is the finding, not a pipeline defect.** 0.0105 was derived as "a modest improvement over the single network's 0.01149" and rounded, with no uncertainty attached. The run landed 0.00002 away, inside any sensible error bar on a number chosen that way. Same failure as gate 3.5's 0.99: a threshold that sounded principled and was not. The bar is recorded as missed and is **not** moved retroactively, and the run was **not** repeated with a different seed until it passed.

**Proceeding anyway, on the evidence rather than on the verdict.** 1.75x Table 7 on RMSE, against their tuned ensemble, after one afternoon of bounded tuning. The sklearn single MLP was 2.3x, so the PyTorch port plus ensembling closed roughly a third of that gap. A pipeline bug does not look like 1.75x; it looks like 5x, or R² near zero.

⚠️ **R² is NOT comparable to Table 7 and must not be reported as if it were.** Ours is 0.98181 against their 0.997, which reads far worse than the RMSE ratio implies. R² depends on the test set's own spread: our target has std 0.0786, while back-solving std = RMSE / NRMSE from their table gives about 0.115. Same RMSE, different denominator, different R². **RMSE is the comparable number; R² is not**, and 5.3 already restricts the comparability claim to RMSE and R² jointly, which this narrows further.

**Incidental, and it is the baseline for steps 3 and 4:** in-region epistemic spread is 0.00631, which is 8% of the target's standard deviation and about 60% of the ensemble's total error. The members genuinely disagree even where the data is dense. That number has to grow off-distribution or the project has no subject.

**My decision:**
> Train the plain MSE version first. It is what A.4 actually built so it is the fair comparison, and it verifies the whole pipeline in one cheap run. After that, if the mean-variance means come out worse, I know the cause is the variance head and not the data handling.

---

**4.7 Recipe freeze** `SETTLED`

**Decided:** Architecture, epochs, and stopping rule are fixed before Stage 6 begins.

**Why:** If you train longer at larger N, or early-stop against a validation set that grows with N, you have confounded "more data" with "more optimization" and the N-sweep cannot be attributed to either.

**FROZEN 2026-08-29**, from the 4.10 sanity check. Live in `src/constellaration_uq/nets.py` as module constants, deliberately not config fields (4.8).

| setting | value | source |
|---|---|---|
| architecture | (256, 256, 256), tanh | 4.1, Appendix A.4 |
| optimizer | Adam, no weight decay | ours, A.4 does not say |
| learning rate | 1e-3 | 4.10 |
| batch size | 128 | 4.10 |
| early-stopping validation | 500 points, fixed count | 4.10, confirmed at the sweep floor |
| max epochs | 500 | cap, not a target |
| patience | 20, restoring best weights | ours |

⚠️ **The chosen learning rate is not the sanity check's nominal winner, and that is deliberate.** 3e-4 at batch 128 scored best_val 0.02119 against 1e-3's 0.02138, a 0.9% difference at a single seed, which is noise. It cost 210 epochs against 75. The sweep trains 300 networks, so that is roughly 3.7 hours against 1.3 on CPU for a difference that would not survive a second seed. Ties on accuracy are broken on cost, and the reasoning is recorded here so the table's apparent winner does not look like an error later.

**Batch 512 lost outright** across all three learning rates, which is the one unambiguous result in the check.

**500 validation points confirmed sufficient, including at the sweep floor.** Jitter, the mean absolute epoch-to-epoch change in validation loss over the last 20 epochs relative to the best, was 0.030 for the chosen configuration at full training size and 0.010 at N=1000. The stopping rule is picking a real minimum, not noise, at both ends of the sweep.

**Incidental, and it is the sweep's premise appearing early:** at N=1000 the same recipe gives RMSE 0.045 against 0.0115 at full size, a 4x degradation from data volume alone.

**Blocks:** 6.1
**Blocked by:** 4.10 `RESOLVED`

---

**4.8 Config-driven code, lightly** `SETTLED`

**Decided:** A single typed config object carrying target column, split axis, split direction, and seed, threaded through one entry point. No external config framework, no Hydra, no YAML. Architecture and training hyperparameters stay fixed constants, not config fields.

**Why:** Those four fields are exactly what varies across runs, and the day 1-2 grid needs to loop over combinations of them without duplicating code. A full framework is engineering overhead that buys nothing at this size and is more to defend in an interview. Keeping architecture out of the config is deliberate: the recipe is frozen by 4.7, so making it configurable would imply things are tunable that the design says must not move.

**My decision:**
> One dataclass with the four things that actually change per run, passed into one function. The grid check just builds a config per combination and calls it. Anything heavier is infrastructure I would have to justify without it doing any work.

---

**4.9 No joint multi-head network across targets** `SETTLED`

**Decided:** One target per ensemble. The target column is a swappable parameter, so the same architecture and recipe rerun unchanged per target. No shared trunk with multiple heads.

**Why:** A shared trunk confounds the decomposition. Ensemble disagreement on one target would be entangled with how well the network also fits the other, and the epistemic/aleatoric split is the thing the whole project rests on. It also reopens the frozen-recipe question, since a joint loss needs weighting between two differently-scaled targets and its own stopping rule. There is no efficiency win to offset that: two separate ensembles cost compute, which is cheap, while a joint model costs a clean story, which is not.

**My decision:**
> Separate ensembles per target. Sharing a trunk would mean member disagreement on one target partly reflects the other target, which muddies exactly the quantity I am claiming to measure. Rerunning the same recipe on a different y-column is nearly free by comparison.

---

**4.10 Hyperparameter sanity check** `SETTLED` ✅ 2026-08-29

- [x] Decided

**Decided: one bounded check, six single-network fits, run before anything else in the ensemble phase. Not hyperparameter optimization.**

**Why it is needed at all.** Appendix A.4 specifies three layers, 256 units, tanh, MSE loss, z-scored targets and ten members. It says nothing about optimizer, learning rate, batch size, epoch budget, weight decay or initialisation, and the A.4 training code is not in their public repo (established in 5.3), so those values cannot be looked up. They have to be chosen here and declared as ours in the write-up.

**Why not full HPO.** The contribution is the uncertainty study, not accuracy. The recipe is frozen before the sweep anyway (4.7), so tuning is a one-shot activity rather than an ongoing one, and a tuning rabbit hole is exactly the kind of scope that a 2.5 week budget does not survive.

**Why not zero tuning either.** A bad learning rate can cost an order of magnitude. The MSE ensemble (4.6) exists to verify the pipeline, and if it lands far from Table 7 with the learning rate never varied, the result is ambiguous between a pipeline bug and a bad optimizer setting. That would defeat the only reason 4.6 exists.

**The check.** Three learning rates by two batch sizes, six fits, single networks and not ensembles. Starting box: Adam, lr in {3e-4, 1e-3, 3e-3}, batch in {128, 512}. Select on in-region validation loss. An afternoon at most.

⚠️ **The serious trap.** Run the check on the **random split only**, and select on the **in-region validation set** only. Selecting hyperparameters by looking at tail-split performance leaks the extrapolation condition into the model choice and invalidates the entire study, in a way no reviewer could detect from the results. This is the same failure mode as recalibrating on out-of-region data, and it is easier to commit by accident.

**Then freeze.** The chosen values get written into 4.7, which currently declares the recipe frozen without recording what it froze, and they do not move again.

**Where it lands in the execution order:** step 0, before the plain MSE ensemble.

**My decision:**
> Check the one knob that can cost an order of magnitude, on the split that cannot leak, then stop. Six fits buys the right to read the Table 7 comparison as a pipeline check rather than as a question with two possible answers.

---

## Stage 5. Decomposition

---

**5.1 Definitions** `SETTLED`

**Decided:** Epistemic is the spread of member means. Aleatoric is the mean of member variances. Total is their sum.

**Why:** Standard law-of-total-variance decomposition for an ensemble. Both terms come out directly, so no subtracting one from the other. Note for the write-up: this is a decomposition of the ensemble mixture, not of a Bayesian posterior, and the split is relative to model class and input representation rather than absolute.

---

**5.2 Reporting space** `SETTLED`

- [x] Decided

**Decided:** Physical units. Precisely: un-z-scored, with any monotone transform chosen for modelling reasons left in place.

**Why the distinction matters:** z-scoring exists only to help optimization and carries no meaning, so it always gets undone. The log10 on qi is a deliberate choice about what being wrong should cost, so it stays. Practically, edge rotational transform reports in its actual units and qi reports in log10(qi), never raw qi.

**Why not raw qi**, three reasons, any one sufficient: (1) qi spans ~2 orders of magnitude, so raw-qi RMSE would be dominated by the high end and a uniformly good model would score badly for reasons unrelated to its quality; relative error is the natural cost here and log space is where relative becomes absolute. (2) The predictive distribution stays Gaussian under un-z-scoring, which is an affine rescale, so the closed-form Gaussian CRPS in 8.3 holds; exponentiating to raw qi makes it log-normal and that formula no longer applies. (3) Table 7 reports `log_10_qi`, so comparability requires the same space.

**Also note:** Table 7's RMSE is itself in physical units, not z-scored. Recoverable from their own numbers: if NRMSE = RMSE / std, then std = RMSE / NRMSE per row, which gives ~1.8 for aspect ratio (range 4 to 12), ~0.38 for log10(qi), ~0.115 for edge rotational transform, all plausible. Z-scored data would have std 1, forcing NRMSE = RMSE, which is not what the table shows. So this decision keeps the sanity anchor valid rather than breaking it.

**Blocks:** 8.3

**My decision:**
> Report in physical units, but "physical" means undo the z-scoring, not undo the log. The z-score is a training convenience with no meaning. The log10 is me choosing that being wrong by a factor should cost the same everywhere, and undoing it would silently swap in a different error metric than the one I picked.

---

**5.3 NRMSE and SNR definitions** `SETTLED`

**Decided:** Define them ourselves and say so. NRMSE = RMSE / std(y_true). SNR = var(y_true) / var(residual). Only RMSE and R² are claimed as directly comparable to Table 7.

**Why:** Checked the public repo (`proximafusion/constellaration`) directly. The A.4 MLP ensemble and its NRMSE/SNR code are not published there, so their exact formulas are unknown and cannot be matched by inspection.

**Partial evidence, worth knowing:** our NRMSE convention is probably theirs. Back-solving std = RMSE / NRMSE from Table 7 gives plausible spreads for every metric, while RMSE/range does not fit at all. That is an inference from twelve rows, not documentation, so "consistent with, inferred numerically" is defensible and "matches theirs" is not. SNR stays genuinely unrecoverable: var(y)/var(resid), 1/NRMSE², and a dB reading all fail to produce a consistent factor across rows.

**My decision:**
> State my own formulas plainly and only claim RMSE and R² as comparable. Their code is not published so I cannot verify the rest. NRMSE looks like it lines up when I back-solve it from their table, but that is an inference and I will describe it as one.

---

## Stage 6. Validation phase

---

**6.1 N-sweep design** `SETTLED`

**Decided:** Roughly five values of N, training set size (e.g. 1k, 2k, 4k, 8k, 18k), 3 seeds each, subsampling done within the training region.

**Why:** Subsampling across the whole space varies coverage and N together, so the curve would tell you nothing about either. A fixed test set is what makes the points comparable. Multiple seeds because ensemble spread is itself a noisy estimator.

**Scope is pinned, do not let it multiply.** Tail split only: the random split has no out-of-region set to measure "epistemic rises off-distribution" against, and the interior hole could serve but would double the work to answer a secondary question. Primary target only: the sweep validates the uncertainty machinery, not any one metric, so repeating it on a second target retests the same thing. If the secondary target is pursued it gets its own ensemble, not its own sweep.

**Fixed test sets, frozen across all N:** the tail held-out set for the off-distribution check, plus a fixed in-region slice for the shrinks-with-N check.

**Both input conditions** (see 6.2), so the whole sweep runs twice.

**Budget: 5 N × 3 seeds × 2 input conditions = 30 ensembles, 300 member networks.** Written down deliberately. Read unbounded, across three splits and two targets, this is 180 ensembles and 1800 networks, which does not survive the time budget. Any expansion past 30 is a visible decision, not a drift.

**If the schedule slips, shrink the grid, never drop the sweep** (amended 2026-08-29). The floor is 3 N × 2 seeds × 2 input conditions = 12 ensembles. This decision already commits to reporting the *shape* of the epistemic decay rather than a fitted exponent, and 12 points show the shape. Dropping the sweep entirely is not an option, because nothing else can test whether the epistemic term is reducible with data, which is the only thing that earns it the name. An earlier version of 6.5 named the sweep as the first thing to cut; that was wrong and is retracted.

⚠️ **Do not assume the sweep is expensive before measuring it.** It was described as the largest remaining compute item on the basis of its ensemble count alone. These are small networks on GPU, so 300 of them may well be an hour or two. Time a single member network before the ensemble work starts, and settle the scheduling question with a measurement instead of a guess.

**My decision:**
> Pin it to the tail split and the primary target. The sweep exists to prove the uncertainty signal behaves, not to produce a result per metric, so once it works on one target and one split I have what I need. Writing the ensemble count down is what stops it quietly growing later.

---

**6.2 Which checks survive** `SETTLED`

**Decided:** All three original checks (epistemic shrinks with N, epistemic rises off-distribution, the two signals are not tightly correlated), but run in both input conditions from 2.4.

**Why:** All three were designed as contrasts between two live signals. If aleatoric sits at the numerical floor, "aleatoric does not shrink with N" is trivially true and the correlation is measured against near-noise. Running the hidden-input condition alongside restores the contrast: epistemic decaying toward a nonzero, N-invariant aleatoric floor.

---

**6.3 What to claim** `SETTLED`

**Decided:** Report the shape of the decay. Do not fit an exponent.

**Why:** At ten members with a fixed architecture, the curve is dominated by how fast members stop disagreeing, which is not the clean statistical rate a fitted number would imply.

---

**6.4 GATE: do the signals behave?** `GATE`

- [ ] Passed

**Stop if:** epistemic does not shrink with N in-region, or does not rise out-of-region.

**Then:** something is broken. Nothing downstream means anything until it is fixed.

---

**6.5 When the sweep runs** `SETTLED` ✅ 2026-08-29

- [x] Decided

**Decided: the full N-sweep runs last, after Stage 7 calibration and Stage 8 deferral. The hidden-coefficient check runs early, immediately after the mean-variance ensembles exist, and costs one extra ensemble.**

**The conflict this resolves.** Stage numbering puts the sweep at 6, before calibration at 7 and deferral at 8, which implies sweep first. The end-of-week-2 calendar gate asks only for three splits, calibration figures and a drafted deferral curve, which implies sweep later. The two disagreed and nobody had reconciled them.

**Why last.** Calibration and the deferral curve are the two named headline deliverables (0.3). The sweep validates the uncertainty machinery, which is a supporting result. Under a 2.5 week budget the ordering should be the one where running out of time costs least, and losing a supporting result costs less than losing a deliverable.

**Why last is safe, and this is the part that makes the whole ordering work.** Calibration and the deferral curve do not take the uncertainty signal on faith, they measure it. The deferral curve asks whether deferring the most uncertain shapes actually reduces error, and a useless signal puts the curve flat on the random floor. Calibration asks whether a 90% interval contains the truth 90% of the time. So a broken signal surfaces at Stage 7 or 8, not at the sweep.

**What the sweep uniquely tests** is narrower: whether splitting the uncertainty into ignorance and noise is a real split or two labels. Epistemic must shrink as training data grows, aleatoric must not. A signal can rank points perfectly well, and so produce valid calibration and deferral results, while failing that decomposition test. That is why the sweep can go last without putting the deliverables at risk.

⚠️ **The one risk that does not wait.** The hidden-coefficient condition (6.2) is the only check that can catch a variance head pinned at its floor rather than working. Aleatoric sits at the numerical floor on this dataset by construction (Stage 2), so a variance head that always reports approximately zero looks correct and is untestable.

**The mitigation: one extra ensemble.** The tail split's mean-variance ensemble already provides the all-80 reference. The check adds a single ensemble on the same split with high mode-number coefficients hidden, then compares the two aleatoric estimates. If the hidden version does not report roughly the magnitude that was removed, the variance head is not working. Ten member networks, run as soon as the tail ensemble exists.

**Order of work from here:**

| # | run | ensembles | purpose |
|---|---|---|---|
| 0 | hyperparameter sanity check (4.10) | 0, six single networks | pick lr and batch size on the random split, then freeze into 4.7 |
| 1 | plain MSE ensemble (4.6) | 1 | pipeline check against Table 7 |
| 2 | mean-variance, random split | 1 | in-domain baseline, no shift |
| 3 | mean-variance, interior hole at p30 | 1 | a gap with training data both sides |
| 4 | mean-variance, tail-low | 1 | the extrapolation condition |
| 5 | hidden-coefficient check | 1 | does the variance head work |
| 6 | Stage 7 calibration | 0 | reads runs 2 to 4 |
| 7 | Stage 8 deferral curve | 0 | reads run 4 |
| 8 | full N-sweep | 30, floor 12 | does the decomposition hold as N grows |

**My decision:**
> Put the sweep last, because calibration and the deferral curve measure the uncertainty signal directly rather than assuming it, so a broken signal cannot hide until step 8. The one thing that can hide is a variance head stuck at its floor, and one extra ensemble rules that out before anything is built on it.

---

## Stage 7. Calibration

---

**7.1 Diagnostic set** `OPEN`

- [ ] Decided

**Options:** PIT histogram / coverage versus nominal level / reliability diagram / CRPS / some subset

**Leaning:** pick a small fixed set and use it identically everywhere

**Why:** Consistency across the three splits is what makes the comparison readable. Adding a fourth diagnostic late means regenerating every figure.

**Principle settled, composition still open.** Prefer diagnostics that reuse the distance bins you are already computing over ones that need their own binning. Ranked by data cost:

- **CRPS.** One number per point, averages straight into the existing distance bins at no extra cost. Almost certainly in regardless.
- **Coverage versus nominal.** Needs enough points to estimate a rate per nominal level, the same proportion maths as 3.7. Moderate cost.
- **PIT histogram and reliability diagram.** Both are themselves histograms, so they need their own bins on top of the distance bins. Most data-hungry by a clear margin.

**Why coverage-versus-nominal is next after CRPS, not just next-cheapest:** CRPS is a summary, it says how badly calibrated but not which way. PIT and coverage-vs-nominal say which way, overconfident, underconfident, or biased mean. "Calibration degrades under shift because the ensemble stays overconfident when intervals should widen" is a sharper and more recognisable finding than "the score got worse," and coverage-vs-nominal is also the diagnostic most directly tied to defending the deferral threshold.

**Decide the final set from the day 1-2 histogram**, once you can see what the held-out region actually affords.

**My decision:**
>

---

**7.2 Conditional, not just marginal** `SETTLED`

**Decided:** Report coverage binned by predicted uncertainty, by region, and by distance from the training region (3.11), not just aggregate coverage.

**Why:** Marginal coverage can look fine while hiding severe conditional failure. Exposing exactly that is the point of the project, so aggregate-only reporting would undercut your own result.

**Updated:** distance binning was added later and is now the primary axis of the headline figure, not an extra. Binary in-region versus out-of-region gives one number that can only come out "gap" or "no gap"; a graded curve reports something either way.

---

**7.3 Post-hoc recalibration** `SETTLED` ⚠️

- [x] Decided

**Decided:** Demoted to the bottom of optional scope (now 9.4). If done at all, the calibration set must come from in-region data.

**Why the demotion:** it does not serve either headline deliverable. Its value is narrow but real, as a robustness check: if calibration degrades out-of-region even after a standard in-region correction, that rules out "you just had a globally miscalibrated ensemble and forgot to fix it" as an alternative explanation for your finding. That is a supporting argument, not a third result.

⚠️ **Why the in-region constraint is not just technical:** recalibrating on out-of-region points assumes access to exactly the labels your premise says you do not have. It would produce a good-looking result that means nothing, and the flaw is invisible in the figure.

**My decision:**
> Push it to the bottom of optional scope. It answers a "but did you just forget to calibrate" objection rather than producing a finding of its own, and it comes with a trap that is easy to trip under time pressure.

---

**7.4 Recalibration role** `SETTLED`

- [x] Decided

**Decided:** Reporting only. Never feeds the deferral rule.

**Why:** Keeping recalibration out of the deferral rule keeps the two headline results independent. If a correction fitted for one result also drove the other, a problem in the correction would contaminate both at once.

**My decision:**
> Reporting only. The deferral curve should stand on the raw ensemble output so the two results cannot fail together.

---

**7.5 Which uncertainty signal for calibration** `SETTLED`

**Decided:** Epistemic alone for the calibration-under-shift analysis.

**Why:** Isolating model ignorance from local fitting difficulty is the entire point of the region split. Total uncertainty also carries the variance head's read on how locally hard-to-fit a point is, which for a deterministic solver is closer to "this part of the function is wiggly" than to label noise. Blending them means a degradation could be either "the model does not know it is extrapolating" or "this region is just harder," and telling those apart is the question.

**Contrast with 8.1:** the deferral curve deliberately does not make this choice, because there the job is predicting error, not attributing it.

**My decision:**
> Epistemic only here. The split was built to isolate model ignorance, so the diagnostic should use the signal that measures it rather than one that mixes it with local difficulty.

---

## Stage 8. Deferral curve

---

**8.1 Which signal drives deferral** `SETTLED`

- [x] Decided

**Decided:** Two curves, epistemic-ranked and total-ranked. Not aleatoric-ranked.

**Why:** Which signal defers best is a result, not a setup detail, and showing both is nearly free, same trained ensemble and same predictions, just two sort orders. It also tests the "total and epistemic should be close" claim empirically instead of assuming it, which matters because that claim rests on aleatoric sitting near zero, which is itself a prediction 6.2 is designed to check.

**Dropped from the original "all three":** aleatoric-ranked. If aleatoric really does sit at the numerical floor, ranking by it is ranking by noise, and the curve would land on the random baseline by construction. It answers nothing that the floor measurement in Stage 2 does not already answer.

**My decision:**
> Show epistemic-ranked and total-ranked. Which one defers better is a finding, and running both costs one extra sort. Drop the aleatoric curve, if aleatoric is at the floor as predicted then that curve is just the random baseline with extra steps.

---

**8.2 X-axis** `SETTLED`

**Decided:** Fraction of solver calls saved. Not wall-clock time.

**Why:** The "around one hour per VMEC++ run" figure on page 5 of the paper is the time budget for a full optimization run, not the cost of one forward equilibrium solve. Quoting it as per-call cost at Proxima would get you corrected. Either measure a single high-fidelity call yourself or stay in units of calls.

---

**8.3 Y-axis** `SETTLED`

- [x] Decided

**Decided:** CRPS of the hybrid system, in physical units per 5.2. Deferred points are credited with the exact solver value, so they contribute zero error.

**Why CRPS and not RMSE or MAE:** this is the one figure where the full predictive distribution, not just the mean, is under test. The deferral decision is driven by predicted uncertainty, so scoring its outcome with a metric that ignores uncertainty would be internally inconsistent, RMSE and MAE score a mean-variance ensemble identically to a plain point predictor with the same means. CRPS has a closed form for a Gaussian predictive distribution, so it is cheap.

**Convenient property:** where aleatoric sits at the floor, CRPS collapses toward MAE numerically. So the difference from RMSE/MAE only appears where it should, out-of-region, where epistemic widens the interval.

**Scope:** CRPS is for this figure only. RMSE stays the right tool everywhere else, the Table 7 comparison and general point-accuracy reporting.

**My decision:**
> CRPS here. The whole figure is about whether the uncertainty estimate is useful, and RMSE or MAE would give the same answer whether or not the variance head did anything, which defeats the point. Keep RMSE for the Table 7 comparison.

---

**8.4 Baselines** `SETTLED`

**Decided:** Four lines. Two "mine" curves (epistemic-ranked and total-ranked, per 8.1), random deferral at matched rate (floor), and oracle deferral (ceiling).

**Why:** Your curve alone is unreadable. The gap to random is the actual result; the gap to oracle is how much headroom the signal leaves on the table.

⚠️ **Corrected:** this item previously said three curves with the oracle deferring by true absolute error. Both halves were wrong once 8.3 settled on CRPS. **The oracle must defer by per-point CRPS, not by |error|.** An |error|-ranked oracle is a ceiling for MAE, not for the metric actually plotted, so it would not bound the curve and the "mine" lines could cross it, which reads as a bug and is awkward to explain.

---

**8.5 Evaluation set** `SETTLED`

- [x] Decided

**Decided:** Out-of-region as the headline, specifically the tail split's held-out portion. In-region shown alongside as contrast, not as a competing headline.

**Why:** In-region deferral is easy and uninteresting. The out-of-region curve is the one that speaks to the paper's stated concern about extrapolation errors.

**My decision:**
> Headline is the tail split's held-out set. That is the case the paper actually warned about, and it is where deferral has to earn its keep. In-region goes next to it for contrast so the difference is visible.

---

**8.6 Solver failure rate** `OPEN`

- [ ] Decided (stretch)

**Options:** ignore / fold into the fallback assumption

**Leaning:** only if 1.6 shows failures cluster in your held-out region

**Why:** The deferral curve assumes the fallback always returns the truth. If VMEC++ fails at a meaningful rate in the compact region, that assumption breaks, and saying so with a figure is a specific and correct criticism of your own method rather than a generic caveat.

**My decision:**
>

---

## Stage 9. Optional, in cut order

*Cut from the bottom up. Nothing here is load-bearing.*

---

**9.1 Second target (log10 qi)** `OPEN`

- [ ] Decided at week 2 gate

**Why first to survive:** one line of code once the pipeline works, and it gives you a comparative claim.

**My decision:**
>

---

**9.2 Region ranking as a sampling recommendation** `OPEN`

- [ ] Decided at week 2 gate

**Why:** Turns your uncertainty map into something actionable for a design team. Cheap if the deferral machinery already exists.

**My decision:**
>

---

**9.3 Active learning simulation** `OPEN`

- [ ] Decided at week 2 gate

**Why:** Compare acquisition on random, total, epistemic, aleatoric. This is the most interesting optional item and the most expensive. Needs a full acquisition loop with retraining, which is not a two-hour job.

**My decision:**
>

---

**9.4 Post-hoc recalibration, fit in-region only** `OPEN`

- [ ] Decided at week 2 gate

**Why:** Demoted here from 7.3. Narrow value as a robustness check, not a standalone finding: it tests whether an out-of-region calibration gap survives a standard in-region correction, which closes off one alternative explanation for the headline result. Reporting only, never feeds the deferral rule (7.4).

⚠️ **Trap:** the calibration set must come from in-region data. Calibrating on out-of-region points assumes access to exactly the labels the premise says are unavailable.

**My decision:**
>

---

**~~9.5 Leave-one-field-period-out~~** `DELETED`

**Why deleted:** impossible under 1.2. NFP is fixed at 3, so there is only one field period and nothing to leave out. Recorded here rather than silently removed so it does not get proposed again.

---

## Stage 10. Write-up

---

**10.1 Format** `OPEN`

- [ ] Decided

**Options:** repo with README and reproducible notebook / separate written report / both

**Leaning:** repo plus README

**Why:** For this audience, runnable and readable beats a PDF. A report is optional weight.

**My decision:**
>

---

**10.2 Figure selection** `OPEN`

- [ ] Decided

**Candidates:** interior-hole versus tail calibration contrast (the main figure, 3.8) / distance-binned calibration per split / N-sweep in both input conditions / deferral curve with four lines / failure rate versus aspect ratio / reliability diagrams

**Leaning:** three or four, cut the rest

**Why:** Every extra figure dilutes the two headline results.

**My decision:**
>

---

**10.3 Comparability statement against Appendix A.4** `OPEN`

- [ ] Decided

**Why:** This is the strongest sentence available to you: they trained an ensemble of ten MLPs on this dataset, reported in-domain accuracy per metric, and explicitly named uncertainty calibration as future work. You matched their in-region numbers and then did the thing they named. Draft it deliberately rather than improvising it.

⚠️ **Two limits on how far this can be pushed** (see 4.1 and 5.3). Architecture: it is unverified whether their ensemble was one multi-output network or twelve single-output ones, so claim same-metric test performance, not a reproduced model. Metrics: only RMSE and R² are directly comparable, since their NRMSE and SNR formulas are not published.

**Also worth stating accurately:** Table 7's RMSE is not uniformly small. It spans 0.006 to 0.581, and three metrics exceed 0.1. R² above 0.97 holds for all twelve, lowest 0.974. The two rows relevant here are edge rotational transform at RMSE 0.006 / R² 0.997 and log_10_qi at RMSE 0.051 / R² 0.982.

**My decision:**
>

---

**10.4 Stated limitations** `OPEN`

- [ ] Decided

**Candidates:** survivorship (24k dropped rows) / optimizer-driven sampling density rather than designed coverage / single field period / single dataset / aleatoric term near-degenerate by construction / plasma boundaries only, no coils

**Leaning:** state all of them, briefly

**Why:** Naming your own limitations before anyone asks is worth more than the limitations cost you. The paper does the same in its Section 6.

**My decision:**
>

---

## Calendar gates

---

**GATE: day 1-2 of week 1** `GATE` ✅ PASSED 2026-08-27

- [x] Passed

**Run:** `scripts/day_1_2_grid.py`, twelve single-MLP fits, 746s total. Full results table below.

| axis | cut | target | rmse_in | rmse_out | ratio | d_max | min_bin | δ |
|---|---|---|---|---|---|---|---|---|
| aspect_ratio | tail_low | edge_rot | 0.01379 | 0.04157 | **3.02** | 2.13 | 276 | 0.059 |
| aspect_ratio | tail_high | edge_rot | 0.01521 | 0.01736 | 1.14 | 1.35 | 196 | 0.070 |
| aspect_ratio | hole | edge_rot | 0.01476 | 0.01406 | 0.95 | 0.20 | 641 | 0.039 |
| max_elongation | tail_low | edge_rot | 0.01431 | 0.02727 | 1.91 | 2.28 | 12 | 0.283 |
| max_elongation | tail_high | edge_rot | 0.01443 | 0.01824 | 1.26 | 37.23 | **0** | ∞ |
| max_elongation | hole | edge_rot | 0.01528 | 0.01391 | 0.91 | 0.26 | 608 | 0.040 |
| aspect_ratio | tail_low | log10_qi | 0.14344 | 0.21976 | 1.53 | 2.13 | 276 | 0.059 |
| aspect_ratio | tail_high | log10_qi | 0.14283 | 0.24935 | 1.75 | 1.35 | 197 | 0.070 |
| aspect_ratio | hole | log10_qi | 0.14842 | 0.16515 | 1.11 | 0.20 | 641 | 0.039 |
| max_elongation | tail_low | log10_qi | 0.13291 | 0.23208 | 1.75 | 2.28 | 12 | 0.283 |
| max_elongation | tail_high | log10_qi | 0.14926 | 0.17981 | 1.20 | 37.25 | 0 | ∞ |
| max_elongation | hole | log10_qi | 0.14604 | 0.14420 | 0.99 | 0.26 | 614 | 0.040 |

**Both pass conditions met.** Aspect ratio with tail-low gives a 3.02 ratio on the primary target, and aspect ratio clears coverage in the mid-range band (641 points, δ 0.039) and the tail band (276 points, δ 0.059).

**Decisions this settles:** 3.1 primary target (edge rotational transform), 3.2 secondary evidence (log10 qi viable, defers to week 2), 3.4 split axis (aspect ratio), 3.6 direction (low), 3.7 δ (0.06). 7.1 remains open, see that entry.

⚠️ **RETRACTED 2026-08-29, see 3.7 and 3.8.** This block previously read: "the most important result is the one that looks like nothing," on the grounds that interior-hole ratios of 0.95 and 0.91 meant the model fills a mid-range gap with no measurable penalty while the same recipe cost 3x at the tail, giving a clean "can interpolate into gaps, cannot leave the region" story.

**Two later measurements killed it.** The placement sweep showed 0.95 held for a hole at the median only; moving the hole toward the sparse compact end raises the penalty to 1.80 and then 1.96, so the absence of a gap penalty was a property of the placement, not of interpolation. The distance-error curves then showed that most of the remaining hole-versus-tail difference is distance rather than edge: at matched distance the tail costs about 15% more, not 200%.

**What survives:** the tail-low ratio of 3.02 and the fact that the compact end degrades badly. What does not: the binary reading of interpolation as free. The kept claim is now narrower and is stated in 3.8.

**Methodological note worth defending.** The grid uses an MLP rather than gradient boosting on purpose. Trees return the boundary value outside the training range, so a tail split would show a large gap by construction and would measure the model class rather than the axis. Given that the eventual model is an MLP ensemble, the MLP baseline also previews the real extrapolation behaviour. Choosing the faster model here would have produced a confidently wrong answer to the gate's central question.

**In-region sanity anchor holds.** RMSE sits about 2.4x above Table 7 on edge rotational transform (0.0138 vs 0.006) and 2.8x on log10 qi (0.143 vs 0.051). One untuned MLP against a tuned ten-member ensemble is expected to land there, and a consistent factor across two unrelated metrics is positive evidence the loader, filters, trimming and scaling are all correct, since a pipeline bug would not produce that.

**Required:** cheap baseline (ridge or a single small MLP, not the ensemble) run across the full candidate grid before any ensemble work begins. Axes (aspect ratio, max elongation) × cuts (tail-low, tail-high, interior hole) × targets (edge rotational transform, log10(qi)). Twelve fits, one afternoon.

**Why this exists:** it front-loads the single risk the whole project depends on. Without it you find out whether the split axis produces a real generalization gap only after building the full ensemble, at the week 2 gate, with no time left to react. The cheap version answers the same question in an afternoon using data you have to load anyway.

**What it decides:** the split axis and direction (3.4, 3.6), real evidence for the primary and secondary target choices (3.1, 3.2), the coverage tolerance δ (3.7), and the calibration diagnostic set (7.1).

**Pass condition:** some (axis, direction) pair shows a real in/out error gap for the primary target AND that axis clears the coverage threshold in both a mid-range band and a tail band, since the main figure needs the hole and tail results both.

**If nothing shows a gap:** reconsider PCA direction or high mode-number spectral energy before committing further hours. Do not proceed to the ensemble on an axis that showed no gap.

---

**GATE: end of week 1** `GATE`

- [ ] Passed

**Required:** data loaded, noise floor checked, one model trained, in-region numbers in the neighbourhood of Table 7.

**If not met:** fall back to a simpler dataset (UCI or semi-synthetic) and keep the same study design. Without regret. The design is the contribution; the dataset is the setting.

---

**GATE: end of week 2** `GATE`

- [ ] Passed

**Required:** three splits run, calibration figures exist, deferral curve drafted.

**If not met:** drop the second target and everything in Stage 9. Spend the remaining time finishing what exists.

---

## Notes and surprises

*Anything that contradicted an assumption. Interview material lives here.*

| Date | What I expected | What I found |
|---|---|---|
| 2026-08-26 | R(0,0) is fixed at exactly 1, so dropping it is trivially safe | It is not fixed. Range 0.894 to 1.016, std 0.0051, only 52.8% within 1e-4 of 1.0. Still correct to drop: 6 to 20x less variable than any genuine coefficient, weak correlation with aspect ratio (0.14), and their `_x_to_surface` hardcodes it to 1.0 on reconstruction. Right answer, wrong reason. |
| 2026-08-26 | `has_neurips_2025_forward_model_error` is the flag that identifies bad rows | It misses generation-stage failures. One row has a fully null boundary with that flag reading `False`, because the solver never ran on it. Filtering on the documented flag alone passes it straight through into the flattener. It crashed the first full-dataset script written for this project. |
| 2026-08-26 | The 80-coefficient encoding would need to be reverse-engineered from the data | Their `_to_X` states it exactly: ravel each `(5,9)` array, drop the first 5 entries. Empirically confirmed across all 182,221 rows, the symmetry zeros are exactly 0.0, not approximately. |
| 2026-08-26 | Matching their documented filters would reproduce A.4's ~23k pool | Gives 27,050, about 15% high. Duplicates, the five-flag filter and the 0.05% trim are all ruled out by direct test. Two intermediate counts land exactly on published figures (158,685 and 68,191), so the chain is right and their appendix is incomplete. |
| 2026-08-26 | The input representation is unique per shape | It is not. Negating every `z_sin` coefficient gives the mirror image, and the stored pool is 89.4% / 10.6% mixed on that sign. Their generative code canonicalizes it, their surrogate feature extractor does not. Left uncanonicalized to match `_to_X`, and because the two handedness classes carry different information (see 1.7). |
| 2026-08-27 | Mirroring flips the sign of the primary target | It does not, in this data. Both handedness classes are overwhelmingly positive (24,041/24,169 and 2,848/2,853), and only 133 of 27,022 targets are negative at all. Their forward model stores VMEC's signed edge iota with no `abs`, all three benchmark problems `abs` it at scoring time, and their own ALM optimizer does not. The generation optimizer selected for positive signed iota, which is why negatives are rare. The 1.7 conclusion survives; its justification was rewritten. |
| 2026-08-26 | The `*_settings.id` columns identify configurations | They identify settings *groups*. 17,671 desc rows share 148 unique ids, 9,379 vmec rows share 8. No per-row config id exists, so pathway must be inferred from which of four blocks is non-null, and dedup by id is impossible. |
| 2026-08-27 | `boundary.r_cos` holds a clean `(5, 9)` numeric block per row | It holds an object array of 5 entries, each its own 9-element float array. `np.stack` merges only the outer level, giving `(n, 5)` object dtype, and the `[:, 5:]` slice then returns an empty `(n, 0)` array **without raising**. Silently produced a zero-width feature matrix. Fix is a per-row `.tolist()` before `np.array`. The regression test for it has to reproduce the nested structure, or a clean-block fixture passes against the broken code. |
| 2026-08-27 | Gate 3.5 would predict aspect ratio near machine precision | Ridge 0.784, gradient boosting 0.988, MLP 0.983. The linear result reflects model class, aspect ratio being roughly R(0,0) over minor radius, and a linear model cannot represent division. The residual gap is budget, not information: every increase in training budget raised the score, the trim changed nothing, and residual correlation with the dropped R(0,0) was +0.075. |
| 2026-08-27 | A pre-registered pass threshold protects against fooling yourself | Only if it measures the right thing. 0.99 was derived from the information floor implied by dropping R(0,0), which bounds what *any* model could know and says nothing about what a quick untuned fit reaches on 21k points. Knowable and learnable are different questions. The gate's question was also qualitative, input-measurable versus solver output, so a single absolute R² was the wrong instrument regardless of its value. Recorded the 0.988 miss rather than relaxing the bar. |
| 2026-08-27 | The interior hole would show a smaller gap than the tail, but still a gap | It shows no gap whatsoever. Ratios of 0.95 and 0.91, meaning out-of-region error is *lower* than in-region. The model fills a mid-range gap for free and pays 3x to leave the region. Better than expected for the project: the contrast is absolute rather than a matter of degree, and it separates "cannot interpolate" from "cannot extrapolate" more cleanly than a graded difference would have. |
| 2026-08-27 | The runner-up split axis would lose on gap size | max_elongation was rejected on coverage instead. It shows real gaps (1.91 and 1.26) but its tail-high held-out set reaches 37 standard deviations from the training region against aspect ratio's 2.13, so an equal-width distance bin comes out empty and no coverage rate can be estimated at any tolerance. The two-condition pass criterion in 3.7 caught something gap size alone would have missed. |
| 2026-08-27 | Model choice for a cheap baseline is a speed convenience | It would have inverted the gate's answer. Gradient boosting returns the boundary value outside the training range, so every tail split shows a large gap by construction regardless of whether the axis means anything. Picking the faster model would have produced a confident, wrong pass on any axis tested. |
| 2026-08-27 | Binning is a presentation choice that does not change conclusions | It flipped one. With uniform deciles the log10 qi noise floor read 0.021, twice its bar, verdict ABOVE FLOOR. The bottom decile spanned distance 0 to 1.53 while genuine near-twins live below 0.5, so the intercept was extrapolated from bins centred at 1.1 to 2.4 and never came near zero. Rebinning with quantile edges packed into the left tail dropped it 10x to 0.0018 and flipped the verdict to AT FLOOR. The tell was that the fit window's own x-values sat nowhere near the point being extrapolated to. |

---

## Open questions to raise later

*Things you noticed but deliberately did not chase.*

-
-
-
