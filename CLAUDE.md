# CLAUDE.md

Context for an assistant helping with this project. Save at the repo root so it loads automatically, or paste as the first message.

---

## How to work with me

**Guide me. Do not do the work for me.** I am writing the code myself. This is a portfolio project whose value depends on my being able to defend every line of it in an interview, so code I did not write is worse than no code.

Concretely:

- When I ask how to do something, explain the approach and the tradeoffs. Do not produce the implementation unless I ask for it directly.
- Small snippets to illustrate a point are fine. Whole modules, full functions, or "here is the script" are not.
- Reviewing code I have written is welcome and useful. Writing it for me is not.
- If I ask a question that has a wrong premise, say so before answering.
- Push back when my reasoning is wrong. I would rather be corrected than agreed with.
- Be concise. No preamble, no summarizing what I just said back to me.
- Never, ever use em dashes. Not in prose, not in code comments, not in commit messages. No exceptions.

If you are uncertain whether I want explanation or implementation, ask.

---

## Who I am

Data scientist. MSc in Data Analytics. Six years before that as a process engineer in polymer additive manufacturing (SLA/DLP). MSc thesis was on uncertainty quantification for time series regression.

Comfortable: standard ML tooling, PyTorch, the UQ literature, ensembles, calibration diagnostics.
Rusty: Gaussian processes and Bayesian inference. I have seen them in coursework but cannot use them fluently.
Zero background: fusion physics, stellarator design, plasma physics of any kind.

Explain domain concepts without assuming prior knowledge. Do not assume I know what a flux surface, Boozer coordinate, or magnetic well is unless I have used the term first.

GPU access available.

---

## The project

An uncertainty-aware surrogate model for Proxima Fusion's ConStellaration dataset, plus a study of where its predictions can actually be trusted.

**Setup.** A stellarator is a twisted toroidal magnetic confinement device. What gets designed is the shape of the plasma boundary surface, encoded as 80 Fourier coefficients. That shape goes into VMEC++, an expensive ideal-MHD equilibrium solver, which produces a magnetic field, from which performance metrics are computed. The surrogate predicts one of those metrics directly from the shape, skipping the solver.

**Model.** Ensemble of mean-variance networks. Inputs are the 80 boundary coefficients. Output is one physics metric, predicted as a mean and a variance.

**Uncertainty decomposition.** Epistemic is the spread of member means. Aleatoric is the mean of member variances. Total is their sum.

**The core design choice.** Three splits, same model and recipe
for each. Random (a plain train/test split, the easy baseline,
in-domain only). Interior hole (a contiguous gap carved from
the middle of the split axis, surrounded by training data on
both sides, tests whether the model can fill a gap it hasn't
seen). Tail (the extreme end of the split axis held out
entirely, the actual extrapolation condition, tests whether the
model can leave the region at all). The gap between the
interior-hole and tail results is the main figure, it separates
"the model can't fill gaps" from "the model can't leave the
region."

**Two headline deliverables.**
1. Calibration measured across all three splits, in-region vs
   out-of-region within each, binned by distance from the
   training region for interior hole and tail. The interior-
   hole-versus-tail contrast is the main finding. Generalization
   behavior under each split is reported regardless of whether
   a gap appears.
2. A deferral curve. Use the surrogate below an uncertainty threshold, fall back to VMEC++ above it. Sweep the threshold, plot solver calls saved against error incurred.

**Why this angle.** The ConStellaration paper's Appendix A.4 trains an ensemble of MLPs on this dataset, reports in-domain accuracy, and then explicitly names uncertainty calibration and active learning as open directions. Nobody has published UQ work on this dataset. The generative modelling lane is crowded; this one is not.

**Audience.** Portfolio project that may double as an interview showcase for Proxima Fusion. Target is an internship on their uncertainty-aware surrogate modelling work.

---

## Constraints

- Roughly 2.5 weeks, 10 to 20 hours per week.
- **A finished narrow project beats an unfinished ambitious one.** When suggesting anything, weigh it against this. Suggesting scope is usually the wrong move.
- No Gaussian processes and no Laplace approximations in the core. They go in future work.

---

## Environment and working state

Work happens in a Lightning AI Studio, reached over SSH from local VS Code (Remote-SSH host alias `lightning-constellaration`). Workspace root is `/teamspace/studios/this_studio/ConStelleration`, which is the persistent path and survives machine-type switches.

⚠️ **Use the right interpreter.** Bare `python3` resolves to `/usr/bin/python3`, which has **none** of the project's packages. The active environment is the conda env `cloudspace`:

```
/home/zeus/miniconda3/envs/cloudspace/bin/python3
```

An interactive VS Code terminal activates it automatically, but a non-interactive `ssh host "command"` does not, so scripted calls must use the full path. Running bare `python3` and seeing every import fail is this trap, not a broken environment.

**Installed:** torch 2.8.0+cu128, pandas 2.1.4, scikit-learn 1.3.2, numpy 1.26.4, matplotlib 3.8.2 (preinstalled by Lightning), plus pyarrow 25.0.1, datasets 5.0.1, huggingface_hub 1.28.0, pytest 8.3.4 (added for this project). Pinned in `requirements.txt`. **Do not reinstall or upgrade torch**, the CUDA build is matched to Lightning's GPU images.

**Package:** installed editable, so `import constellaration_uq` works from anywhere. Source lives in `src/constellaration_uq/`.

**Data:** `data_raw/data/train-0000{0,1,2}-of-00003.parquet`, ~584 MB, the `default` subset only. Gitignored. Re-fetch with `huggingface_hub.snapshot_download(repo_id="proxima-fusion/constellaration", repo_type="dataset", allow_patterns=["data/*"], local_dir="data_raw")`.

**Machine type:** currently CPU. `torch.cuda.is_available()` returning `False` is correct here, not a fault. Everything through the day 1-2 grid check is CPU work; switch to GPU only for the Stage 6 N-sweep, and switch back after. An attached local VS Code session prevents the studio from auto-sleeping, which is harmless on CPU and expensive on GPU.

**Current position:** Phases 0, 0b and A complete (studio, repo, environment). Phase B complete (data pulled, schema verified). Data findings documented in decision log 1.5 to 1.8.

`src/constellaration_uq/data.py` is written and verified against the real files. Five functions: `load_raw`, `filter_valid` (a generator yielding `(step_name, df)` after each step, so the 1.8 table prints straight out of the code path that produces the training data), `trim_target_tails` (step 5, target-dependent, kept separate so the day 1-2 grid can trim per candidate target), `extract_input_features` (the 1.5 flatten to 80 columns), and `load_dataset` wiring the first three together. Run against the real parquet files, the printed counts land exactly on 1.8: 182,222 → 158,685 → 68,191 → 27,050 → 27,050, then 27,022 after the target trim. `X` comes out `(27050, 80)` float64, no NaNs.

⚠️ **The boundary columns are nested, not `(5, 9)` blocks.** Parquet returns, per row, an object array of 5 entries, each its own 9-element float array. `np.stack` only merges the outer level, giving `(n, 5)` object dtype, and the `[:, 5:]` slice then silently returns an empty `(n, 0)` array instead of raising. The working version converts per row first: `np.array([row.tolist() for row in df[col]])`. Cost is 0.14s over 27k rows.

`tests/test_data.py` covers the three functions that do real logic (16 tests, pytest). Fixtures deliberately reproduce the nested boundary structure, verified that the pre-fix implementation fails them.

**Gate 3.5 returned INCONCLUSIVE (2026-08-27), `scripts/gate_3_5_split_axis.py`, and we proceeded on the evidence below.** ⚠️ This read "passed with a caveat" until 2026-08-31, which upgraded the artifact's own recorded verdict. The bar was not moved and the miss was always disclosed; only the verb was softened. Aspect ratio predicted from the 80 coefficients: ridge 0.784, gradient boosting 0.988, MLP 0.983, against a std of 1.639. **Aspect ratio is input-measurable, so it is safe as the split axis** and splitting on it is a shift in the questions, not in the answers.

It missed the pre-registered 0.99 bar. That bar came from the information floor implied by dropping R(0,0) per 1.5, which bounds what any model could know and says nothing about what a quick untuned fit reaches on 21k points. Everything else points one way: the 0.05% trim moved R² by 0.00002, residual correlation with R(0,0) was +0.075, and every increase in training budget raised the score, so the shortfall is budget rather than missing information. The miss is recorded rather than the bar relaxed, see decision log 3.5.

Incidental, worth keeping: error tracks data density, lowest around A 9.9 to 10.1 and highest in the sparse upper tail. The extrapolation premise shows up in a purely geometric quantity before any surrogate exists.

**Stage 2 gate passed (2026-08-27), `scripts/stage_2_noise_floor.py`.** One kNN index (k=10, brute force, z-scored coefficients) answered the residual-spread floor (2.3) and the tolerance duplicate check (2.2). **Aleatoric sits at the numerical floor for both candidate targets**, confirming the 2.1 prediction. Floor intercepts −0.000003 (edge rotational transform, bar 0.0012) and 0.001834 (log10 qi, bar 0.0102). The 28 pairs closer than distance 0.197 have a median target difference of **exactly 0.000000 for both targets**, which is determinism measured rather than asserted.

⚠️ **Bin placement flipped a verdict.** With uniform deciles, log10 qi read ABOVE FLOOR at 0.021. The bottom decile spanned distance 0 to 1.53 while genuine near-twins live below 0.5, so the intercept was extrapolated from bins centred at 1.1 to 2.4. Rebinning with quantile edges packed into the left tail dropped it 10x and flipped the verdict. Recorded in decision log 2.3, not hidden.

⚠️ **High-dimensional caveat for the write-up:** median nearest-neighbour distance is 3.84 in z-scored 80-dimensional space. Genuine near-twins barely exist outside the bottom percentile, which caps what this check can prove.

**`splits.py` written and tested (2026-08-27).** Four functions: `random_split`, `tail_split`, `hole_split` (all returning complementary boolean masks so the grid calls them interchangeably) and `distance_from_training_region`, the one 3.11 formula covering all three splits. Distance uses sort plus `searchsorted`, not all-pairs, which would need 4.7 GB at this pool size; `np.clip` guards the ends where `pos - 1` would otherwise wrap and silently return a wrong distance. 20 tests in `tests/test_splits.py`, 36 across the suite.

**Day 1-2 gate PASSED (2026-08-27), `scripts/day_1_2_grid.py`,** twelve single-MLP fits in 746s. **Decided: split axis is aspect ratio, direction is low (hold out the compact end), primary target is edge rotational transform, δ is 0.06.** log10 qi is viable as a secondary and still defers to the week 2 gate.

Headline numbers, out/in RMSE ratio on the primary target: **aspect ratio tail-low 3.02**, tail-high 1.14, interior hole 0.95.

⚠️ **The "hole is free" reading did not survive, see the placement sweep below.** Ratios of 0.95 and 0.91 held for a hole at the median only, and the clean "interpolates for free, cannot extrapolate" story went with them.

⚠️ **The grid must use an MLP, never gradient boosting.** Trees return the boundary value outside the training range, so a tail split shows a large gap by construction and measures the model class rather than the axis. The faster model would have produced a confident wrong pass on any axis.

⚠️ **max_elongation was disqualified on coverage, not on gap.** It shows real gaps (1.91, 1.26) but its tail-high held-out set reaches 37 std from the training region against aspect ratio's 2.13, so an equal-width distance bin comes out empty and no coverage rate is estimable. The two-condition rule in 3.7 caught what gap size alone would have missed.

**Hole placement settled (2026-08-29), `scripts/hole_placement.py`. Centre p30, spanning p20 to p40.** It sits immediately above the tail's p0 to p20 without overlapping, so the two experiments share no held-out configuration and their sizes stay matched at 5,404 and 5,405 (`tail_split` includes the boundary row). Reach 0.45 against the median placement's 0.20, thinnest bin 583, δ = 0.041. That is the widest reach a non-overlapping hole can have, so 0.45 caps the matched-distance claim. δ = 0.06 is unchanged, it was always set by the tail's 276-point bin. Full reasoning in decision log 3.7, held-out fraction and hole placement.

⚠️ **The penalty is a U in placement, not a monotone climb.** At 20% held out the ratio runs 1.96 / 1.80 / 1.12 / 0.95 / 0.90 / 0.80 / 0.82 for centres p20 through p80. A 10% sweep, the only one that reaches p90, closes the U at 0.91. Ratio tracks reach almost monotonically, so **gap width drives most of the penalty** and sparsity matters because it makes gaps wide. Two caveats: a 20% band cannot be centred above p80 without running off the data, and the U's upper arm rests on the two thinnest bins in either table (about 130 points against roughly 310) at a single seed.

⚠️ **Ratios below 1.0 are expected.** The denominator is a random in-region slice drawn from the whole training region including its thin compact end, while a dense-middle hole contains none of those hard cases. 0.80 means that gap was easier than the model's average job, not that removing data helped.

**Distance-error curves measured (2026-08-29), `scripts/distance_error.py`,** three fits in 165s. Deliverable one in single-model form: per-point error binned by distance from the training region, three splits, one MLP each.

**At matched distance the tail costs about 16% more than the hole**, 1.156 ± 0.052 over three seeds (1.119, 1.119, 1.229), so roughly 3 standard deviations from no effect and the weakest of the headline claims. So most of the 3.02-versus-1.80 headline is **distance, not edge**: the tail simply asks about configurations further from the data. The edge cost is what survives controlling for that, and it is real but modest. Weaker than the old claim, and far more defensible.

⚠️ **Corrected 2026-08-31, and the correction is now computed rather than eyeballed.** This read "about 15% at both 0.2 and 0.4 std out", derived by interpolating two binned curves at their bin medians. The bins are equal-count, so their widths differ where the splits differ in density: the tail's second bin spans 0.13 to 0.31 while the hole's are about 0.05 wide there, and RMSE inside a wide bin is dominated by its far edge. `distance_error.matched_windows` now compares the two inside identical fixed-width windows and writes `results/distance_error_matched.csv`.

| window (std out) | n hole | n tail | RMSE hole | RMSE tail | tail / hole |
|---|---|---|---|---|---|
| 0.0 to 0.1 | 1,344 | 532 | 0.01881 | 0.02063 | 1.10 |
| 0.1 to 0.2 | 1,272 | 406 | 0.02230 | 0.02299 | 1.03 |
| 0.2 to 0.3 | 1,066 | 393 | 0.02806 | 0.03169 | 1.13 |
| 0.3 to 0.4 | 1,129 | 348 | 0.03121 | 0.03275 | 1.05 |
| 0.4 to 0.5 | 593 | 277 | 0.03305 | 0.04457 | 1.35 |
| **whole shared range** | **5,403** | **1,826** | **0.02615** | **0.02929** | **1.12** |

⚠️ **Do not claim the cost grows with distance.** At seed 0 the first four windows sit between 1.03 and 1.13 with no trend, and the 1.35 comes from the thinnest window, which also has the worst-matched medians (0.426 against 0.448) in the range where error rises fastest. **Quote the 12% over the shared range and stop there.** A trend was predicted before the run and the data does not support it, which is the same failure that produced the original 15%.

**The number for a pitch:** at 1.83 std out, RMSE is 73% of the target's own std (0.07892). Predicting the dataset mean would score 100%. The surrogate is barely beating nothing there while still returning a confident-looking number.

⚠️ **Two measurement choices that would otherwise flatter the result.** Distance is computed against the fit set, not `train_mask`; the mask contains the in-region slice, which would then get distance 0 by construction rather than by measurement. And the in-region slice is one anchor at distance 0, not part of the binned curve, since it is half the evaluation set and quantile binning spent three of eight bins stacking it on the y axis.

⚠️ **Per-experiment summaries do not plot.** Three attempts to draw the placement sweep were all misread, including by their author, because nothing in a scatter of eight points tells a reader that each point is its own model fit rather than a sample from one curve. The sweep is now a table. The figure plots configurations, roughly 675 per bin, which is what made it readable.

**In-region sanity anchor holds:** RMSE about 2.4x Table 7 on edge rotational transform and 2.8x on log10 qi. One untuned MLP against a tuned ten-member ensemble should land there, and a consistent factor across two unrelated metrics is positive evidence the pipeline is correct.

**`src/constellaration_uq/baseline.py` holds the shared single-MLP recipe** (`load_pool`, `fit_mlp`, `rmse`, and the frozen architecture constants). Three scripts fit the same network on different splits, and "same model and recipe for every split" is the premise that makes their numbers comparable at all, so three copy-pasted definitions is where that premise quietly stops being true. The hole placement sweep reproduces its previous ratios exactly after the extraction, which is the check that it changed nothing.

**PyTorch training layer built and tested (2026-08-29).** `src/constellaration_uq/nets.py` holds the network, the training loop, `split_validation` and the frozen recipe constants. `src/constellaration_uq/ensemble.py` holds the ten-member wrapper. 25 tests across `tests/test_nets.py` and `tests/test_ensemble.py`, 68 in the suite. The one that matters most is `test_different_seeds_give_different_members`: if the per-member seed does not reach the initialisation, every member is identical, the spread is exactly zero, and the epistemic term reads as perfect confidence everywhere with nothing raising.

⚠️ **`predict` un-z-scores by multiplying by `y_std`, which the mean-variance version cannot reuse**, because a variance rescales by `y_std` squared. That is why it gets its own function rather than a flag, and it is the most likely place for a silent unit error in the whole codebase.

**Step 0 done (2026-08-29), `scripts/hp_check.py`, 436s.** Six single-network fits, three learning rates by two batch sizes, on the random split with selection on validation loss only. Recipe frozen, see decision log 4.7. Batch 512 lost outright at every learning rate.

**Step 1 done (2026-08-29), `scripts/mse_ensemble.py`, ten members in 454s. Ensemble RMSE 0.01052 against a pre-registered bar of 0.0105. `results/mse_ensemble.json` records `verdict: FAIL`, by 0.2%, and we proceeded on the evidence below.** The bar was badly calibrated, derived as a round number with no uncertainty, and the miss is recorded rather than the bar moved and rather than the seed rerolled. Proceeding on the evidence: 1.75x Table 7 against their tuned ensemble, the sklearn single MLP was 2.3x, and the ensemble beats its best member by 14% so it is not carried by one lucky run.

⚠️ **Do not report R² against Table 7.** Ours is 0.982 against their 0.997, which reads far worse than the RMSE ratio implies, because R² depends on the test set's own spread: 0.0786 for us against roughly 0.115 back-solved from their table. RMSE is the comparable number, R² is not.

**In-region epistemic spread is 0.00631**, 8% of the target std and about 60% of this ensemble's total error.

⚠️ **Do not compare 0.00631 to any later epistemic number.** `mse_ensemble.py` still aggregates as `mean(sqrt(variance))`, the convention retired everywhere else on 2026-08-30, and this run also used a different loss (MSE, no variance head) and 21,118 training rows against the mean-variance runs' 16,794. Three changes at once. The hole and tail growth comparisons use step 2's own in-region 0.00887 as their baseline, so nothing downstream depends on this figure.

**Step 2 done (2026-08-29, renumbered 2026-08-30), `scripts/mv_ensemble.py random`, ten members.** In-region RMSE 0.01256, epistemic 0.00887, aleatoric 0.01194, total 0.01487. Out-of-region is near-identical, which is correct for a random split.

⚠️ **All uncertainty numbers on this page were re-aggregated on 2026-08-30 and every one of them rose.** See "the aggregation fix" below. RMSE never moved and the per-point predictions are byte-identical, so nothing here required retraining a model, only re-summarising one.

**The uncertainty is slightly conservative in-region:** total 0.01487 against an actual RMSE of 0.01256, 18% over-dispersed, confirmed independently by coverage at nominal 0.9 coming out 0.966. Nothing pinned on the variance floor, 0.00%, so decision log 4.4's β-NLL trigger did not fire.

⚠️ **Aleatoric came out at 0.01194, not near zero.** See the corrected commitment above. Short version: Stage 2 measured that the world has no label noise, while the variance head measures scatter *this model* cannot predict, which includes its own misfit. Both are right. Do not quote the aleatoric number as the dataset's noise level.

⚠️ **The "+19.4% cost of the variance head" the script prints is confounded.** Step 1 trained on 21,118 rows; step 2 carves an in-region slice first and trains on 16,794. Part of that gap is 20% less data. Treat it as an upper bound.

**Step 3 done (2026-08-29, re-aggregated 2026-08-30), `scripts/variance_check.py`, three ensembles in 1101s: RECOVERS.** Gaussian noise of known σ added to the training targets, evaluated against clean ones.

| σ | expected aleatoric | measured | ratio | RMSE | epistemic |
|---|---|---|---|---|---|
| 0 | 0.01194 | 0.01194 | 1.00 | 0.01256 | 0.00887 |
| 0.005 | 0.01294 | 0.01356 | 1.05 | 0.01301 | 0.00958 |
| 0.020 | 0.02329 | 0.02490 | 1.07 | 0.01568 | 0.01360 |
| 0.050 | 0.05141 | 0.04838 | 0.94 | 0.02164 | 0.02444 |

**All ratios inside 0.94 to 1.07** against a 0.75 to 1.35 band, across a tenfold range, monotone in σ. The verdict was unchanged by the aggregation fix (previously 1.07 / 1.09 / 0.94), because expected and measured both shifted, which is itself mild evidence the check is robust to how it is summarised. **This is what converts the epistemic/aleatoric split from an assertion into a measurement**, and it is the reason the mean-variance head is not decorative. It also closes β-NLL: nothing pinned, nothing destabilised, in any of the forty member networks trained so far.

**Epistemic rises with σ too, and that is correct.** Every member saw the same noisy targets, so this is not disagreement about noise draws. Noisy labels underdetermine the fit, so different initialisations reach genuinely different functions.

⚠️ **One reading that looks like failure and is not.** At σ = 0.05 total predicted uncertainty is 0.053 against a clean-target error of 0.0216, which reads as badly over-dispersed. The model estimates uncertainty for the noisy distribution it trained on, while being scored against clean targets. Against noisy targets the expected error is sqrt(0.0216² + 0.05²) = 0.0545 versus 0.053 predicted. Correctly calibrated for its own distribution.

**Steps 4 and 5 done (2026-08-29), `scripts/mv_ensemble.py hole` and `tail`.** All three headline ensembles now exist, and this is the result the project was built to produce.

⚠️ **REPLICATED ACROSS THREE SEEDS, 2026-08-31.** The table below is seed 0, which every figure and downstream script still uses. `scripts/seed_spread.py` re-measures every quantity over seeds 0, 1 and 2 and writes `results/seed_spread.json`. **Seed noise runs 0.2% to 6.7%, mostly 2 to 5%, and every claim in this section clears it by roughly an order of magnitude.** Means with spreads:

| split | region | RMSE | epistemic | aleatoric | total | ratio | cov @ 0.9 | PIT mean |
|---|---|---|---|---|---|---|---|---|
| random | in | 0.01286 ± 0.00027 | 0.00895 ± 0.00018 | 0.01160 ± 0.00057 | 0.01465 ± 0.00053 | 1.140 ± 0.039 | 0.961 ± 0.003 | 0.514 ± 0.008 |
| random | out | 0.01325 ± 0.00033 | 0.00898 ± 0.00018 | 0.01156 ± 0.00060 | 0.01464 ± 0.00059 | 1.105 ± 0.047 | 0.959 ± 0.003 | 0.509 ± 0.008 |
| hole p30 | in | 0.01278 ± 0.00063 | 0.00841 ± 0.00020 | 0.01082 ± 0.00054 | 0.01371 ± 0.00055 | 1.075 ± 0.072 | 0.958 ± 0.007 | 0.496 ± 0.007 |
| hole p30 | out | 0.02423 ± 0.00046 | 0.01186 ± 0.00018 | 0.01293 ± 0.00059 | 0.01755 ± 0.00039 | **0.725 ± 0.028** | 0.878 ± 0.005 | 0.534 ± 0.002 |
| tail-low | in | 0.01219 ± 0.00006 | 0.00853 ± 0.00027 | 0.01085 ± 0.00038 | 0.01381 ± 0.00040 | 1.132 ± 0.034 | 0.959 ± 0.002 | 0.507 ± 0.002 |
| tail-low | out | 0.04048 ± 0.00135 | 0.01505 ± 0.00047 | 0.02290 ± 0.00073 | 0.02740 ± 0.00079 | **0.677 ± 0.021** | 0.788 ± 0.017 | **0.353 ± 0.012** |

**Out-over-in RMSE: random 1.03 ± 0.01, hole 1.90 ± 0.12, tail 3.32 ± 0.10.** The three ranges do not come close to overlapping.

**The random split is the control and it behaves.** Its gap is 1.03 ± 0.01 and its in and out ratios are statistically identical. A diagnostic that manufactured a gap there would invalidate everything downstream, and across three seeds it does not.

⚠️ **Seed 0 was the most conservative seed on the tail**, gap 3.24 against a mean of 3.32, so the published headline understates the effect slightly rather than flattering it.

⚠️ **What the seed varies, and what it does not.** Member initialisation, shuffle order, the in-region slice and the validation set all move. **The hole and tail held-out sets do not**, since they are quantile cutoffs on the axis with no random component, so the out-of-region numbers are replicated on a fixed test set. Only the random split's test set moves.

⚠️ **Member seeds are spaced by N_MEMBERS.** `train_mv_ensemble` uses `seed = base_seed + k`, so consecutive replication seeds would otherwise share nine of ten member initialisations and this would have measured almost nothing. `base_seed = seed * N_MEMBERS` keeps them disjoint, and seed 0 is unchanged, which is how the existing results stayed byte-identical.

| split | region | RMSE | epistemic | aleatoric | total | total / RMSE | cov @ 0.9 | PIT mean |
|---|---|---|---|---|---|---|---|---|
| random | in | 0.01256 | 0.00887 | 0.01194 | 0.01487 | **1.18** | 0.966 | 0.518 |
| random | out | 0.01280 | 0.00895 | 0.01161 | 0.01466 | **1.15** | 0.962 | 0.514 |
| hole p30 | in | 0.01252 | 0.00813 | 0.01010 | 0.01297 | **1.04** | 0.950 | 0.506 |
| hole p30 | out | 0.02484 | 0.01197 | 0.01209 | 0.01702 | **0.69** | 0.871 | 0.531 |
| tail-low | in | 0.01223 | 0.00834 | 0.01032 | 0.01327 | **1.09** | 0.956 | 0.507 |
| tail-low | out | 0.03962 | 0.01457 | 0.02189 | 0.02630 | **0.66** | 0.779 | **0.348** |

**Calibration holds in region and breaks out of it.** The ratio column is predicted uncertainty over the error it predicts. In region it is 1.04 to 1.18 across all three splits, so slightly conservative. Out of region it falls to 0.69 and 0.66, so the model under-states its own error by 31% and 34%.

⚠️ **The tail's out-of-region set is conservative by about 2%, measured 2026-08-31** (`scripts/mv_ensemble.py tail --sensitivity`). The 0.05% target trim reads held-out labels, and 22 of the 28 rows it deletes fall inside this split's held-out region. Restoring the 13 ordinary ones moves out-of-region RMSE from 0.03962 to 0.04051 and the ratio from 3.24x to 3.31x; the 9 sign-flipped rows take it to 0.04200 and 3.44x. The trimmed set stays the headline for A.4 comparability, and the sign class is reported separately because those rows measure unfamiliarity with a 0.5% regime rather than distance along the split axis. Full reasoning in decision log 1.4, outlier trimming.

**The pitch sentence:** at the compact edge the error grows 3.24x while the reported uncertainty grows only 1.98x. Overconfident precisely where it is wrong, with nothing in the output to say so.

**In coverage terms, which is the form a deferral threshold can use:** the nominal 90% interval holds 0.95 to 0.97 of the truth in region, and 0.779 at the tail, falling to **0.587 in the furthest distance bin**. So at the compact edge a stated 90% interval is closer to a 60% interval.

**The signal does respond**, total rising 31% into the hole and 98% into the tail, so it should rank points for deferral even though it cannot be read as an absolute interval off-distribution. Whether it ranks well enough is the deferral curve's question.

**The hole-versus-tail contrast survives the move to ensembles:** out-of-region RMSE is 1.98x in-region for the hole against 3.24x for the tail, against 1.80 and 3.02 from the single-MLP grid.

⚠️ **Aleatoric rises off-distribution too**, 0.01032 to 0.02189 at the tail, and it rises *more* than epistemic does (2.1x against 1.7x). Noise in the world cannot depend on where you stand, so this is more evidence the term measures model misfit. Consistent with step 2, and worth stating plainly rather than being caught on.

⚠️ **The tail is BIASED, not merely overconfident, and this is new (2026-08-30, step 6).** PIT mean out of region is **0.348** at the tail against 0.51 to 0.53 everywhere else, falling to **0.153** in the furthest bin. PIT below 0.5 means the truth keeps landing below the predicted mean, so **the model systematically over-predicts edge rotational transform at the compact edge.** That is a different fault from a too-narrow interval and widening the error bars would not fix it. The hole shows nothing like it (0.531), so this separates the hole and tail conditions a second time, independently of the coverage gap.

⚠️ **That contrast was confounded until 2026-08-31 and now is not.** The tail's held-out points reach 2.13 std out while the hole's stop at 0.45, so comparing the two aggregates compared different distance distributions, exactly the error the matched-distance analysis exists to prevent. PIT is now binned by distance (`scripts/calibration.py`), and the contrast survives: **the hole sits flat at 0.52 to 0.55 across its entire range with no trend, while the tail is already at 0.429 in its nearest bin and falls to 0.153.** So the tail is biased everywhere out of region, not only far out, and the hole is not biased anywhere. ⚠️ The previously quoted 0.223 for the furthest bin was never computed by anything; the real value is 0.153. Binning PIT means is not what decision log 7.1 cut, which was the per-bin PIT *histogram* resting on about 60 points a bar; a per-bin mean rests on all 675. ⚠️ It also means the calibration failure at the tail is not purely a variance problem, so do not describe it as one. This came from the aggregate PIT histogram, the diagnostic that was cut from decision log 7.1 and reinstated the same day.

⚠️ **THE AGGREGATION FIX (2026-08-30). Every uncertainty number in this file changed and none of the models did.** `mv_ensemble.summarise` and `variance_check.run_level` both summarised per-point variances as `mean(sqrt(variance))`, the average standard deviation. The correct quantity is `sqrt(mean(variance))`, the root mean square, and it follows directly from what the calibration ratio claims: a calibrated point satisfies E[(y − mu)²] = sigma², so averaging over points gives RMSE² = mean(sigma²), and the thing that should equal RMSE is sqrt(mean(sigma²)). By Jensen, mean(sigma) sits below that, by a margin that grows with how much sigma varies across points, which is exactly the off-distribution case this project is about.

**How it was caught:** `scripts/calibration.py` computed the same quantity a second way and disagreed, 0.01487 against 0.01314 on random in-region. Independently confirmed by in-region coverage reading 0.95 to 0.97 against a nominal 0.9 on all three splits, which is over-dispersion the old ratio of 1.05 could not explain.

**What moved:** every epistemic, aleatoric and total figure rose. Calibration ratios went from 0.93 to 1.05 in region and 0.56 to 0.62 out, to 1.04 to 1.18 and 0.66 to 0.69. The tail's under-statement is 34%, not 44%. "Well calibrated in region" became "slightly conservative in region", which is a better claim anyway, since perfect calibration on the first attempt invites suspicion.

**What did not move:** RMSE, every per-point prediction, and the direction of every finding. Reruns produced byte-identical `_points.csv` files, which was the check that this was a reporting fix and not a modelling one. The variance-head verdict is still RECOVERS.

**The fix lives in `metrics.rms_uncertainty`,** used by all three scripts, with four tests. It was inline in two places, which is what a missing shared definition looks like: the same bug written twice, independently.

**Step 6 done (2026-08-30), `scripts/calibration.py`.** No training, reads the three `_points.csv` in seconds. Coverage versus nominal at six levels, CRPS, and PIT, all on total uncertainty, with coverage and CRPS binned by distance using `metrics.distance_bins` so the curves share the distance-error figure's x axis. Writes `results/calibration.json`, `_bins.csv` and `_points.csv`.

**Coverage degrades with distance, which is the deliverable.** Tail: 0.928 at the nearest bin falling to 0.587 at 1.63 to 2.13 std out, with one non-monotone step (0.781 then 0.822 at the third and fourth bins). ⚠️ This read "monotonically" until 2026-08-31, which the bins contradict. Hole: 0.916 to 0.806, shallower and it flattens. At matched distance around 0.4 std the tail is worse than the hole (0.78 against 0.81), consistent with the single-MLP distance-matched premium.

**Step 7 done (2026-08-30), `scripts/deferral.py`.** Deliverable two, no training, reads the tail points file in seconds. Rank the held-out shapes by uncertainty, defer the worst fraction to VMEC++, credit those exact, score the hybrid system by CRPS in physical units.

| ranking | 0% | 10% | 20% | 30% | 50% | AUC | halve at |
|---|---|---|---|---|---|---|---|
| epistemic | 0.01947 | 0.01526 | 0.01181 | 0.00908 | 0.00525 | 0.00667 | 27.4% |
| total | 0.01947 | 0.01467 | 0.01150 | 0.00892 | 0.00497 | **0.00642** | **26.9%** |
| random | 0.01947 | 0.01752 | 0.01556 | 0.01358 | 0.00974 | 0.00973 | 50.0% |
| oracle | 0.01947 | 0.01144 | 0.00792 | 0.00569 | 0.00290 | 0.00451 | 14.2% |

**Deferral works.** Solving the worst 20% by uncertainty cuts CRPS 41%, against 20% for a random 20%. To halve the error you solve 27% of designs instead of 50%. Perfect ranking would need 14.2%, so the signal captures **63% of the available headroom** out of region, and 76% in region.

**The result that pairs steps 6 and 7, and the honest framing of the whole project:** the intervals are badly miscalibrated at the compact edge (ratio 0.66, coverage 0.587 in the furthest bin), and the ranking still works there. **A signal can be useless as an absolute interval and still be good at ordering.** Say this rather than either half alone.

**Total beats epistemic, and this is now a result rather than a lean.** Paired within each ensemble, total-ranked wins in **all nine runs**, three splits by three seeds, by 0.8% to 5.8% on AUC with a mean near 3.5%. On the tail out of region it is 0.00647 ± 0.00018 against 0.00677 ± 0.00026. ⚠️ Compared *unpaired* the two overlap across seeds, so the paired comparison is the one to quote: both signals come from the same ensemble, so the difference is what varies, not the level. ⚠️ It read "at every rate and in both regions" until 2026-08-31, which was false at 1 of 49 rates per region. Decision log 8.1 deliberately left this open, so the answer is a result either way. Consistent with aleatoric measuring model misfit rather than noise: misfit carries real information about local difficulty, so including it helps the ranking. ⚠️ A 4% margin at one seed. Report it as a consistent lean, not a finding, and note that decision log 8.1 left this open deliberately so the answer is a result either way.

**Sanity check built into the output:** the 0% column is identical across all four rankings, since nothing is deferred there and the ranking cannot matter. A mismatch would mean the cumulative-sum indexing is wrong.

⚠️ **The curve assumes the solver always succeeds.** Deferred points are credited the exact value. Decision log 8.6 leaves it open as a stretch item whether VMEC++ failures cluster in the compact region, which would make this optimistic.

**Step 8 done (2026-08-30), `scripts/n_sweep.py`, 30 ensembles in 1386s on a T4.** The last training run. 5 sizes x 3 seeds x 2 noise conditions, tail split, primary target. Everything frozen except N and the injected noise.

**VERDICT: the decomposition holds.** Both terms behave as their names claim.

| N | epistemic (clean) | aleatoric (clean) | aleatoric (σ=0.020) | recovered σ |
|---|---|---|---|---|
| 1,000 | 0.01475 | 0.03981 | 0.05145 | 0.0318 ± 0.0041 |
| 2,000 | 0.01300 | 0.02554 | 0.03525 | 0.0241 ± 0.0023 |
| 4,000 | 0.01121 | 0.01977 | 0.02980 | 0.0223 ± 0.0003 |
| 8,000 | 0.00984 | 0.01431 | 0.02648 | 0.0223 ± 0.0009 |
| 16,793 | 0.00873 | 0.01111 | 0.02362 | **0.0208 ± 0.0003** |

**The clean statement of the result, and the form to use in the write-up:** the recovered σ column is the injected component pulled back out by quadrature subtraction, `sqrt(noisy² − clean²)`, within seed. **The reducible part falls from 0.0398 to 0.0111 as N grows 17x, while the recovered injected part converges to 0.020 from above**, reaching 0.0208 ± 0.0003 against a true 0.020.

⚠️ **This read "sits at 0.020 and does not move" until 2026-08-31, which the column contradicts:** it runs 0.0318, 0.0241, 0.0223, 0.0223, 0.0208, and all fifteen cells sit above 0.020. The mechanism is worth stating rather than softening the sentence. `sqrt(noisy² − clean²)` credits the *clean* run's misfit to the noisy run, but noisy training fits worse, so the subtraction over-attributes and every value lands above σ. Both misfits shrink with N, so the excess shrinks too. That makes the convergence evidence for the decomposition rather than noise against it. Epistemic shrinks monotonically in both conditions, 0.01475 to 0.00873 clean, with a per-rung seed spread around ±1% against a trend of x0.59.

This is what earns the two terms their names, and it independently reproduces step 3's verdict by a different route, since step 3 compared magnitudes at one N and this tracks them across seventeenfold.

⚠️ **The pre-registered prediction was correct, and the attestation is this session's record rather than the repository's.** Predicted ~0.023 for noisy aleatoric at full N before the run, measured 0.0236. ⚠️ `git log -S` shows the prediction text entering in the same commit as the sweep results, so prediction and outcome are indistinguishable in the git history. Nothing contradicts the claim and nothing independently supports it. **Standing rule from 2026-08-31: commit a prediction in its own commit before the run that tests it.** The earlier wording, "aleatoric stays at 0.020 as N grows twentyfold", was corrected to the quadrature form before the sweep ran, not after seeing it.

**Second finding, and it was not what the sweep was built for: the extrapolation gap widens with data.** Out-of-region over in-region RMSE, averaged over seeds: 1.71, 2.13, 2.53, 2.92, **3.39**. Monotone, spreads of ±0.06 to ±0.14, and the endpoints do not overlap. In-region error falls 67% over the sweep while out-of-region falls only 29%.

⚠️ **State that finding narrowly.** Subsampling happens *within* the training region, so "more data" here means more of the same distribution, and more of a distribution cannot populate a region it excludes. The defensible claim is that **scaling the existing dataset does not buy extrapolation reliability**, so the answer is targeted sampling in the sparse region or deferral, not more of the same. That is also what the two 2026 compact-stellarator papers are doing.

⚠️ **A third claim was drafted and then withdrawn on the evidence, before it reached any figure.** "Out-of-region calibration degrades as N grows" came from reading seed 0 alone, where the ratio ran 0.96 to 0.62. Across all three seeds it is 0.85 ± 0.11 at N = 1,000 and then **flat within noise from N = 2,000 onward: 0.62, 0.62, 0.65, 0.66.** The entire trend was the N = 1,000 point, where the model is poor everywhere and its uncertainty is honestly large. **Do not claim calibration worsens with data.** The correct statement is that out-of-region calibration is stuck near 0.65 and more data does not fix it, which is a supporting point for deferral rather than a finding of its own.

⚠️ **One artifact, already understood from step 3.** Coverage in the noisy condition rises to 0.99 in-region. The model predicts for the noisy distribution it trained on and is scored against clean targets, so over-covering is correct behaviour.

**No rate or exponent is reported**, per the standing decision to report the shape only. Five rungs at three seeds show a shape and nothing finer.

**Next: work through `audit-findings.md`.** All training is done and the figures exist.

⚠️ **Three adversarial audits ran on 2026-08-30** (data and leakage, uncertainty mathematics, claims against artifacts), by a separate model told to treat this file and the decision log as claims under test. **Every table recomputed reproduced to the printed digit, and the core machinery came back clean:** units, variance-space aggregation, the closed forms, the NLL, set disjointness. The risk sits in the reporting, where prose compressed curves into single numbers the bins contradict.

**Findings not yet fixed, so parts of this file are known to be wrong until they are.** Full list, with fixes and reasoning, in `audit-findings.md`. The four that matter:

- ✅ **The 0.05% target trim reads held-out labels**, deleting 22 rows from the tail's held-out set. **Measured and closed 2026-08-31:** the headline is conservative by about 2% (3.24x against 3.31x), or 6% including the sign class. Decision log 1.4 carries the table and the rewritten justification.
- ✅ **PIT "0.223 in the furthest bin" was never computed by anything. Closed 2026-08-31:** the real value is 0.153, PIT is now binned by distance, and the hole/tail bias contrast survives at matched distance.
- ✅ **"About 15% at matched distance" was wrong**, an artifact of unequal bin widths. **Closed 2026-08-31:** measured in identical windows it is 12% over the shared range, with no reliable trend across windows.
- ✅ **Everything except the N-sweep was one seed. Closed 2026-08-31** by replicating all three splits over seeds 0, 1 and 2. Seed noise is 0.2% to 6.7% and every headline clears it by about an order of magnitude. `scripts/seed_spread.py`, `results/seed_spread.json`.

Two loose ends, neither blocking anything: the mirror-pair search (folded into decision log 2.2, the tolerance duplicate check) and a three-seed rerun of the narrow sweep's p30 and p90 if the U's upper arm is ever load-bearing.

---

## The dataset (verified facts, do not re-derive)

Hugging Face: `proxima-fusion/constellaration`. Paper: arXiv 2506.19583 (NeurIPS 2025). Code: `github.com/proximafusion/constellaration`.

**Twelve subsets, not one.** `default` (182k rows), `finite_beta_1pct` through `5pct` (113k, 82.7k, 55.4k, 36.4k, 26.7k), plus six matching `vmecpp_wout` subsets holding full equilibrium objects. They sum to roughly 958k, which explains the ~959k row count the HF viewer shows. The paper's ~158k is the converged vacuum configurations inside `default`'s 182k candidates.

**We use `default` only.** Mixing vacuum and finite-beta rows without beta as an input would put different answers on identical inputs. That is the one genuine source of unobserved-confounder noise here and it is avoidable.

**Inputs.** `boundary.r_cos` and `boundary.z_sin`, each a nested list of shape `(5, 9)`: poloidal `m` = 0..4, toroidal `n` = −4..4. `boundary.r_sin` and `boundary.z_cos` are null-typed columns because all configurations are stellarator-symmetric.

**The 80 columns, verified against their code and the data (2026-08-26).** `_to_X` in `src/constellaration/generative_model/bootstrap_dataset.py` ravels each `(5,9)` array to 45 and drops the first 5 entries (`m=0, n=−4..0`), keeping 40 each, 80 total. Those 5 dropped entries are the four symmetry zeros plus R(0,0). Confirmed empirically over all 182,221 usable rows: `r_cos[m=0,n<0]`, `z_sin[m=0,n<0]` and `Z(0,0)` are all *exactly* 0.0.

⚠️ **R(0,0) is NOT exactly 1 in the stored data**, contrary to what this file previously said. Range 0.894 to 1.016, std 0.0051, only 52.8% of rows within 1e-4 of 1.0. Dropping it is still correct: it is 6 to 20x less variable than the least-variable genuine coefficient, correlates weakly with aspect ratio (0.14), and their inverse `_x_to_surface` hardcodes `r_cos[max_toroidal_mode] = 1.0` on reconstruction. Treat it as optimizer residue around a fixed convention, not a degree of freedom.

⚠️ **The encoding is not canonicalized for sign.** Negating every `z_sin` coefficient leaves R unchanged and sends Z to −Z, i.e. the mirror image through the horizontal midplane, so two different 80-vectors can describe mirror-image shapes. The pool is 89.4% / 10.6% mixed on the reference coefficient `z_sin[m=0,n=1]`. Their `_flip_z_sin_if_negative` canonicalizes this, but only inside `_augment_dataset`, which is generative-model-specific; their surrogate feature extractor `_to_X` does not. **Decision: do not canonicalize** (decision log 1.7). Two reasons: `_to_X` does not canonicalize and Table 7 comparability is the point, and the two handedness classes are not interchangeable (the reference sign correlates with the target at −0.29, and the classes have different aspect ratio, elongation and triangularity distributions).

⚠️ **Do not say mirroring flips the sign of the target.** That was the old justification and it is wrong (corrected 2026-08-27, decision log 1.7). Verified in their repo: `forward_model.py:135-144` stores VMEC's *signed* edge iota with no `abs`; all three benchmark problems apply `np.abs()` at scoring (`problems.py` 174, 248, 384); their own ALM optimizer does not, so their scorer and their generator disagree. In the pool both handedness classes are overwhelmingly positive (24,041/24,169 and 2,848/2,853), with only 133 negative targets in 27,022. The generation optimizer selected for positive signed iota, which is why negatives are rare. Whether a genuine mirrored *pair* carries opposite iota is still unproven and parked; the pair search folded into decision log 2.2 answers it nearly free, and nothing downstream depends on it.

**Field periods.** `boundary.n_field_periods` takes values 1 to 5, with 15k, 20k, 68k, 27k and 28k configurations respectively. The toroidal mode index is in units of NFP, so the coefficient vector means something different at each value. Proxima's own example filters to NFP=3.

**The twelve metric columns:**

Geometry only, computable from the coefficients without solving:
- `aspect_ratio` (compactness; major over minor radius)
- `max_elongation`
- `average_triangularity`

Require the equilibrium solve:
- `edge_rotational_transform_over_n_field_periods` and `axis_rotational_transform_over_n_field_periods`
- `qi` (quasi-isodynamicity residual; how far the field is from the ideal confining symmetry, lower is better; spans ~2 orders of magnitude, needs log10)
- `vacuum_well` (MHD stability proxy)
- `edge_magnetic_mirror_ratio` and `axis_magnetic_mirror_ratio`
- `flux_compression_in_regions_of_bad_curvature` (turbulent transport proxy)
- `minimum_normalized_magnetic_gradient_scale_length` (coil simplicity proxy)
- `aspect_ratio_over_edge_rotational_transform` (divides by the *signed* edge transform, `forward_model.py:214`, so it inherits the sign; matters only if it is ever promoted to a target)

There is **no effective ripple column**. Computing it would need the `vmecpp_wout` equilibria plus a Boozer transform and a neoclassical code. Out of scope.

**How the data was generated.** Four pathways: physics-informed heuristics (30k), near-axis expansion via pyQSC (49k), DESC optimization run twice per target (88k), and VMEC++ in the loop (15k targets). Roughly 182k candidates, 158k evaluated without errors. Targets were sampled from ranges (Table 6: NFP 1-5, edge rotational transform per field period 0.1-0.3, aspect ratio 4.0-12.0, max elongation 4.0-7.0, edge mirror ratio 0.1-0.4), not laid out as a designed experiment. Several quantities were upper-bound constraints during optimization, so achieved values are skewed relative to targets.

**Consequence that matters:** sampling density reflects where their optimizers converged, not uniform coverage of design space. This is the foundation of the extrapolation argument.

**Independently corroborated, do not present this as only our own inference.** Two later papers describe the low-aspect-ratio region as underpopulated and are actively generating configurations there: the domain-adaptive latent diffusion work (arXiv 2608.16938) targets "the sparsely sampled low-aspect-ratio regime" in those words, and "Data-Driven Generation of Compact Quasi-Isodynamic Stellarators" (2026) works the same region. This upgrades the split-axis premise from a reading of the generation process to a citable claim, and it strengthens the deferral argument: people are pushing into exactly the region where a silently-wrong surrogate does the most damage. Caveat: both work at NFP=4 while we are at NFP=3, so this corroborates the shape of the design space, not our exact slice.

**Failures are flagged, not deleted.** `misc.has_neurips_2025_forward_model_error` marks rows where the solver failed. Separate columns exist for generation-stage failures.

**Null convention, confirmed from their loader.** `load_source_datasets_with_no_errors` applies `.fillna(False)` across all five error flags and drops any row with a True. So nulls count as "no error", and the filter is **all five flags**, not just the NeurIPS one.

⚠️ **The null-row trap.** One row in 182,222 (file 3, index 60739) has `boundary.r_cos`, `boundary.z_sin`, `n_field_periods` and every metric set to `None`. It failed at boundary *generation*, so the solver never ran, so `has_neurips_2025_forward_model_error` reads `False` while `has_optimize_boundary_omnigenity_desc_error` is `True`. Filtering on the NeurIPS flag alone lets it through, and it then either throws on `np.array(None)` or silently yields NaNs. **Always drop null `boundary.r_cos`/`z_sin` as an explicit standalone check**, never as a side effect of an error flag.

**The verified filter chain (2026-08-26).** Raw 182,222 → five-flag error filter 158,685 → NFP=3 68,191 → `desc` or `vmec` pathway 27,050 → null-boundary guard 27,050 → 0.05% target-only trim ~27,023. Two intermediate counts land exactly on published figures: 158,685 is the paper's "~158k evaluated without errors", and 68,191 is the documented "68k at NFP=3" (post-error-filter; the raw NFP=3 count is 82,043).

**Generation pathway has no single column.** Read it off which of four `*_optimization_settings.id` fields is non-null. Partitions cleanly (182,221 of 182,222 have exactly one). Counts match the documented pathways: `desc` 87,727, `nae_init` 48,851, `qp_init` 29,886, `vmec` 15,757. These ids identify *settings groups*, not rows (17,671 desc rows share 148 unique ids), so they cannot be used for deduplication and there is no per-row config id.

**Class imbalance on feasibility:** roughly 41 and 52 feasible points out of ~160k for two relaxed benchmark problems. This is why the project does regression on a continuous metric rather than feasibility classification.

**Appendix A.4 is the direct precedent.** Proxima trained an ensemble of ten MLPs (three layers, 256 hidden units, tanh, MSE loss, z-scored targets using training statistics) predicting the twelve metrics from boundary coefficients. Whether that is one network with twelve outputs or twelve single-output models is not stated anywhere, make no assumption either way (see the comparability note below). Filtered to vacuum, NFP=3, DESC or VMEC-optimized boundaries only, 0.05% tails trimmed per metric, ~23k points, 80/20 split.

⚠️ **Their ~23k does not reproduce; we get 27,050.** Applying every filter their published loader actually performs gives 27,050, about 15% high. Ruled out by direct test: exact duplicates (8 in 27,050), the five-flag filter (removes 1 row beyond the NeurIPS flag alone), and the documented 0.05% per-metric trim (leaves 26,746; reaching 23k needs roughly a 1% trim). Two surviving explanations, not distinguishable from public information: A.4's appendix describes its filter chain incompletely, or the HF dataset grew after the NeurIPS submission. **Decided: accept 27,050 and document the discrepancy** rather than inventing an undocumented filter to force the number down. A 15% pool difference will not change which (axis, direction) pair wins the day 1-2 grid.

Table 7, accurately: R² is above 0.97 for every metric (lowest 0.974, `axis_magnetic_mirror_ratio`). RMSE is **not** uniformly small, it spans 0.006 to 0.581 and three metrics exceed 0.1: `aspect_ratio_over_edge_rotational_transform` 0.581, `minimum_normalized_magnetic_gradient_scale_length` 0.330, `max_elongation` 0.161. The two rows that matter here: `edge_rotational_transform_over_n_field_periods` at RMSE 0.006 / R² 0.997, and `log_10_qi` at RMSE 0.051 / R² 0.982.

They then wrote that such surrogates are prone to extrapolation errors when queried far from the training distribution, and named uncertainty calibration, active learning, and physics-informed strategies as future work.

Reproducing their in-region numbers (RMSE and R² specifically,
see NRMSE/SNR caveat below) is the sanity anchor and the
comparability claim.

Architecture comparability is unverified, say so accurately
rather than asserting a specific mismatch. The paper never
states whether A.4's ensemble is one shared network with a
twelve-metric output or twelve separate single-metric
ensembles trained with the same architecture template, "the
MLPs mapped coefficients to target key metrics" is genuinely
ambiguous either way, and the A.4 training code isn't in the
public repo to check. What's certain: ours is single-target,
one mean-variance network per metric. The comparison is
same-metric test performance, not a verified same-or-different
architecture claim in either direction.

**Cost caveat.** The paper says a VMEC++ run took around an hour on a 32 vCPU machine. Read in context (Appendix A.3), that is the time budget for a full optimization run, not a single forward equilibrium solve. Do not quote it as per-call cost. The deferral curve's x-axis is solver calls saved, not wall-clock.

---

## Conceptual commitments (already worked out, do not relitigate)

**Label noise is zero, and that is measured, not assumed.** The inputs are fully observed (no truncation), the solver is deterministic, a fixed convergence tolerance produces a deterministic high-frequency function rather than noise, and the four generation pathways shift the distribution over x without changing y given x. Stage 2 confirmed it: near-twin shapes differ in target by exactly 0.000000. Anyone arguing that the mixture of pathways creates aleatoric noise is confusing covariate shift with label noise.

⚠️ **This does NOT mean the variance head will report near zero, and the earlier version of this file wrongly said it would.** Corrected 2026-08-29 by step 2, `scripts/mv_ensemble.py random`, which reports **aleatoric 0.01194**, 15% of the target std and larger than the epistemic term's 0.00887.

**Both facts are true because they answer different questions.** Stage 2 measured whether the *world* is noisy: it is not. The variance head measures the residual scatter *this model* cannot predict, which includes its own misfit. A three-layer MLP on 80 coefficients cannot represent the function exactly, so it is consistently off by about 0.01, and from inside the model that error is indistinguishable from noise. Under NLL the honest thing is to widen the interval, so it does. What the head reports is "irreducible given this architecture and this input representation", not "irreducible in principle".

**Consequence for the write-up:** never present the aleatoric number as the dataset's noise level. It is a property of the model, and the gap between it and Stage 2's zero is model misfit wearing the wrong label. This is the concrete case of the relativity note below, not a contradiction of it.

**Consequence for the variance-head check:** it gets stronger, not weaker. The question is no longer "does aleatoric rise off a floor of zero", which a saturated head could fake. It is "does aleatoric rise by roughly the injected amount, on top of a visible baseline of 0.0119", and there is now room for that answer to be wrong.

**Therefore: manufacture aleatoric deliberately**, then check whether the variance head recovers the right magnitude. This converts the decomposition from an assertion into a validated measurement, and it is the reason the mean-variance head is not decorative.

⚠️ **The instrument changed on 2026-08-29, from hiding input coefficients to adding target noise.** `scripts/variance_check.py`, decision log 6.2. Hiding was tried first and priced before training: four rules, dropping 18 to 36 of the 80 coefficients, every one injecting about 0.003, a 3 to 6% rise on the then-reported 0.01048 baseline (0.01194 after the aggregation fix, which does not change the conclusion), which is inside seed noise. Dropping 36 columns injected less than dropping 18. Either the near-twin estimator is biased low, because pairs close in the kept coordinates are also close in the dropped ones on a pool where every shape came from the same optimizers, or those coefficients genuinely carry little information about the rotational transform. The replacement adds Gaussian noise of known σ to the training targets at three levels, 0.005 / 0.020 / 0.050, so the injected magnitude is exact rather than estimated and the answer is a curve rather than one coincidence. It trades realism for an exact answer key, which is the right trade for a validation.

**Keep from the failed attempt:** dropping up to 45% of the boundary description barely changes what is predictable about the edge rotational transform. A real observation about the dataset, caveated by the possible estimator bias.

(Hiding field period was never viable, it's fixed at NFP=3 across the whole dataset.)

**Aleatoric and epistemic are relative, not absolute.** The split is relative to model class, prior, and input representation, and only separates cleanly under correct specification. The ensemble decomposition is a law-of-total-variance statement about the mixture, not about a Bayesian posterior.

**Calibration quality and posterior-approximation fidelity are different things and come apart.**

**Marginal coverage hides conditional miscalibration.** Report coverage binned by predicted uncertainty and by region, not just aggregate.

**Gaussian NLL failure mode.** The gradient on the mean is scaled by 1/σ², so points the model fits poorly get high predicted σ and are then down-weighted, which stalls learning where it is most needed. Fixes: MSE warm-up, variance floor, β-NLL (Seitzer et al. 2022). Note this is not literally "variance collapse" in the σ→0 sense; the loose usage invites correction.

**Do not subtract known aleatoric from total to recover epistemic.** Algebraically valid, fragile, and unnecessary since ensembles give both terms directly.

---

## Decisions already settled

- `default` subset only, no finite-beta mixing.
- Ensemble of mean-variance networks; epistemic from spread of means, aleatoric from mean of variances.
- Architecture follows A.4: three layers, 256 units, tanh.
- **Recipe FROZEN 2026-08-29** by `scripts/hp_check.py`, the
  step 0 sanity check. Values live in
  `src/constellaration_uq/nets.py` as module constants, not
  config fields: Adam with no weight decay, lr 1e-3, batch 128,
  500 fixed validation points for early stopping, 500 max
  epochs, patience 20 restoring best weights. Recorded in
  decision log 4.7. Do not change them to make a run finish
  faster, the N-sweep's claim is that only data volume varied.
  ⚠️ **The chosen lr is not the check's nominal winner.** 3e-4
  beat 1e-3 by 0.9% at one seed, which is noise, and cost 210
  epochs against 75. Over 300 sweep networks that is 3.7 hours
  against 1.3. Ties on accuracy break on cost.
  Batch 512 lost outright at all three learning rates.
  500 validation points confirmed sufficient at full size,
  jitter 0.030.
  ⚠️ **The N=1000 half of that claim was withdrawn on
  2026-08-31.** `results/hp_check.json` shows the small-N row
  ran at lr 3e-4, the nominal validation-loss winner, not the
  frozen 1e-3, and a higher learning rate gives a noisier
  validation curve, so the frozen recipe's stopping signal at
  N=1000 was never checked by that script. **Better evidence
  exists and supersedes it:** the N-sweep ran three seeds at
  N=1000 under the frozen recipe and in-region epistemic came
  out 0.01477 / 0.01486 / 0.01462, a spread near ±1%.
  Incidental, and it is the sweep's premise arriving early: the
  same recipe at N=1000 gives RMSE 0.045 against 0.0115 at full
  size, a 4x degradation from data volume alone.
  ⚠️ **The check ran on the random split, selecting on
  validation loss only.** Choosing hyperparameters by tail-split
  performance leaks the extrapolation condition into the model
  and invalidates the study invisibly. `hp_check.py` never
  constructs a tail split, which makes that impossible rather
  than merely discouraged.
- Log10 transform for qi. Targets z-scored using training statistics only.
- Input scaler fitted on training data only. Out-of-region inputs will fall outside the fitted range. Do not clip.
- Early-stopping validation set drawn from in-region data only.
- Failed rows loaded separately for the survivorship figure, never used for training.
- N-sweep: ~5 values of N (training set size, e.g. 1k/2k/4k/8k/
  18k), 3 seeds each, subsampling done **within** the training
  region so N varies and coverage does not.
  Scope is pinned, do not let it multiply: the **tail split
  only** (the random split has no out-of-region set to measure
  "epistemic rises off-distribution" against; the interior hole
  could serve but would double the work to answer a secondary
  question) and the **primary target only** (the sweep validates
  the uncertainty machinery, not any one metric, so repeating it
  on a second target retests the same thing). If the secondary
  target is pursued it gets its own ensemble, not its own sweep.
  Fixed test sets, frozen across all N: the tail held-out set
  for the off-distribution check, plus a fixed in-region slice
  for the shrinks-with-N check.
  Run the whole sweep in **both noise conditions**: clean
  targets, and targets with Gaussian noise of σ = 0.020 added.
  ⚠️ This said "both input conditions", all 80 coefficients
  against high mode-numbers hidden, until 2026-08-29. Decision
  log 6.2 retired that instrument after measuring it injects
  about 0.003 however much is dropped. The sweep inherits the
  replacement.
  Without a second condition one of the three checks is vacuous:
  on clean targets aleatoric is the model's own misfit, which
  *does* shrink with N, so "aleatoric stays flat" has no content.
  A fixed injected floor gives it something that genuinely should
  not move, and the contrast is the result: epistemic decaying
  toward a flat aleatoric floor.
  σ = 0.020 because 0.005 is too close to the baseline misfit to
  separate from it and 0.050 dominates everything, hiding the
  epistemic decay. 0.020 is 25% of the target's spread and
  recovered at ratio 1.07 in step 3.
  ⚠️ **PRE-REGISTERED 2026-08-30, before the sweep ran, and it
  corrects an earlier over-claim.** This entry used to read
  "aleatoric stays at 0.020 as N grows twentyfold", which is
  wrong and would have been falsified by the run. The two terms
  add in variance, so what the head reports under injection is
  sqrt(misfit² + σ²), and misfit is large at the small rungs:
  step 0 measured RMSE about 0.045 at N = 1000 against 0.0115 at
  full size. A tiny 4-ensemble smoke run at N = 200 and 400 makes
  it concrete, aleatoric came out 0.067 with σ = 0.020 injected,
  because the injection is invisible next to a misfit of 0.056.
  **Predicted, in advance:**

  | N | clean aleatoric | noisy aleatoric |
  |---|---|---|
  | 1,000 | ~0.05 | ~0.054 |
  | full | ~0.012 | ~0.023 |

  **So the correct claim is not a flat line.** The two conditions
  converge at small N and separate at large N, and the signature
  to look for is **the clean curve continuing to fall while the
  noisy one flattens near 0.023.** The floor is still exact and
  still falsifiable to a number, it is just sqrt(misfit² + σ²)
  rather than σ, and the misfit half is measured at each N by the
  clean condition. That is what the two conditions are for.
  ⚠️ Do not report "aleatoric stayed flat" if it did not. The
  contrast between the conditions is the result, not either curve
  on its own.
  Budget: 5 N x 3 seeds x 2 noise conditions = 30 ensembles,
  300 member networks. Any expansion past that is a visible
  decision, not a drift.
- If the schedule slips, **shrink the sweep grid, never drop
  it**: 3 N x 2 seeds x 2 noise conditions = 12 ensembles. The
  reporting commitment is the shape of the epistemic decay, not
  a fitted exponent, and 12 points show the shape. Nothing else
  can test whether the epistemic term is reducible with data,
  which is the only thing that earns it the name.
  ⚠️ Do not assume the sweep is expensive before measuring it.
  These are small networks on GPU. Time one member network
  before the ensemble work starts.
- N-sweep ordering: the full sweep runs **last**, after
  calibration and the deferral curve, because those two are the
  named deliverables and the sweep validates machinery.
  **Why that is safe:** calibration and the deferral curve
  measure the uncertainty signal rather than assuming it. A
  useless signal puts the deferral curve flat on the random
  floor, and bad intervals show up in the coverage plot. So a
  broken signal surfaces at those steps, not at the sweep. What
  the sweep uniquely tests is narrower: whether the split into
  ignorance and noise is real, meaning epistemic shrinks with N
  and aleatoric does not. A signal can rank points well, and so
  give valid calibration and deferral results, while failing
  that test.
  ⚠️ **The one risk that does not wait** is a variance head that
  reports a number unrelated to the data. There is no ground
  truth for aleatoric on this dataset, so a broken head looks
  correct. The variance-head check is the only thing that
  catches it, and it costs **three ensembles**: the random
  split's mean-variance run is the clean-target reference, and
  the check adds one run per injected noise level, 0.005 / 0.020
  / 0.050. Run it as soon as the random-split ensemble exists,
  before the hole and tail runs, so a bad loss costs one headline
  ensemble rather than three. Full reasoning in decision log 6.5,
  when the sweep runs, and 6.2, what the injection is.
- Report the shape of the epistemic decay, not a fitted exponent.
- Deferral curve needs four lines: two "mine" curves
  (epistemic-ranked and total-ranked, see below), random
  deferral at matched rate (floor), and oracle deferral
  (ceiling). The oracle defers by per-point CRPS, not by true
  absolute error, since CRPS is the y-axis. An |error|-ranked
  oracle is a ceiling for MAE and would not actually bound the
  curve being plotted, leaving the "mine" curves free to cross
  it.
- Before building the ensemble, validate the split axis with a
  cheap baseline (ridge regression or a single small MLP): fit
  in-region, compare error in-region vs out-of-region. Proceed
  with the full ensemble only if a real gap shows up.
- Calibration reported binned by continuous distance from the
  training region, not only as a binary in-region/out-of-region
  comparison.
- Field periods: NFP=3, matching Proxima's own example filter.
- Generation pathways: match A.4 exactly (DESC/VMEC-optimized
  only, NFP=3, ~23k points), for direct comparability to
  Table 7. Chosen over keeping all four pathways since the
  day 1-2 baseline check already covers the risk of this pool
  being too thin at the compact aspect-ratio end.
- Outlier trimming: a 0.05% per-tail trim on the target metric
  only, never on the split axis. ⚠️ **Justified on its own terms,
  not on matching A.4** (decision log 1.4, restated 2026-08-31):
  a squared-error metric must not be decided by 0.1% of rows, and
  the extreme low tail is a sign-convention artifact their own
  scorer removes with `np.abs()`. A.4 alignment is a bonus. Their
  exact procedure is inferred from appendix prose, their training
  code is not public, and their random split makes the trim
  harmless in a way it is not here. The cost on the tail split is
  measured, not assumed: about 2%.
- Ensemble size: 10, matching A.4.
- **Seed replication FROZEN 2026-08-31: three seeds (0, 1, 2)
  on all three splits, plus three on the single-MLP distance
  curve.** `scripts/seed_spread.py` aggregates whatever seeds
  exist and reports mean, spread and range for every headline
  quantity. Seed 0 keeps the unsuffixed filenames, so
  replication is additive and no downstream script had to learn
  about seeds.
  **Why three and not more:** seed noise came out 0.2% to 6.7%
  while every headline effect clears it by roughly an order of
  magnitude, so more seeds would tighten error bars on
  conclusions that are not in doubt. The one claim near the
  edge, the matched-distance premium at 3 sigma, is where extra
  seeds would buy something if it ever becomes load-bearing.
  ⚠️ **`base_seed = seed * N_MEMBERS`, not `seed`.**
  `train_mv_ensemble` assigns member seeds as `base_seed + k`,
  so consecutive replication seeds would share nine of ten
  member initialisations and the measured spread would be
  almost meaningless. Seed 0 is unchanged, since 0 * 10 = 0,
  which is what let the existing results stay byte-identical
  and served as the regression check.
- Diversity mechanism: init and shuffle order only, not
  bootstrap.
- Reporting space: physical units, not z-scored. Coverage and
  PIT are transform-invariant either way; RMSE, CRPS, interval
  width, and the deferral curve's y-axis all get unwound from
  z-scores before reporting.
  "Physical units" means un-z-scored, with any monotone
  transform chosen for modeling reasons left in place. Z-scoring
  exists only to help optimization and carries no meaning, so it
  always gets undone. The log10 on qi is a deliberate choice
  about what being wrong should cost, so it stays. Practically:
  edge rotational transform reports in its actual units, qi
  reports in log10(qi), never raw qi. Three reasons, any one
  sufficient. (1) Raw-qi RMSE would be dominated by the high end
  of a ~2-order-of-magnitude range, so a uniformly good model
  scores badly for reasons unrelated to its quality; relative
  error is the natural cost here and log space is where relative
  becomes absolute. (2) The predictive distribution stays
  Gaussian under un-z-scoring (an affine rescale), so the
  closed-form Gaussian CRPS holds; exponentiating to raw qi
  makes it log-normal and that formula no longer applies.
  (3) Table 7 reports `log_10_qi`, so comparability requires the
  same space.
- Code is config-driven, lightly: a single typed config object
  (target column, split axis, split direction, seed) threaded
  through one entry point, not an external config framework
  (no Hydra, no YAML). Architecture and training hyperparameters
  stay fixed constants, not config fields, consistent with the
  frozen recipe.
- No joint multi-head network across targets. Confounds the
  epistemic/aleatoric decomposition (ensemble disagreement on
  one target would be entangled with how well it also fits the
  other) and reopens the frozen-recipe question. Instead, the
  target column is a swappable parameter, so the same
  architecture and recipe reruns per target unchanged, used by
  the day 1-2 grid check and, if pursued, a fully separate
  ensemble for the secondary target.
- Held-out coverage threshold, method settled, exact value
  deferred: required points per calibration bin follows
  n = z^2 * p(1-p) / delta^2, z=1.96 for 95% confidence,
  conservative p=0.5 (worst-case variance, valid regardless of
  which nominal coverage level gets checked). 8 to 10 distance
  bins. Delta itself (and so the exact per-bin and total
  threshold) gets picked from the day 1-2 histogram once it's
  known what's actually achievable, not chosen blind beforehand.
  The threshold must be cleared twice, independently: once for a
  mid-range band (the interior hole) and once for a tail band.
  Failing either disqualifies the axis, since the main figure
  needs both results to exist. Note this is a property of the
  histogram, not a partition: the three splits are three
  separate experiments, each training its own ensemble on the
  full ~23k minus its own held-out region, so they never compete
  for data simultaneously.
- "Distance from training region" (used in the distance-binned
  calibration deliverable) is distance along the split axis to
  the nearest training point, normalized by a split-independent
  constant: the split axis's standard deviation over the full
  filtered dataset. Not full 80-dimensional input-space
  distance, which keeps the causal story tied to the one axis
  actually manipulated, and std normalization keeps the figure
  comparable regardless of which axis (aspect ratio or max
  elongation) wins the day 1-2 check.
  One formula covers all three splits. Tail: distance past the
  cutoff. Interior hole: distance from the nearer edge, peaking
  at the hole's center. Random: ~0 everywhere, which is why the
  random split gets no distance-binned figure.
  Deliberately NOT normalized to "the training region's own
  width" (the earlier wording). That denominator differs between
  hole and tail, since the hole's training region spans nearly
  the whole axis while the tail's stops at the cutoff, so it
  would put the two on different scales and break exactly the
  comparison the main figure exists to make. A dataset-level
  constant means degradation at distance 0.5 in one split is
  directly comparable to 0.5 in the other. Expect the hole's
  x-range to be short and the tail's long: that is the geometry,
  not a defect, and it is what supports "at matched distance,
  leaving the region costs more than filling a gap."
- **Calibration diagnostics FROZEN 2026-08-30** (decision log
  7.1): **coverage versus nominal, CRPS, and aggregate PIT
  histograms.** All on **total** predicted uncertainty.
  Coverage and CRPS bin by distance from the training region,
  reusing the distance-error figure's bins. All three report
  in-region and out-of-region across all three splits.
  Coverage is what converts "under-states its error by 34%"
  into "the 90% interval holds the truth X% of the time past
  this distance", which is the sentence a deferral threshold
  rests on. Measured 2026-08-30: 0.779 at the tail overall,
  0.587 in the furthest bin.
  CRPS is already the deferral curve's y-axis, so it
  costs nothing here and keeps both deliverables on one scale.
  **The one cut is PIT binned by distance.**
  ⚠️ **This started as a wider cut and was corrected the same
  day, before any figure existed.** The first version cut "PIT
  histograms and reliability diagrams" on the grounds that a
  histogram needs its own bins on top of the distance bins.
  Both halves were wrong. "Reliability diagram" in regression
  names two plots already in the set: nominal-versus-empirical
  coverage **is** coverage-versus-nominal, and predicted
  uncertainty versus realised error is already required by
  decision log 7.2's coverage-binned-by-predicted-uncertainty.
  And PIT is free: it is the same numbers as coverage, since
  coverage at level α is the fraction of PIT values inside the
  central α. One `norm.cdf` on arrays already in memory, about
  25 lines of plotting, zero compute.
  **The data-hunger objection survives for one variant only.**
  A coverage curve is cumulative so noise averages out; a
  histogram is not. Per distance bin, 10 PIT bars rest on about
  60 points each and jump around from sampling noise alone.
  Aggregate PIT rests on about 480 per bar and is solid.
  **Why aggregate PIT is in rather than merely affordable:** it
  is the only one of the four that shows the *shape* of the
  miscalibration. A U says the intervals are too narrow, a lean
  says the mean is biased instead. No single number separates
  those.
- ⚠️ **Coverage uses TOTAL, not epistemic. This amends decision
  log 7.5 (2026-08-30), which said epistemic alone.** That
  entry predates any mean-variance ensemble. In-region
  epistemic is 0.00887 against an RMSE of 0.01256, so an
  epistemic-only 90% interval under-covers badly **in-region**,
  on the random split, where step 2 measured the model honest
  to within 5%. That is arithmetic from using 60% of the
  predicted spread, not a calibration finding, and it would
  break the in-region reference point in every split at once.
  The attribution 7.5 wanted still happens, in the
  decomposition table, where the tail's epistemic rise from
  0.00704 to 0.01300 is the ignorance signal. The original
  error was location, not principle: it put the decomposition
  inside the interval instead of beside it. Total is also what
  a user of the surrogate would actually act on.
- Deferral curve does not commit to one ranking signal. Show
  epistemic-ranked and total-ranked as two separate "mine"
  curves, alongside the shared random floor and oracle ceiling.
  Which signal defers best is a result, not a setup detail, and
  showing both is nearly free (same trained ensemble, same
  predictions, just two sort orders) while directly testing the
  "total and epistemic should be close" claim empirically
  instead of assuming it.
- Loss FROZEN 2026-08-29: 25 epochs of plain MSE, then Gaussian
  NLL, with the predicted variance floored at 1e-6 in z-scored
  space. 25 comes from the step 0 history, where the frozen
  recipe reaches 1.5x its best validation loss by epoch 18 and
  1.2x by epoch 32, out of 75. The floor is six orders of
  magnitude below the signal and safely above float32 noise, so
  it stops 1/variance exploding without manufacturing an
  aleatoric term that the step 3 variance-head check would
  then read as real. β-NLL stays unused unless one of two
  observable symptoms appears in the first mean-variance run:
  variance pinned on the floor across most in-region data, or
  training destabilising once warm-up ends.
- Variance parameterization: log-variance with a floor, not
  softplus. Standard deep-ensemble choice, pairs directly with
  the already-locked MSE-warm-up-plus-variance-floor loss.
  Isolated behind one small function (raw output in, positive
  variance out), not a config field, so a future softplus swap
  is a one-function edit, not a search through the codebase.
- Noise-floor kNN check (Stage 2, before any model is built):
  distance on standardized (z-scored) coefficients, not raw,
  same reasoning as the distance-from-training-region decision,
  raw distance across 80 coefficients would be dominated by the
  low-order modes. k=5 to 10, not worth further tuning, this is
  a coarse sanity check, not a figure.
- Deferral curve evaluation set: out-of-region is the headline
  (this is the tail split's held-out portion), in-region shown
  alongside as contrast, not as a competing headline.
- Deferral curve y-axis: CRPS, not RMSE or MAE. This is the one
  figure where the full predictive distribution, not just the
  mean, is under test, and RMSE/MAE would score it identically
  whether or not the variance head does anything useful. Closed
  form for a Gaussian predictive distribution, cheap to compute.
  Collapses toward MAE numerically where aleatoric sits near the
  floor, so the difference from RMSE/MAE only shows up where it
  should, out-of-region, where epistemic widens the interval.
  RMSE stays the right tool everywhere else in the project
  (Table 7 comparison, general point-accuracy reporting).
- Train a plain MSE ensemble first, before the mean-variance
  one. Same architecture minus the variance head, one output,
  MSE loss, ten members. This is what A.4 actually built ("by
  minimizing mean squared error"), so it is the honest
  like-for-like number against Table 7, and comparing a
  mean-variance ensemble's RMSE to theirs would be comparing
  two different training objectives. Its real job is isolating
  failure: if it lands near Table 7, the loader, filters,
  trimming, scaling, and architecture are all verified at once,
  so any later degradation is attributable to the variance head
  or the NLL loss rather than the pipeline. Near-zero cost, the
  locked loss starts with an MSE warm-up phase anyway.
- NRMSE and SNR are ours, not a reproduction of the paper's.
  Checked the public repo (`proximafusion/constellaration`)
  directly, the Appendix A.4 MLP ensemble and its NRMSE/SNR
  code are not published there, so their exact formula is
  unknown. We define NRMSE = RMSE / std(y_true) and
  SNR = var(y_true) / var(residual). Only RMSE and R² are
  directly comparable to Table 7. Never claim our NRMSE or SNR
  values match theirs.

---

## Decisions still open

Full reasoning is in `constellaration-uq-decisions.md` in this repo.

**None. Every decision is settled as of 2026-08-30.** What
remains is execution: steps 6, 7 and 8 in the run table below,
then figures and the write-up.

**Recently closed (2026-08-30):** the calibration diagnostic
set (decision log 7.1) and which uncertainty signal calibration
uses (7.5, amended). See "Calibration diagnostics" under
decisions already settled.

**Recently closed (2026-08-29):** the recipe freeze (decision
log 4.7, Adam, lr 1e-3, batch 128, 500 validation points, 500
max epochs, patience 20) and the loss (decision log 4.4, MSE
warm-up of 25 epochs then Gaussian NLL, variance floored at
1e-6 in z-scored space, β-NLL held behind an observable
trigger).

**Recently closed by the day 1-2 grid (2026-08-27):** split axis
(aspect ratio), split direction (low), primary target (edge
rotational transform), secondary-target evidence (log10 qi
viable, still deferred to the week 2 gate), and δ (0.06).

**Recently closed by the placement sweep (2026-08-29):** hole
geometry. Centre the interior hole at percentile 30, spanning
p20 to p40, which is adjacent to the tail without overlapping it
and reaches 0.45 std instead of the median placement's 0.20.
That 0.45 is the ceiling on the matched-distance comparison.

---

## Traps to watch for

- Trimming outliers on the split axis (silently destroys the test set).
- Filtering bad rows on `has_neurips_2025_forward_model_error` alone (misses generation-stage failures; see the null-row trap above).
- Canonicalizing the `z_sin` sign (destroys the handedness signal and breaks comparability with `_to_X`). Note the reason: **not** because it would create opposite labels, which is the retracted argument above.
- Recalibrating on out-of-region data (silently invalidates the premise).
- Selecting hyperparameters against tail-split performance. Same failure as the line above, one step earlier in the pipeline, and easier to commit by accident. Tune on the random split, select on in-region validation.
- Splitting on the target rather than on an input-measurable quantity (confounds covariate shift with label shift).
- Letting the early-stopping validation set include out-of-region points.
- Varying training length with N during the sweep.
- Quoting the one-hour VMEC++ figure as per-call solver cost.
- Claiming broad novelty. ~14 published works already use this
  dataset (searched 2026-08-26, full record in decisions doc
  0.5). "No UQ work on this dataset" is falsifiable in one
  sentence. The defensible claim: as of August 2026, no
  published work studies uncertainty calibration or
  extrapolation reliability of surrogates on this dataset. Two
  near-misses to know by name: Bayesian optimization of
  alpha-particle confinement (arXiv 2606.19523, GP uncertainty
  but on a 38-configuration subset, as optimizer machinery, no
  calibration or shift study) and RAMBO (arXiv 2601.20043,
  cites the dataset without using it). Re-run the search if the
  write-up lands materially later than August 2026.

---

## Calendar gates

**Day 1-2 of week 1:** cheap baseline run across the full
candidate grid: axes (aspect ratio, max elongation) x cuts
(tail-low, tail-high, interior hole) x targets (edge rotational
transform, log10(qi)). Twelve cheap fits, still an afternoon.
A hole has no direction, which is why the grid is 2x3x2 and not
2x2x2. Place and size the hole by percentile, not by axis
value, centered on the median: achieved aspect ratio is skewed
(it was an upper-bound constraint during generation), so a band
defined in value space can land somewhere sparse while a
percentile band guarantees mass. Size it to hit the same
per-bin coverage requirement the tail has to meet.
Whichever (axis, direction) pair shows the strongest real
in/out error gap for the primary target, with adequate held-out
coverage in BOTH a mid-range band and a tail band, wins as the
split axis. Failing either band disqualifies the axis, the main
figure needs both the hole and the tail results to exist. The
same grid gives real evidence for whether log10(qi) is worth
pursuing as a genuine secondary target. If nothing shows a
gap, reconsider before committing further hours to the
ensemble.

**End of week 1:** data loaded, noise floor checked, one model trained, in-region numbers near Table 7. If not met, fall back to a simpler dataset (UCI or semi-synthetic) keeping the same study design, without regret. The design is the contribution; the dataset is the setting.

**End of week 2:** three splits run, calibration figures exist, deferral curve drafted. If not met, drop the second target and all optional scope. The full N-sweep is deliberately not in this gate, it runs after the deferral curve. The variance-head check is in scope for week 2 and takes about fifteen minutes.

**After the deferral curve:** the full N-sweep, 30 ensembles. In scope and not droppable. If the schedule slips it shrinks to 12, never to zero.

**Remaining training runs, in order.** Nothing else states this in one place.

| # | run | ensembles | purpose |
|---|---|---|---|
| 0 | hyperparameter sanity check | 0, six single networks | ✅ done 2026-08-29, recipe frozen in 4.7 |
| 1 | plain MSE ensemble | 1 | ✅ done 2026-08-29, RMSE 0.01052, missed a badly set 0.0105 bar by 0.2%, 1.75x Table 7 |
| 2 | mean-variance, random split | 1 | ✅ done 2026-08-29, RMSE 0.01256, aleatoric 0.01194, total 18% over-dispersed |
| 3 | variance-head check | 3 | ✅ done 2026-08-29, RECOVERS, ratios 0.94 to 1.07 over a 10x noise range |
| 4 | mean-variance, interior hole at p30 | 1 | ✅ done 2026-08-29, out/in RMSE 1.98x, calibration 0.69 out of region |
| 5 | mean-variance, tail-low | 1 | ✅ done 2026-08-29, out/in RMSE 3.24x, calibration 0.66 out of region |
| 6 | calibration figures | 0 | ✅ done 2026-08-30, coverage 0.779 at the tail against 0.956 in region, and PIT found a mean bias out there |
| 7 | deferral curve | 0 | ✅ done 2026-08-30, solving the worst 20% cuts CRPS 41% against 20% at random; total-ranked beats epistemic in all nine paired runs (run 9) |
| 8 | full N-sweep | 30, floor 12 | ✅ done 2026-08-30, HOLDS: injected σ recovered at 0.0208 against 0.020 while misfit fell 3.6x, and the extrapolation gap widened 1.71 to 3.39 |
| 9 | seed replication | 6, plus 6 single MLPs | ✅ done 2026-08-31, every headline claim survives, and total-ranked deferral wins in all nine paired runs |

Runs 2, 4 and 5 are the headline result. Run 8 is validation and goes last.

⚠️ **Runs 2 through 5 were retrained on 2026-08-30 after the aggregation fix.** Seeds are fixed, so every per-point prediction came back byte-identical and only the summaries moved. The dates above are when each run was first done; the numbers are the corrected ones.

⚠️ **The variance-head check moved ahead of the hole and tail ensembles** (decision log 4.4, the loss). It used to run after all three. If the loss is wrong that meant redoing three ensembles; now it costs one, and the reorder is free.

---

## Optional scope, in cut order

Everything here is expendable. Do not suggest starting any of it before the week 2 gate.

1. Second target (`log10(qi)`).
2. Ranked list of design-space regions where the surrogate is least confident, framed as a sampling recommendation.
3. Simulated pool-based active learning comparing acquisition on random, total, epistemic, aleatoric; and whether out-of-region calibration predicts which signal acquires best.
4. Post-hoc recalibration, fit in-region only. Value is narrow:
   a robustness check on whether an out-of-region calibration
   gap survives a standard in-region correction, not a
   standalone finding. Reporting only, never feeds the deferral
   rule, keeps the two results independent. ⚠️ Trap: the
   calibration set must come from in-region data, calibrating
   on out-of-region points assumes access to exactly the labels
   the premise says are unavailable.
