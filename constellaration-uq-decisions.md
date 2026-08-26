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

**1.5 Input vector** `SETTLED`

**Decided:** The 80 coefficients as the paper defines them, with major radius fixed at 1.

**Why:** Poloidal and toroidal mode numbers capped at 4, stellarator symmetry zeroing certain entries, R(0,0) fixed at 1. The paper states 80 degrees of freedom explicitly. Confirm your loader returns exactly 80 columns before going further; if it does not, you have misread the symmetry conditions.

---

**1.6 Failed rows** `SETTLED`

**Decided:** Load `default` unfiltered to get the ~24k flagged failures. Use them only for the survivorship figure, never for training.

**Why:** `misc.has_neurips_2025_forward_model_error` is a real column, so survivorship is measurable rather than speculative. Watch for nulls: a null almost certainly means "this check does not apply to this generation pathway," not "no error." Also check whether `boundary.r_cos` is populated for flagged rows, since some may have failed at generation rather than simulation.

---

## Stage 2. Noise floor

*Your stated priority 1. Target: one afternoon.*

---

**2.1 Pre-register the prediction** `SETTLED`

**Decided:** Write down before looking: aleatoric will sit at the numerical floor.

**Why:** Inputs are fully observed (no truncation), the solver is deterministic, fixed tolerance produces a deterministic high-frequency function rather than noise, and the four generation pathways shift the distribution over x without changing y given x. Predicting this and confirming it is a result. Hunting for noise that is not there is a wasted afternoon.

---

**2.2 Duplicate detection tolerance** `OPEN`

- [ ] Decided

**Options:** round coefficients to N decimals and group / exact match only / nearest-neighbour distance threshold

**Leaning:** start coarse, tighten until groups appear

**Why:** You are looking for identical inputs with different targets. Also check whether `plasma_config_id` repeats across rows, which would be the trivial explanation for anything you find.

**My decision:**
>

---

**2.3 kNN residual spread setup** `SETTLED`

- [x] Decided

**Decided:** Distance on standardized (z-scored) coefficients. k = 5 to 10, not tuned further.

**Why:** Coefficient magnitudes decay hard with mode number, so raw Euclidean distance is dominated by the low-order modes and your "neighbours" are only neighbours in a few dimensions. k is not worth optimising, this is a coarse sanity check, not a figure.

**Note:** unlike 3.11, full 80-dimensional distance is the right tool here. This check asks a general question about local consistency across the whole dataset, so it has no reason to be tied to one axis.

**My decision:**
> Standardize first, otherwise "nearest neighbour" just means "similar in the few low-order modes that happen to be biggest." k anywhere from 5 to 10 is fine, this is a sanity check and I am not going to tune it.

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

**2.5 GATE: does the data behave as expected?** `GATE`

- [ ] Passed

**Stop if:** duplicates exist with large target spread, or kNN residual spread in dense regions is far above the numerical floor.

**Then:** find out why before building anything. Something about the data is not what you think it is, and every downstream number would inherit that misunderstanding.

---

## Stage 3. Target and split

---

**3.1 Primary target** `OPEN`

- [ ] Decided

**Options:** `edge_rotational_transform_over_n_field_periods` / `vacuum_well` / `log10(qi)` / other

**Leaning:** edge rotational transform over field periods

**Why:** Passes all four filters: requires the equilibrium solve (so deferral has stakes), tidy range with no heavy tail, appears as a constraint in all three benchmark problems, and is not defined as a max or min over a surface (so no kinks). It is the low-risk choice for week one, and swapping the target later is one line.

**Blocks:** everything downstream

**My decision:**
>

---

**3.2 Secondary target** `OPEN`

- [ ] Decided (defer this until end of week 2)

**Options:** `log10(qi)` / none

**Leaning:** log10(qi) if time allows

**Why:** It is the metric the dataset is named around, it spans decades so relative error is the natural cost, and A.4 reports `log_10_qi` so you have a comparison number. Reporting two targets lets you say something comparative: epistemic dominates on the smooth metric, the variance head earns its keep on the rough one.

**My decision:**
>

---

**3.3 Target transform and scaling** `SETTLED`

**Decided:** Log10 for qi. Z-score the target using training-set statistics only.

**Why:** qi spans more than two orders of magnitude and is strictly positive. A.4 did the same. Training-set statistics only, or you have leaked test information into the scaling.

---

**3.4 Split axis** `OPEN`

- [ ] Decided

**Options:** aspect ratio / max elongation / high mode-number spectral energy / PCA direction

**Leaning:** aspect ratio

**Why:** Physically legible, an explicit constraint in all three benchmarks, and measurable from the inputs alone so it is a shift in the questions rather than in the answers. Never split on the target, which would confound covariate shift with label shift.

**External corroboration (found during the 0.5 search, 2026-08-26).** The premise that the compact end is undersampled is no longer just our reading of how the data was generated. Two independent papers treat it as established: the domain-adaptive latent diffusion work (arXiv 2608.16938) states it targets "the sparsely sampled low-aspect-ratio regime," and "Data-Driven Generation of Compact Quasi-Isodynamic Stellarators" (2026) works the same region. Both are generative rather than UQ work, so they do not compete, they corroborate. Useful in two places: it makes the split-axis justification citable rather than inferred, and it makes the deferral curve more pointed, since researchers are actively generating candidates in exactly the region where a confidently-wrong surrogate is most dangerous. **Caveat:** both work at NFP=4 while we are at NFP=3, so this supports the shape of the design space, not our specific slice. Say it that way.

**Options removed:** generation pathway is a constant column after 1.3, field period after 1.2. Neither is a usable fallback any more.

**How it gets decided:** not from reasoning, from the day 1-2 grid. Aspect ratio and max elongation are checked together, both directions each, against both candidate targets. Winner is whichever (axis, direction) pair shows the strongest real in/out error gap AND clears the coverage threshold in both a mid-range band and a tail band, since the main figure needs the hole and the tail results both. If aspect ratio and max elongation tie, take aspect ratio for its tighter link to the paper's compactness narrative (Section 3.3). If nothing shows a gap, reconsider PCA direction or high mode-number spectral energy before going further.

**Blocks:** 3.5, 3.6, 3.7

**My decision:**
>

---

**3.5 GATE: is the split axis really a function of the inputs?** `GATE`

- [ ] Passed

**Check:** predict `metrics.aspect_ratio` from the raw coefficients. It should come out near machine precision.

**If yes:** the leakage objection is dead and you can say so in one sentence.
**If no:** you have misunderstood how aspect ratio is defined here. Resolve before continuing, or pick a different axis.

---

**3.6 Split direction** `OPEN`

- [ ] Decided

**Options:** hold out low aspect ratio (compact) / hold out high aspect ratio

**Leaning:** low, but not pre-committed

**Why:** Compact devices are the commercially interesting frontier, and the paper's Section 3.3 makes the compactness versus coil complexity trade-off explicit. But Table 6 shows aspect ratio targets were drawn uniformly on 4 to 12 while A was an upper-bound constraint during optimization, so achieved values will not be uniform.

**Updated:** direction is no longer decided separately from axis. Both directions for both candidate axes go into the same day 1-2 grid (see 3.4), so the direction falls out of the same evidence rather than being chosen first and checked second. The old fallback wording, "flip to the high end and change the story," is superseded: if aspect ratio fails in both directions the fallback is max elongation, not a narrative change.

**My decision:**
>

---

**3.7 Threshold and held-out fraction** `OPEN`

- [ ] Decided

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

**4.4 Loss and variance-collapse fix** `OPEN`

- [ ] Decided

**Options:** plain Gaussian NLL / MSE warm-up plus variance floor / β-NLL / warm-up plus floor, with β-NLL in reserve

**Leaning:** MSE warm-up plus variance floor, β-NLL held in reserve

**Why:** The NLL gradient on the mean is scaled by 1/σ², so points the model fits poorly get high predicted σ and are then down-weighted, which stalls learning exactly where it is most needed. Note this is not really "variance collapse" in the σ→0 sense; calling it that in an interview invites a correction. β-NLL (Seitzer et al. 2022) counteracts it directly if warm-up plus floor is not enough. If you go to β-NLL, β becomes another choice.

**My decision:**
>

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

**My decision:**
> Train the plain MSE version first. It is what A.4 actually built so it is the fair comparison, and it verifies the whole pipeline in one cheap run. After that, if the mean-variance means come out worse, I know the cause is the variance head and not the data handling.

---

**4.7 Recipe freeze** `SETTLED`

**Decided:** Architecture, epochs, and stopping rule are fixed before Stage 6 begins.

**Why:** If you train longer at larger N, or early-stop against a validation set that grows with N, you have confounded "more data" with "more optimization" and the N-sweep cannot be attributed to either.

**Blocks:** 6.1

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

**GATE: day 1-2 of week 1** `GATE`

- [ ] Passed

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
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

---

## Open questions to raise later

*Things you noticed but deliberately did not chase.*

-
-
-
