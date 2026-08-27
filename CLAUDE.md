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

**Gate 3.5 passed with a caveat (2026-08-27), `scripts/gate_3_5_split_axis.py`.** Aspect ratio predicted from the 80 coefficients: ridge 0.784, gradient boosting 0.988, MLP 0.983, against a std of 1.639. **Aspect ratio is input-measurable, so it is safe as the split axis** and splitting on it is a shift in the questions, not in the answers.

It missed the pre-registered 0.99 bar. That bar came from the information floor implied by dropping R(0,0) per 1.5, which bounds what any model could know and says nothing about what a quick untuned fit reaches on 21k points. Everything else points one way: the 0.05% trim moved R² by 0.00002, residual correlation with R(0,0) was +0.075, and every increase in training budget raised the score, so the shortfall is budget rather than missing information. The miss is recorded rather than the bar relaxed, see decision log 3.5.

Incidental, worth keeping: error tracks data density, lowest around A 9.9 to 10.1 and highest in the sparse upper tail. The extrapolation premise shows up in a purely geometric quantity before any surrogate exists.

**Stage 2 gate passed (2026-08-27), `scripts/stage_2_noise_floor.py`.** One kNN index (k=10, brute force, z-scored coefficients) answered the residual-spread floor (2.3) and the tolerance duplicate check (2.2). **Aleatoric sits at the numerical floor for both candidate targets**, confirming the 2.1 prediction. Floor intercepts −0.000003 (edge rotational transform, bar 0.0012) and 0.001834 (log10 qi, bar 0.0102). The 28 pairs closer than distance 0.197 have a median target difference of **exactly 0.000000 for both targets**, which is determinism measured rather than asserted.

⚠️ **Bin placement flipped a verdict.** With uniform deciles, log10 qi read ABOVE FLOOR at 0.021. The bottom decile spanned distance 0 to 1.53 while genuine near-twins live below 0.5, so the intercept was extrapolated from bins centred at 1.1 to 2.4. Rebinning with quantile edges packed into the left tail dropped it 10x and flipped the verdict. Recorded in decision log 2.3, not hidden.

⚠️ **High-dimensional caveat for the write-up:** median nearest-neighbour distance is 3.84 in z-scored 80-dimensional space. Genuine near-twins barely exist outside the bottom percentile, which caps what this check can prove.

**Next: the mirror-pair search** (folded into 2.2, answers the 1.7 leftover, nothing downstream depends on it), then `splits.py` and the day 1-2 grid.

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

**Aleatoric will be near zero, and that is the correct answer.** The inputs are fully observed (no truncation), the solver is deterministic, a fixed convergence tolerance produces a deterministic high-frequency function rather than noise, and the four generation pathways shift the distribution over x without changing y given x. Anyone arguing that the mixture of pathways creates aleatoric noise is confusing covariate shift with label noise.

**Therefore: manufacture aleatoric deliberately.** Hide the high mode-number coefficients to create a known noise floor, then check whether the variance head recovers the right magnitude. (Hiding field period is no longer a viable alternative, it's fixed at NFP=3 across the whole dataset, so there's nothing left to hide there.) This converts the decomposition from an assertion into a validated measurement, and it is the reason the mean-variance head is not decorative.

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
- Recipe frozen (architecture, epochs, stopping rule) before the N-sweep, so "more data" is not confounded with "more optimization."
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
  Run the whole sweep in **both input conditions**, all 80
  coefficients and high mode-numbers hidden. Without the second
  condition two of the three checks are vacuous, "aleatoric
  stays flat as N grows" says nothing when aleatoric is already
  at the numerical floor. The hidden condition restores the
  contrast: epistemic decaying toward a nonzero, N-invariant
  aleatoric floor, and lets the variance head be checked against
  a known injected magnitude.
  Budget: 5 N x 3 seeds x 2 input conditions = 30 ensembles,
  300 member networks. Any expansion past that is a visible
  decision, not a drift.
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
- Outlier trimming: match A.4's 0.05% per-metric tail trim,
  applied to the target metric only, never to the split axis.
- Ensemble size: 10, matching A.4.
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
- Calibration-under-shift uses epistemic alone, since isolating
  model ignorance from local fitting difficulty is the whole
  point of the region split.
- Deferral curve does not commit to one ranking signal. Show
  epistemic-ranked and total-ranked as two separate "mine"
  curves, alongside the shared random floor and oracle ceiling.
  Which signal defers best is a result, not a setup detail, and
  showing both is nearly free (same trained ensemble, same
  predictions, just two sort orders) while directly testing the
  "total and epistemic should be close" claim empirically
  instead of assuming it.
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

Highest priority first. Full reasoning is in `constellaration-uq-decisions.md` in this repo.

1. **Split axis.** Leaning aspect ratio, direction not
   pre-committed. Both candidate axes (aspect ratio, max
   elongation) and all three cuts (tail-low, tail-high,
   interior hole) get checked together in the day 1-2 baseline
   check, held-out histogram thickness plus in/out error gap.
   Whichever (axis, direction) pair shows the strongest real
   gap wins, but only if that axis also clears the coverage
   threshold in both a mid-range band and a tail band, since
   the main figure needs the hole and the tail results both. If aspect ratio and max elongation tie, prefer aspect
   ratio for its tighter link to the paper's own compactness
   narrative (Section 3.3). If nothing shows a gap, reconsider
   PCA direction or high mode-number spectral energy before
   falling back further. (Field period and generation pathway
   are no longer usable fallbacks, both are constant columns
   once filtered to NFP=3, DESC/VMEC-optimized only.)
1a. **Calibration diagnostic set.** Undecided until the day 1-2
   histogram is seen. What's settled is the principle and
   priority order, not a pick: diagnostics that reuse the
   existing distance bins are preferred over ones that need
   their own. CRPS costs nothing extra and is the most likely
   floor. If there's room past it, coverage-versus-nominal is
   next, not because it's next-cheapest, but because it's most
   directly tied to making the deferral threshold defensible.
   PIT and reliability diagrams are last, they need their own
   binning on top of the distance bins. Final set decided once
   the histogram shows what's actually affordable.
2. **Primary target.** Leaning `edge_rotational_transform_over_n_field_periods` (tidy range, no heavy tail, requires the solve, appears as a constraint in all three benchmarks, not defined as a max or min over a surface so no kinks). Secondary target `log10(qi)` if time allows. Both this choice and the secondary-target call get real evidence from the day 1-2 grid check, not just Table 7 priors.
3. **Loss.** MSE warm-up plus variance floor, β-NLL in reserve.

---

## Traps to watch for

- Trimming outliers on the split axis (silently destroys the test set).
- Filtering bad rows on `has_neurips_2025_forward_model_error` alone (misses generation-stage failures; see the null-row trap above).
- Canonicalizing the `z_sin` sign (destroys the handedness signal and breaks comparability with `_to_X`). Note the reason: **not** because it would create opposite labels, which is the retracted argument above.
- Recalibrating on out-of-region data (silently invalidates the premise).
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

**End of week 2:** three splits run, calibration figures exist, deferral curve drafted. If not met, drop the second target and all optional scope.

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
