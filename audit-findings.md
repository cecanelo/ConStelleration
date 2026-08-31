# Audit findings, 2026-08-30

Three adversarial audits run against the repository by a separate model with no
access to the working session: data and leakage, uncertainty mathematics, and
experimental design and claims. Each was told to treat `CLAUDE.md` and
`constellaration-uq-decisions.md` as claims under test rather than as ground
truth, and to verify against code and `results/*.json`.

**Overall:** every table recomputed from the artifacts reproduced to the printed
digit. The measurement is sound. The risk sits almost entirely in the reporting,
where prose compressed curves into single numbers the bins contradict.

**Status: items 1 to 16 closed, all 2026-08-31. Items 17 to 19 open**, and those are remaining project work rather than audit findings: the three missing story assets, the write-up, and switching the studio back to CPU. Closed items keep their full text with the result appended, so the reasoning stays readable next to what it produced.

---

## A. The one real methodological issue

### 1. ~~The target trim reads held-out labels and deletes the tail's hardest test rows~~ ✅ CLOSED 2026-08-31

`data.py:48-52`, called before the split in every script.

`trim_target_tails` computes its 0.05% quantiles over the whole pool and drops
rows by their target value, before any split exists. Target and split axis are
correlated, which is this project's own premise, so the deleted rows do not land
evenly: **22 of 28 fall inside the tail's held-out region**, 9 of them
sign-flipped configurations carrying negative edge rotational transform.

**MEASURED, `scripts/mv_ensemble.py tail --sensitivity`.** The ensemble is
trained exactly as before, then additionally scored on the deleted rows.

| held-out set | n | out-of-region RMSE | ratio |
|---|---|---|---|
| trimmed (headline) | 5,405 | 0.03962 | 3.24 |
| + 13 ordinary tail rows | 5,418 | 0.04051 | **3.31** |
| + 9 sign-flipped rows | 5,427 | 0.04200 | **3.44** |

**The leak is real and small: the headline is conservative by about 2%, or 6%
with the sign class.** It flatters us, which is why it needed measuring rather
than assuming. No finding changes; each holds slightly more strongly.

**Resolution.** The trimmed set stays the headline. The 9 sign-flipped rows fail
because the model has barely seen a negative-target configuration anywhere in
training, not because they sit at low aspect ratio, so folding them in would
conflate "extrapolating along the split axis is hard" with "a 0.5% class is
unlearnable". Reported separately instead.

**The justification was rewritten at the same time.** Decision 1.4 now rests on two standalone
reasons rather than on matching A.4: a squared-error metric must not be decided
by 0.1% of rows, and the extreme low tail is a sign-convention artifact their
own scorer removes with `np.abs()`. The A.4 match is inferred from appendix
prose, their training code is not public, and their random split makes the trim
harmless in a way it is not here.

**Incidental result kept.** The ensemble errs by about 0.275 on the sign-flipped
rows rather than the ~0.65 it would score by predicting the pool mean, so it has
more signal for that class than its 133 rows in 27,022 would suggest.

**Verified:** `mv_ensemble_tail_points.csv` came back byte-identical, and the
sensitivity run writes to `mv_ensemble_tail_sensitivity` so it cannot clobber
the files calibration and deferral read.

## B. Numbers the artifacts contradict

### 2. ~~The furthest-bin PIT of 0.223 does not exist~~ ✅ CLOSED 2026-08-31

No script computed per-bin PIT: decision log 7.1 cut PIT binned by distance. The
number was written into both documents anyway. **The real value is 0.153.**

**Fixed** by adding `pit_mean` to the calibration bin rows. ⚠️ 7.1's cut does not
forbid this: it cut the per-bin PIT *histogram*, where ten bars rest on about
sixty points each, while a per-bin *mean* rests on all ~675. Two different
objects, conflated when the cut was written.

**The real reason it was needed, which is not the wrong number.** The claim it
supports was confounded. "The tail is biased (0.348) and the hole is not
(0.531)" compared two aggregates over different distance ranges, since the tail
reaches 2.13 std out and the hole stops at 0.45. If bias grows with distance the
whole contrast could have been a distance effect.

**Result: the contrast survives, and is stronger than the aggregates suggested.**

| | nearest bin | ~0.4 std | furthest bin |
|---|---|---|---|
| hole p30 | 0.522 | 0.527 | 0.527 |
| tail-low | 0.429 | 0.412 | **0.153** |

The hole is flat between 0.52 and 0.55 across its whole range with no trend, so
it is not biased anywhere. The tail is already at 0.429 in its nearest bin, so it
is biased everywhere out of region and worsens with distance.

### 3. ~~"The tail costs about 15% more at both 0.2 and 0.4 std"~~ ✅ CLOSED 2026-08-31

The 15% came from interpolating two binned curves at their bin medians. The bins
are equal-count, so their widths differ where the splits differ in density: the
tail's second bin spans 0.13 to 0.31 while the hole's are about 0.05 wide there,
and a wide bin's RMSE is dominated by its far edge.

**Fixed** by `distance_error.matched_windows`, which compares the two splits in
identical fixed-width windows and writes `results/distance_error_matched.csv`.

| window (std out) | n hole | n tail | RMSE hole | RMSE tail | tail / hole |
|---|---|---|---|---|---|
| 0.0 to 0.1 | 1,344 | 532 | 0.01881 | 0.02063 | 1.10 |
| 0.1 to 0.2 | 1,272 | 406 | 0.02230 | 0.02299 | 1.03 |
| 0.2 to 0.3 | 1,066 | 393 | 0.02806 | 0.03169 | 1.13 |
| 0.3 to 0.4 | 1,129 | 348 | 0.03121 | 0.03275 | 1.05 |
| 0.4 to 0.5 | 593 | 277 | 0.03305 | 0.04457 | 1.35 |
| **whole shared range** | **5,403** | **1,826** | **0.02615** | **0.02929** | **1.12** |

**Result: about 12% over the shared range.**

⚠️ **The audit's proposed replacement was also a trend, and the trend is not
there.** It suggested "~3% at 0.2 std and ~27% at 0.4, so the premium grows with
distance", and that was predicted again before this ran. Four of the five windows
sit between 1.03 and 1.13 with no ordering, and the single 1.35 comes from the
thinnest window, which also has the worst-matched medians (0.426 against 0.448)
in the range where error rises fastest. **Quote 12% over the shared range and
nothing finer.** Reading a trend off five noisy windows is the same mistake one
level down.

**Unaffected:** the load-bearing conclusion. Hole 0.02615 against tail 0.02929
over the shared range, against 0.04157 for the full tail, so most of the raw
3.02-versus-1.80 gap is distance rather than edge. Verified that the refit left
the per-split bins byte-identical.

### 4. ~~Small stale numbers~~ ✅ PARTLY CLOSED 2026-08-31

- `scripts/calibration.py` docstring corrected: 1.04 to 1.18 in region, 0.66 to
  0.69 out, epistemic 0.00887. ✅
- Still open: "sizes stay matched at 5,404 each" should be **5,405**, and target
  std quoted as 0.0786 should be **0.078924**.

---

## C. Sentences that overclaim their own tables

### 5. ~~"Coverage degrades monotonically with distance"~~ ✅ CLOSED 2026-08-31

**Fixed** in CLAUDE.md and the decision log: "falls from 0.93 to 0.59 with one non-monotone step".

Tail bins: 0.928, 0.873, 0.781, **0.822**, 0.803, 0.766, 0.675, 0.587. The hole
rises first too. Downward trend, not monotone. Say "falls from 0.93 to 0.59 with
one non-monotone step".

### 6. ~~"The irreducible part sits at 0.020 and does not move"~~ ✅ CLOSED 2026-08-31

**Fixed** in both documents and in `n_sweep.py`'s own docstring, which carried the same wording. Replaced with the convergence and its mechanism rather than a softened adjective.

The recovered column **converges from above**: 0.0318, 0.0241, 0.0223, 0.0223,
0.0208. Every one of the 15 cells sits above 0.020.

**State the mechanism rather than softening the sentence.** `sqrt(noisy² −
clean²)` credits the *clean* run's misfit to the noisy run, but noisy training
fits worse, so the subtraction over-attributes and every value lands above σ.
Both misfits shrink with N, so the excess shrinks too. That makes the
convergence evidence rather than noise. The pre-registered form, "the noisy
curve flattens near sqrt(misfit² + σ²)", was already correct; the "does not
move" gloss was written over it afterwards.

### 7. ~~"Total beats epistemic at every rate and in both regions"~~ ✅ CLOSED 2026-08-31

**Fixed:** 48 of 49 rates per region, exceptions losing by 0.000012 and 0.000007.

48 of 49 interior rates out-of-region (one loss by 0.000012) and 48 of 49
in-region (one loss by 0.000007). The AUC lean and the "consistent lean, not a
finding" framing both survive. "At every rate" is literally false.

### 8. ~~Verdict verbs disagree with the artifacts~~ ✅ CLOSED 2026-08-31

**Fixed:** gate 3.5 now reads "returned INCONCLUSIVE and we proceeded on the evidence", step 1 now cites `verdict: FAIL` from its own JSON. Reasons for proceeding kept verbatim.

`results/gate_3_5_split_axis.json` records `verdict: "INCONCLUSIVE"`; CLAUDE.md
says "passed with a caveat". `results/mse_ensemble.json` records
`verdict: "FAIL"`; the doc says "MISSED by 0.2%".

Both misses are disclosed and neither bar was moved, which is to the project's
credit. Align the verbs with the artifacts and keep the stated reasons for
proceeding. **"The bar failed and here is why we proceeded" is more credible
than a softened verb.**

### 9. ~~Step 1's epistemic 0.00631 is not comparable to any later number~~ ✅ CLOSED 2026-08-31

**Fixed:** marked non-comparable in both documents, naming all three simultaneous changes (aggregation, loss, training size), and noting that nothing downstream uses it.

`scripts/mse_ensemble.py:132,160` computes `epistemic.mean()`, the mean of
per-point standard deviations, which is exactly the aggregation removed
everywhere else on 2026-08-30. "0.00631 is the baseline that has to grow"
compares across three simultaneous changes: aggregation, loss (MSE against NLL)
and training size (21,118 against 16,794). Nothing downstream uses it, since the
hole and tail comparisons use step 2's own 0.00887. Mark it non-comparable, or
update the script and rerun.

### 10. ~~Single-seed disclosure~~ ✅ CLOSED 2026-08-31

**Fixed:** one paragraph directly above the headline table in CLAUDE.md, stating that every figure except the N-sweep is seed 0 and citing the sweep's spreads as the only direct evidence for how much seed noise to expect.

**Everything except the N-sweep is seed 0**: all six calibration ratios, all
coverage and PIT figures, both deliverable curves, the matched-distance
comparison. The docs flag this for the deferral margin and the placement U but
not for the headline numbers.

One plain sentence beside the headline table, rather than scattered caveats.
Stated once it retires the objection; discovered, it reads as an omission.

### 11. ~~The N-sweep pre-registration cannot be verified from the repository~~ ✅ CLOSED 2026-08-31

**Fixed:** the attestation is now stated as this session's record rather than the repository's, with the git evidence named. **Standing rule adopted: commit a prediction in its own commit before the run that tests it.**

`git log -S` finds the prediction text entering only in commit `830792a`, the
same commit that added the sweep results. The sweep ran 15:15 to 15:53 against
`029f05f` with a dirty tree. Nothing contradicts the claim and nothing supports
it: prediction and outcome are indistinguishable in the record.

The predictions did land (~0.023 predicted against 0.0236 measured at full N).
Note the attestation honestly, and **from now on commit predictions in their own
commit before running.**

### 12. ~~Decision 4.7's "500 validation points confirmed sufficient at N=1000"~~ ✅ CLOSED 2026-08-31

**Fixed:** the N=1000 half is withdrawn, since that row ran at lr 3e-4 rather than the frozen 1e-3. Superseded by better evidence: the N-sweep's three seeds at N=1000 under the frozen recipe, epistemic 0.01477 / 0.01486 / 0.01462.

`results/hp_check.json` shows that row ran at **lr 3e-4**, the nominal
validation-loss winner. The frozen recipe is lr 1e-3, chosen afterwards on cost.
A higher learning rate gives a noisier validation curve, so the frozen recipe's
stopping signal at N=1,000 was never actually checked.

**Reword to cite the N-sweep instead**, which ran three seeds at N=1,000 under
the frozen recipe with epistemic at 0.01477 / 0.01486 / 0.01462, a spread of
about ±1%. That is better evidence under the recipe actually used.

---

## D. Latent code hazards, none of which has fired

### 13. ~~The variance clamp zeroes the gradient in both directions~~ ✅ CLOSED 2026-08-31

**Fixed:** recorded in `gaussian_nll`'s docstring, naming the `pinned_fraction` monitor as the only thing catching it and instructing that it not be removed.

`nets.py:208-210`. `raw_log_variance.clamp(...)` has zero derivative outside its
bounds. The floor correctly blocks the pull down, but it also blocks the pull
up: **a pinned point cannot un-pin itself through its own NLL gradient**, only
through shared-weight updates from other points. A head that collapses early in
the NLL phase would stay collapsed and aleatoric would silently read
`1e-6 · y_std²` everywhere.

Never fired: `pinned_fraction` is 0.0 in all three runs. The monitor at
`mv_ensemble.py:190-195` is the only thing standing between this and a silent
wrong number. Record that in a comment beside it.

### 14. ~~Warm-up checkpoint escape~~ ✅ CLOSED 2026-08-31

**Fixed:** `train_one_mv` raises when `max_epochs <= warmup_epochs`. Two tests, one for the rejection and one bounding it (`max_epochs = warmup_epochs + 1` must still run).

`nets.py:274-306`. The best-weight reset fires at `epoch == warmup_epochs`. If
the loop never reaches it (`max_epochs <= warmup_epochs`, reachable through the
`**train_kwargs` chain at `ensemble.py:95-97`), `best_state` is the best
MSE-phase checkpoint and `predict` returns variances from an entirely untrained
head, silently. Unreachable with the frozen constants (500 > 25).

Add `if max_epochs <= warmup_epochs: raise`, plus a test.

### 15. ~~NaN targets vanish silently in the trim~~ ✅ CLOSED 2026-08-31

**Fixed:** `trim_target_tails` raises on NaN targets rather than folding them into the tail fraction. Docstring also now records that the function reads labels before any split exists, pointing at 1.4.

`data.py:52`. `between()` is False for NaN, so a NaN-target row disappears in the
trim with no count discrepancy attributable to it. Zero NaNs in the current pool,
so latent only. Add an explicit guard.

### 16. ~~Two comment inaccuracies~~ ✅ CLOSED 2026-08-31

**Fixed:** the distance-reference comment now says fit ∪ validation and why that is deliberate; `n_sweep.py`'s unused distance computation is deleted along with its now-unused import.

- `mv_ensemble.py:206-208` and `n_sweep.py:126-128` set `fit_mask` from the
  pre-validation index array, so the distance reference is fit ∪ validation while
  the comment says "the fit set". Defensible in substance, imprecise as written.
- `n_sweep.py` computes `distance` and never uses it. Dead, not wrong.

---

## E. Remaining deliverable work

### 17. Three missing story assets

Designed from the narrative rather than from the notebook:

- **Table: ours against Table 7.** RMSE only, never R², two metrics. This is what
  buys permission to make any later claim.
- **Table: the six-row decomposition.** Currently only a CLAUDE.md block. The
  ratio column is the most quotable thing in the project.
- **Figure: extrapolation gap against N**, 1.71 rising to 3.39. Data already in
  `results/n_sweep.json`, unplotted.

### 18. The write-up

### 19. Switch the studio back to CPU

Nothing remaining needs the T4 except item 1's sensitivity run.

---

## Suggested order

1. ~~Item 1~~ done 2026-08-31, on CPU in 129s
2. ~~Items 2 and 3~~ ✅ done 2026-08-31, both results files regenerated
3. ~~Everything in B, C and D~~ ✅ done 2026-08-31
4. E

**Cut for the pitch, keep in an appendix:** coverage-versus-nominal, the
placement table, and the three gate figures. They answer "did you check?" rather
than "what did you find?"
