# Audit findings, 2026-08-30

Three adversarial audits run against the repository by a separate model with no
access to the working session: data and leakage, uncertainty mathematics, and
experimental design and claims. Each was told to treat `CLAUDE.md` and
`constellaration-uq-decisions.md` as claims under test rather than as ground
truth, and to verify against code and `results/*.json`.

**Overall:** every table recomputed from the artifacts reproduced to the printed
digit. The measurement is sound. The risk sits almost entirely in the reporting,
where prose compressed curves into single numbers the bins contradict.

**Nothing here is fixed yet.** This is the work list for the next session.

---

## A. The one real methodological issue

### 1. The target trim reads held-out labels and deletes the tail's hardest test rows

`data.py:48-52`, called before the split in every script.

`trim_target_tails` computes its 0.05% quantiles over the whole pool and drops
rows by their labels. **22 of the 28 dropped rows land inside the tail's
held-out region**, because target and split axis are correlated, which is this
project's own premise.

⚠️ **The magnitude is far larger than a first estimate suggested.** The 28 rows
are two distinct populations, not one smooth tail:

- **13 rows at z ≈ −9**, targets −0.45 to −0.56. These are **sign-flipped
  configurations**, negative edge rotational transform, the class recorded as
  133 of 27,022. A different physical regime, not distribution outliers.
- **15 rows at z ≈ +3**, targets 0.48 to 0.52. Ordinary upper tail.

Of the 22 inside the tail's region, **9 are sign-flipped**. A model trained on a
distribution centred at 0.241 predicting a truth of −0.46 errs by about 0.65,
roughly **16x** the tail's RMSE of 0.0396, not the 3x first assumed. Restoring
all 22 moves out-of-region RMSE from about 0.0396 to about 0.047, so the 3.24x
headline becomes roughly 3.9x and the calibration ratio worsens past 0.66.

**So the leak flatters us, by about 20% on the headline rather than 2%.**

**But restoring them would introduce a worse confound.** Those 9 rows fail
because the model has barely seen a sign-flipped configuration anywhere in
training, not because they sit at low aspect ratio. Including them conflates
"extrapolating along the aspect-ratio axis is hard" with "a class covering 0.5%
of the pool is unlearnable". Both are true; only the first is this project's
question.

**Fix.** Keep the A.4-comparable trimmed set as the headline and add a
**sensitivity result**, not a caveat. Retrain the tail ensemble (seeds are fixed,
so it reproduces exactly) evaluating additionally on the untrimmed
out-of-region set, then report:

> Restoring the 22 trimmed rows to the tail's held-out set raises out-of-region
> RMSE from 0.0396 to X and the ratio from 3.24 to Y. Nine of the 22 carry
> negative edge rotational transform, a sign class representing 0.5% of the
> pool, so the increase measures unfamiliarity with that class rather than
> distance along the split axis. The headline is reported on the trimmed set for
> comparability with Appendix A.4, and is conservative as a result.

**Why it works.** It converts an undisclosed leak into a quantified disclosure
running against our own interest, explains why the trimmed number is still the
right headline, and keeps A.4 comparability. Cost is about 7 minutes on a T4.

---

## B. Numbers the artifacts contradict

### 2. The furthest-bin PIT of 0.223 does not exist. The real value is 0.153

No script ever computed per-bin PIT: decision log 7.1 cut PIT binned by
distance. The number was written into both documents anyway.

Recomputed per-bin tail PIT means: **0.429, 0.422, 0.412, 0.421, 0.393, 0.346,
0.211, 0.153**.

**Fix.** Add `pit_mean` to the calibration bin rows, rerun `scripts/calibration.py`
(seconds), correct both documents.

**Why it works.** ⚠️ The 7.1 cut applies to the per-bin PIT *histogram*, where 10
bars rest on about 60 points each. A per-bin PIT *mean* rests on all 675 and is
stable. Those are different objects and were conflated. This makes the number
reproducible from an artifact and gains the bias-versus-distance curve, 0.429
falling to 0.153, which the project does not currently report at all.

### 3. "The tail costs about 15% more at both 0.2 and 0.4 std" is not supported

The premium is about **3% at 0.2 std and 27% at 0.4 std**, roughly **12% over
the whole shared range**.

The 15% came from interpolating two binned curves at their bin medians. The bins
have very different widths: the tail's second bin spans 0.13 to 0.31 while the
hole's are about 0.05 wide there. RMSE in a wide bin is dominated by its far
edge, so reading it at its median inflates the near value.

Recomputed in shared windows with matched within-window medians:
`[0.15, 0.25)` gives 1.025 (n = 389 vs 1,161), `[0.35, 0.45)` gives 1.271,
`[0, 0.45)` gives 1.121.

**Fix.** Add a matched-window comparison to `scripts/distance_error.py` so the
number is computed rather than hand-derived, and reword.

**Why it works.** The corrected claim is stronger: **up close a gap and an edge
cost the same, and the edge penalty emerges with distance.** The flat number was
hiding that. The load-bearing conclusion is untouched: hole RMSE over the shared
range is 0.02615 against the tail's 0.02932, versus 0.04157 for the full tail,
so most of the raw 3.02-versus-1.80 gap is still distance rather than edge.

⚠️ The coverage version of the same comparison is robust: matched window
`[0.35, 0.45)` gives hole 0.801 against tail 0.762, consistent with the claimed
0.81 against 0.78.

### 4. Small stale numbers

- "sizes stay matched at 5,404 each": the tail's out set is **5,405**
  (`tail_split` includes the boundary row at `axis <= cutoff`).
- Target std quoted as 0.0786; every results file records **0.078924**. The
  derived percentages still hold.
- `scripts/calibration.py:3-4,19` quotes pre-aggregation-fix values (0.93 to
  1.05, 0.56 to 0.62, epistemic 0.00768). Its own output says otherwise.

---

## C. Sentences that overclaim their own tables

### 5. "Coverage degrades monotonically with distance"

Tail bins: 0.928, 0.873, 0.781, **0.822**, 0.803, 0.766, 0.675, 0.587. The hole
rises first too. Downward trend, not monotone. Say "falls from 0.93 to 0.59 with
one non-monotone step".

### 6. "The irreducible part sits at 0.020 and does not move"

The recovered column **converges from above**: 0.0318, 0.0241, 0.0223, 0.0223,
0.0208. Every one of the 15 cells sits above 0.020.

**State the mechanism rather than softening the sentence.** `sqrt(noisy² −
clean²)` credits the *clean* run's misfit to the noisy run, but noisy training
fits worse, so the subtraction over-attributes and every value lands above σ.
Both misfits shrink with N, so the excess shrinks too. That makes the
convergence evidence rather than noise. The pre-registered form, "the noisy
curve flattens near sqrt(misfit² + σ²)", was already correct; the "does not
move" gloss was written over it afterwards.

### 7. "Total beats epistemic at every rate and in both regions"

48 of 49 interior rates out-of-region (one loss by 0.000012) and 48 of 49
in-region (one loss by 0.000007). The AUC lean and the "consistent lean, not a
finding" framing both survive. "At every rate" is literally false.

### 8. Verdict verbs disagree with the artifacts

`results/gate_3_5_split_axis.json` records `verdict: "INCONCLUSIVE"`; CLAUDE.md
says "passed with a caveat". `results/mse_ensemble.json` records
`verdict: "FAIL"`; the doc says "MISSED by 0.2%".

Both misses are disclosed and neither bar was moved, which is to the project's
credit. Align the verbs with the artifacts and keep the stated reasons for
proceeding. **"The bar failed and here is why we proceeded" is more credible
than a softened verb.**

### 9. Step 1's epistemic 0.00631 is not comparable to any later number

`scripts/mse_ensemble.py:132,160` computes `epistemic.mean()`, the mean of
per-point standard deviations, which is exactly the aggregation removed
everywhere else on 2026-08-30. "0.00631 is the baseline that has to grow"
compares across three simultaneous changes: aggregation, loss (MSE against NLL)
and training size (21,118 against 16,794). Nothing downstream uses it, since the
hole and tail comparisons use step 2's own 0.00887. Mark it non-comparable, or
update the script and rerun.

### 10. Single-seed disclosure

**Everything except the N-sweep is seed 0**: all six calibration ratios, all
coverage and PIT figures, both deliverable curves, the matched-distance
comparison. The docs flag this for the deferral margin and the placement U but
not for the headline numbers.

One plain sentence beside the headline table, rather than scattered caveats.
Stated once it retires the objection; discovered, it reads as an omission.

### 11. The N-sweep pre-registration cannot be verified from the repository

`git log -S` finds the prediction text entering only in commit `830792a`, the
same commit that added the sweep results. The sweep ran 15:15 to 15:53 against
`029f05f` with a dirty tree. Nothing contradicts the claim and nothing supports
it: prediction and outcome are indistinguishable in the record.

The predictions did land (~0.023 predicted against 0.0236 measured at full N).
Note the attestation honestly, and **from now on commit predictions in their own
commit before running.**

### 12. Decision 4.7's "500 validation points confirmed sufficient at N=1000"

`results/hp_check.json` shows that row ran at **lr 3e-4**, the nominal
validation-loss winner. The frozen recipe is lr 1e-3, chosen afterwards on cost.
A higher learning rate gives a noisier validation curve, so the frozen recipe's
stopping signal at N=1,000 was never actually checked.

**Reword to cite the N-sweep instead**, which ran three seeds at N=1,000 under
the frozen recipe with epistemic at 0.01477 / 0.01486 / 0.01462, a spread of
about ±1%. That is better evidence under the recipe actually used.

---

## D. Latent code hazards, none of which has fired

### 13. The variance clamp zeroes the gradient in both directions

`nets.py:208-210`. `raw_log_variance.clamp(...)` has zero derivative outside its
bounds. The floor correctly blocks the pull down, but it also blocks the pull
up: **a pinned point cannot un-pin itself through its own NLL gradient**, only
through shared-weight updates from other points. A head that collapses early in
the NLL phase would stay collapsed and aleatoric would silently read
`1e-6 · y_std²` everywhere.

Never fired: `pinned_fraction` is 0.0 in all three runs. The monitor at
`mv_ensemble.py:190-195` is the only thing standing between this and a silent
wrong number. Record that in a comment beside it.

### 14. Warm-up checkpoint escape

`nets.py:274-306`. The best-weight reset fires at `epoch == warmup_epochs`. If
the loop never reaches it (`max_epochs <= warmup_epochs`, reachable through the
`**train_kwargs` chain at `ensemble.py:95-97`), `best_state` is the best
MSE-phase checkpoint and `predict` returns variances from an entirely untrained
head, silently. Unreachable with the frozen constants (500 > 25).

Add `if max_epochs <= warmup_epochs: raise`, plus a test.

### 15. NaN targets vanish silently in the trim

`data.py:52`. `between()` is False for NaN, so a NaN-target row disappears in the
trim with no count discrepancy attributable to it. Zero NaNs in the current pool,
so latent only. Add an explicit guard.

### 16. Two comment inaccuracies

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

1. Item 1 while the GPU is up
2. Items 2 and 3, which change results files
3. Everything in B, C and D as one documentation pass
4. E

**Cut for the pitch, keep in an appendix:** coverage-versus-nominal, the
placement table, and the three gate figures. They answer "did you check?" rather
than "what did you find?"
