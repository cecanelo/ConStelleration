# Where can you trust a stellarator surrogate?

**Short version:** I trained a fast model to replace a slower physics simulation,
then measured where its answers stop being trustworthy. It turns out the model's
error bars are least trustworthy in exactly the place it is most wrong. But the
model is still good at telling you *which* predictions to double-check, and that
is the useful part in this project.

---

## What the project is about

A **stellarator** is a magnetic confinement fusion device. Its confining field is
produced entirely by external coils, twisted into a three-dimensional geometry
that holds the plasma in place without driving a current through it.

The object being designed is the **plasma boundary**, the outer surface the
confined plasma is held to. In this dataset a boundary is parameterised by
**80 Fourier coefficients**.

To find out whether a shape is good, engineers run an equilibrium solver,
VMEC++, and compute performance metrics from the field it produces. That solve
is the expensive step: this dataset alone is the product of roughly 182,000
candidate boundaries evaluated that way.

So people train a **surrogate**: a neural network that looks at the 80 numbers
and predicts the simulation's answer directly. However, it inherits one catch, true of
supervised models generally:

> A model is accurate on inputs resembling its training data and degrades outside
> it. A point prediction gives no indication of which case you are in, so the
> output looks the same whether the input was well covered or not.

This is the covariate-shift failure mode, and it is silent: nothing in the
prediction itself flags the problem. The standard response is to have the model
report an uncertainty alongside each prediction. **Whether that uncertainty can
be trusted in the region where it matters most (the compact end) is what this project measures.**

---

## How the uncertainty is built

The model is an ensemble of 10 networks. Each one with two
outputs (a mean and a variance) for its prediction, indexed here as $\mu_k(x)$
and $\sigma_k^2(x)$ for network $k$.

- **Ensemble mean**: the average of the 10 predicted means, the number actually
  reported as the prediction.
  $$\bar\mu(x) = \frac{1}{K}\sum_{k=1}^{K} \mu_k(x)$$

- **Epistemic uncertainty**: how much the 10 means disagree with each other.
  Wide agreement means confidence, wide disagreement means the training data
  left room for different networks to learn different things. Shrinks with
  more data.
  $$\sigma^2_{\text{epi}}(x) = \frac{1}{K-1}\sum_{k=1}^{K} \big(\mu_k(x) - \bar\mu(x)\big)^2$$

- **Aleatoric uncertainty**: the average of the 10 networks' own reported
  variances, each network's estimate of the error it expects on top of its
  best guess.
  $$\sigma^2_{\text{ale}}(x) = \frac{1}{K}\sum_{k=1}^{K} \sigma_k^2(x)$$

- **Total uncertainty**: epistemic plus aleatoric, the number reported as the
  model's error bar.
  $$\sigma^2_{\text{total}}(x) = \sigma^2_{\text{epi}}(x) + \sigma^2_{\text{ale}}(x)$$

**Aleatoric is not literally noise here.** This dataset has none: two distinct
but near-identical shapes get simulation results that agree to six decimal places.
What the term actually measures is the network's own misfit, the part of the
function it cannot fit exactly, which looks like noise from inside the model.

---

## Project findings

**1. The model underestimates its own error outside the training region.**

On the least familiar shapes, the actual error roughly triples. The error bar
only roughly doubles, so the model notices it is on shakier ground, just not
by enough.

Concretely: the model offers a range it says will contain the true answer 90% of
the time. On familiar shapes it does. On the least familiar shapes tested, that
same "90%" range contains the truth only 59% of the time, and the range itself
gives no sign that it has shrunk to mean less than it claims.

**2. Post-hoc recalibration makes it worse.**

If the error bars are miscalibrated, the standard repair is to measure by how
much and correct for it: check how far off the ranges are, then stretch or
shrink every future range by that same factor. The catch is that you can only
measure "how far off" using data you already have, which means the familiar
shapes. On those, the model turned out to be a little too cautious, so the fix
correctly narrowed its ranges.

Applied to unfamiliar shapes, that same correction made things worse. The "90%"
range, which already held the truth only 78% of the time across the whole
unfamiliar set, dropped to 67% after the fix. A correction learned on familiar
territory does not carry over: the model is too cautious near home and too
confident far away, and one number cannot fix both directions at once.

This matters because it rules out the easy answer. There is no cheap
post-processing trick that fixes the intervals here, which is part of why the
project turns to a different remedy below.

**3. Ranking by uncertainty is still useful, even though the ranges are not correct.**

On unfamiliar shapes, where the ranges themselves cannot be trusted as absolute
numbers, they are still right about the *order*. The predictions the model flags 
as least certain tend to be the worst ones.

So you can use it as a filter. Send the most uncertain 20% of shapes to the
simulation and keep the fast model's answer for the other 80%.

| approach | error reduction |
|---|---|
| check the 20% the model flags | **41%** |
| check a random 20% | 20% |

Put another way: checking designs at random, you would need to send half of
them, 50 out of 100, to cut the error in half, that is just what "no signal"
looks like. With the model's ranking, checking the worst 27 out of 100 gets you
there instead.

**Findings 1-2 and Finding 3 together are the real result: wrong as a number,
still right as a ranking.**
Reporting only the failure would make the model sound less useful than it is.
Reporting only the ranking would hide that its error bars cannot be read
literally.

The practical payoff: the vertical axis is average prediction error, so the
curve reads directly. Pick how many simulations you are willing to run, and read
off how much error remains.

![The deferral curve](figures/deferral_curve_mae.png)

**The two grey lines are the reference points that make the other two
readable.**

- **Deferred at random**, the dashed line, is the same budget spent blindly.
  Send 20% of shapes to the solver chosen by coin flip and you remove about 20%
  of the error, which is what "no useful information" looks like.
- **Perfect ranking**, the dotted line, is the best any ordering could possibly
  do: it assumes you already knew which predictions were wrong and sent exactly
  those. Nobody can have that, which is why it is a ceiling rather than a target.

The model's curve sits roughly two thirds of the way from the blind line to the
perfect one. That fraction is the honest summary of how much the uncertainty
signal is worth: clearly far better than nothing, and clearly not clairvoyant.

---

## How the failure was measured

The tricky part is defining "unfamiliar". You cannot just test on random held-out
shapes, because those are all familiar by construction. So I built **three
different train/test splits** and trained the same model on each.

![Where the cuts fall](figures/split_design.png)

- **Random split.** Ordinary train/test. Every test shape looks like the training
  shapes. This is the **control**: if this split shows a problem, the method is
  broken.
- **Tail.** Hold back one whole end of the range: the most compact shapes. The
  model has never seen anything like them, and there is no data on the far side.
  Can it step outside what it knows?
- **Interior hole.** Cut a band out and hold it back, but choose one with
  training data on *both sides* of it. Can it fill in a gap?

The hole is placed directly next to the tail's band, not in the middle of the
data. Two reasons, and neither is aesthetic. The two bands must not overlap, or
the experiments would share shapes and could not be compared. And a gap in the
dense middle is too easy: the model is never more than a short step from
familiar data, so there is almost nothing to measure. Sitting next to the tail
puts the hole in the thinnest data an interior gap can reach, which is what makes
the comparison against the tail a fair one.

Splitting this way lets me separate two failures that are usually lumped
together. **"Cannot fill in a gap" and "cannot go past the edge" need different
responses.** A gap can be filled by running a few more simulations in that spot.
Going past the edge cannot be fixed that way at all.

### How the error and the error bars behave with distance

![Error against distance](figures/distance_error.png)

The horizontal axis is how far a test shape sits from the nearest training shape.
The vertical axis is how wrong the model was. Both conditions get worse with
distance, and the tail is worse than the hole.

![Coverage against distance](figures/coverage_distance.png)

The same horizontal axis, now asking whether the model's error bars kept up. The
dotted line is what an honest model would do. It falls away instead.

| split | error outside vs inside | are the error bars honest? |
|---|---|---|
| random (control) | 1.0x, no penalty | yes |
| interior hole | 1.9x worse | no, too small by about 1/4 |
| tail (compact end) | 3.3x worse | no, too small by about 1/3 |

The control behaving correctly is what makes the other two rows believable.

---

## Two more results

**The model is not just imprecise at the compact end, it is biased.** Its
predictions are systematically too high there, not merely too uncertain. That is
a different problem, and making the error bars wider would not fix it. The
interior hole shows no such bias anywhere in its range, which separates the two
conditions a second time.

**More data does not solve this.** Across five training-set sizes, from 1,000
shapes up to about 17,000, performance on familiar shapes improved a lot, by 67%.
Performance on unfamiliar shapes improved only 29%, so the gap between them got
*wider*, not narrower.

The careful version of that claim: the extra data was drawn from the region the
model already knew, and more of a region cannot teach it about a region it
excludes. So the answer is not "collect more of the same". It is either sample
deliberately in the sparse region, or use deferral.

---

## Why this particular question

The team behind the dataset, Proxima Fusion, published it with a baseline
surrogate of their own ([arXiv 2506.19583](https://arxiv.org/abs/2506.19583),
NeurIPS 2025). In their appendix they report how accurate it is on familiar
shapes, then note that surrogates like this go wrong when asked about unfamiliar
ones, and list uncertainty calibration as something nobody has done yet.

This project does that.

**A deliberately narrow claim:** as of August 2026, a search across four channels
turned up no published work studying uncertainty calibration or extrapolation
reliability of surrogates on this dataset. About fourteen papers use the dataset,
and the closest one uses uncertainty as machinery inside an optimizer rather than
studying it. A broader claim than that would be easy to knock down, so it is not
made here.

**The premise is not just my own reading.** Two independent 2026 papers describe
the compact end of this design space as thinly sampled and are actively
generating new shapes there. That is precisely the region where a confidently
wrong surrogate does the most damage.

---

## Does the model actually work?

Before claiming anything about uncertainty, the underlying model has to be
credible. Compared against the published baseline on the same metric, mine is
about 1.75x their error. That is the expected gap between an afternoon of tuning
and a fully tuned published model, and a plain bug would look like 5x, not 1.75x.

Three honest limits on that comparison:

- Only the raw error is comparable, not R-squared, because R-squared depends on
  how spread out the test set is and ours differs from theirs.
- Their paper does not say enough about their architecture to claim I reproduced
  it. I matched the metric, not the model.
- Applying every filter their published code performs gives me 27,050 rows where
  they report about 23,000. Two intermediate counts match their paper exactly, so
  my pipeline is right and their appendix is incomplete somewhere. I document the
  discrepancy rather than inventing a filter to force the number down.

---

## What is in this repository

```
src/constellaration_uq/   the reusable pieces: data loading, splits, model, metrics
scripts/                  one script per experiment, each writing to results/
results/                  56 committed data files, so you can check any number
figures/                  14 figures, PNG and PDF
notebooks/figures.ipynb   every figure and table, with the reasoning beside it
tests/                    automated tests for the parts that do real logic
```

Two documents carry the thinking rather than the code:

- **[`constellaration-uq-decisions.md`](constellaration-uq-decisions.md)** is a
  decision log. Every choice, why it was made, and what would change it, including
  the ones I got wrong and corrected.
- **[`audit-findings.md`](audit-findings.md)** records three adversarial reviews
  run by a separate model instructed to treat my own documentation as claims under
  test, and what each finding cost to fix.

**All results and figures are committed on purpose**, so nothing has to be run to
check the numbers.

---

## Terms you will meet in the code and the decision log

None of these are needed to read this page, where each idea is explained
where it appears. They are here because the notebook and the decision log
use the field's vocabulary throughout.

| term | plain meaning |
|---|---|
| **surrogate** | fast model standing in for a slow simulation |
| **calibrated** | the error bar matches how wrong the model actually is |
| **coverage** | how often the truth really falls inside the predicted range |
| **in / out of region** | test shapes similar to training data, versus unfamiliar ones |
| **deferral** | handing the uncertain cases back to the slow simulation |
| **aspect ratio** | how fat or skinny the doughnut is. Low means compact |
| **PIT** | where the true answer landed inside the predicted range. Reveals whether the model is biased, not just imprecise |
| **CRPS** | a score for the whole predicted range, not just the single best guess. Used in the notebook version of the deferral curve |

---

## Running it

Use the Python environment that has the package installed, not a bare `python3`.

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

python3 scripts/mv_ensemble.py random
python3 scripts/mv_ensemble.py hole
python3 scripts/mv_ensemble.py tail

python3 scripts/calibration.py
python3 scripts/deferral.py
python3 scripts/recalibration.py
```

Each of the three models takes about 8 minutes on a laptop CPU. Everything after
that reads their saved output and finishes in seconds.

Runs are deterministic, so rerunning with the same seed reproduces the committed
files exactly. That is the regression check: if a saved file changes after a
cleanup that was supposed to change nothing, then it changed something.

The raw dataset is 585 MB and not stored in git:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='proxima-fusion/constellaration',
    repo_type='dataset',
    allow_patterns=['data/*'],
    local_dir='data_raw',
)
```

---

## Limitations

Stated up front, because they are the first thing a careful reader will look for.

- **The model never saw failures.** About 24,000 shapes crashed the simulation and
  were excluded, so it only ever learned from shapes that worked.
- **The data was not designed as an experiment.** It is wherever the original
  optimizers happened to converge. That imbalance is what makes this study
  possible and is also a limit on it.
- **One machine configuration and one dataset.** Everything here is at three field
  periods, matching Proxima's own example filter.
- **One target metric.** All uncertainty results are for one physics quantity. A
  second one also degrades at the compact end, so the basic premise is not
  specific to my choice, but the size and direction of the effect are.
- **Shapes only, no coils.** Whether a shape could actually be built is a separate
  question this does not touch.
- **Deferral assumes the simulation answers, and sometimes it does not.** In the
  region deferral sends work to, roughly a third of simulations fail, which would
  turn the 41% improvement into about 28%.

  The tempting conclusion is that compact shapes break the solver. **That is
  wrong, and the check that shows why is the useful part.** Every recorded
  failure comes from one particular shape-generating pipeline, and within that
  pipeline compactness barely matters. Sorting by compactness was accidentally
  sorting by *which program proposed the shape*. So solver reliability is a
  property of the proposal process, not of the geometry, which is also the more
  useful conclusion because it says what to change.

---

## What I would do next

1. **Active learning.** Use the uncertainty to choose which shapes to simulate
   next, and see whether it beats choosing at random. Cut here for compute cost,
   not for lack of interest.
2. **A second target metric**, to test whether "where the model can be trusted" is
   a property of the design space or of the specific quantity being predicted.
3. **Other uncertainty methods**, Gaussian processes and Laplace approximations,
   deliberately excluded so this result stands on the simpler method alone.
4. **A corrected deferral curve** that accounts for simulations failing, instead of
   reporting that as a caveat.

---

## References

- Proxima Fusion, *ConStellaration: A dataset of QI-like stellarator plasma
  boundaries and optimization benchmarks*, [arXiv
  2506.19583](https://arxiv.org/abs/2506.19583), NeurIPS 2025.
- Lakshminarayanan et al., *Simple and scalable predictive uncertainty estimation
  using deep ensembles*, NeurIPS 2017.
- Seitzer et al., *On the pitfalls of heteroscedastic uncertainty estimation with
  probabilistic neural networks*, ICLR 2022.
- Kuleshov et al., *Accurate uncertainties for deep learning using calibrated
  regression*, ICML 2018.
