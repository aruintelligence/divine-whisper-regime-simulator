# Divine Whisper Regime Simulator

A deterministic Python model of four initial conditions, a hand-written phase classifier, and a routing policy. Created by Daniel Jacob Read IV and Shane Travis Horman for ĀRU Intelligence's Inward Physics research. This is a symbolic simulation: its variables and phase labels are not measurements of cognition, consciousness, or physical phase transitions.

## Run

Python 3.8 or newer is sufficient for the JSON and CSV outputs; no packages are required.

```bash
python divine_whisper_regime_simulator.py --out-dir results --compare-boost
```

This writes `results/summary.json`, `results/trajectories.csv`, and `results/boost_comparison.json`. The CSV records the state, the classifier's **resulting phase**, and the **policy selected before the step**. Repeating a run with the same arguments yields the same model values. To render an optional PNG, install Matplotlib and add `--plot`:

```bash
python -m pip install matplotlib
python divine_whisper_regime_simulator.py --out-dir results --plot
```

## What the model computes

The six state variables are attention A, distortion D, coherence C, instability F, remembrance fidelity λ, and salience S. Each step uses fixed differential update rules and clamps the state to specified numerical ranges. The clarity proxy is

```math
\Psi = \frac{\lambda C S}{(D+\epsilon)^2(F+\epsilon)}\left[1+b\max\left(0,\frac{T-D}{T}\right)^2\right],
```

where ε is a stabilizer, T is the **boost threshold** (default 0.16), and b is the boost (default 1.8). The bracketed factor is **continuous** at T. The word “breakthrough” is a categorical label from a separate classifier requiring D below a distinct **phase threshold** (default 0.14), C > 1.15, λ > 0.95, F < 0.55, S > 1, and Ψ > 12. These values are chosen model constants, not discovered critical points.

Four names identify **initial conditions**: `distracted`, `normal`, `focused`, and `high_start`. They do not promise final phases. By default, `--policy feedback` applies the classifier's current phase to choose the next routing control. `--policy scripted` reproduces the older case-anchored policy, including its case-specific shocks. Comparing them separates routing chosen from state from routing imposed by the case label.

## A falsifiable baseline

At 220 steps, dt=0.05, feedback policy, T=0.16, phase threshold=0.14, and b=1.8, **none** of the four cases enters the breakthrough phase. The `high_start` case ends focused; its distortion ends near 0.1933, above both thresholds. Setting b to zero yields identical trajectories under this baseline because the boost condition is never reached. A high Ψ value alone does not satisfy the phase gate.

Try a threshold sensitivity check with the same initial states and policy:

```bash
python divine_whisper_regime_simulator.py --boost-threshold 0.20 --compare-boost --out-dir results-boost-020
python divine_whisper_regime_simulator.py --boost-threshold 0.20 --phase-threshold 0.20 --compare-boost --out-dir results-both-020
```

Changing only the boost threshold to 0.20 slightly raises `high_start`'s maximum Ψ (about 49,690) but still produces **no** breakthrough label. Also changing the phase threshold to 0.20 makes `high_start` first receive that label at step 78. With b=1.8 its maximum Ψ is about 1,062,307; with b=0 it is about 561,359. Both first receive the label at step 78. The large difference comes from the chosen gate changing routing, followed by the formula's amplification; it is sensitivity to model choices, not evidence of a natural transition.

Run the regression suite with `python -m unittest discover -s tests -v`. It checks the threshold boundary, default non-activation, policy routing, reproducibility, and exported data.

## Limits

The initial conditions, coefficients, clamps, policy, and classifier were selected for an illustrative dynamical system. Large values are a consequence of a small denominator and clamped state, not evidence of insight or emergence. No external dataset or benchmark validates the model against cognition. The tool writes only to the output directory selected by the user.

See [LICENSE](LICENSE) for terms.

---

### Inward Physics research navigation

[Complete Inward Physics index](https://github.com/aruintelligence/aruintelligence/blob/main/INWARD_PHYSICS.md) · [Daniel Jacob Read IV’s ORCID](https://orcid.org/0009-0000-6133-1872) · [Zenodo DOI series (record index)](https://github.com/aruintelligence/aruintelligence/blob/main/INWARD_PHYSICS.md#zenodo-record-index)

These links index project-authored proposals and simulations; they do not imply independent validation of a physical model.
