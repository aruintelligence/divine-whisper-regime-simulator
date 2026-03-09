from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import textwrap

out = Path("/mnt/data/divine_whisper_phase_sim")
out.mkdir(parents=True, exist_ok=True)

code = r'''
"""
Divine Whisper Cognitive Regime Simulator

What this prototype includes
1) Phase classifier
2) Orchestrator routing policy
3) Simulation loop that plots four trajectories

Regimes
- distracted
- normal
- focused
- breakthrough

Main clarity equation:
    Psi(t) = [lambda(t) * C(t) * S(t)] / [ (D(t) + eps)^2 * (F(t) + eps) ]

This is a compact research-style prototype intended to be easy to modify.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List
import json
import math

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------

@dataclass
class SimConfig:
    steps: int = 220
    dt: float = 0.05
    eps: float = 1e-3

    # Dynamics
    gamma_A: float = 0.70
    rho_A: float = 0.25

    kappa: float = 1.10
    xi1: float = 0.60
    xi2: float = 0.45
    xi3: float = 0.55

    eta1: float = 0.90
    eta2: float = 0.55
    mu_C: float = 0.40
    eta3: float = 0.60

    phi1: float = 0.45
    phi2: float = 0.35
    phi3: float = 0.65
    phi4: float = 0.75

    c1: float = 0.55
    c2: float = 0.50
    c3: float = 0.35

    s1: float = 0.65
    s2: float = 0.22
    s3: float = 0.50
    s4: float = 0.70

    # Breakthrough support
    breakthrough_distortion_threshold: float = 0.16
    breakthrough_boost: float = 1.80  # nonlinear lift once D is very low


# ----------------------------
# State
# ----------------------------

@dataclass
class State:
    A: float          # attention
    D: float          # distortion
    C: float          # coherence
    F: float          # free energy / instability
    lam: float        # remembrance fidelity
    S: float          # signal salience
    psi: float = 0.0  # clarity

    def clamp(self) -> None:
        self.A = float(np.clip(self.A, 0.0, 5.0))
        self.D = float(np.clip(self.D, 0.01, 5.0))
        self.C = float(np.clip(self.C, 0.0, 5.0))
        self.F = float(np.clip(self.F, 0.05, 5.0))
        self.lam = float(np.clip(self.lam, 0.0, 5.0))
        self.S = float(np.clip(self.S, 0.0, 5.0))
        self.psi = float(np.clip(self.psi, 0.0, 500.0))


# ----------------------------
# Phase classifier
# ----------------------------

class PhaseClassifier:
    """
    Hybrid phase classifier:
    - primarily uses state geometry (D, C, F, lam, S)
    - also sanity-checks with Psi
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def classify(self, s: State) -> str:
        # Hard breakthrough gate first
        if (
            s.D < 0.14
            and s.C > 1.15
            and s.lam > 0.95
            and s.F < 0.55
            and s.S > 1.00
            and s.psi > 12.0
        ):
            return "breakthrough"

        # Focused regime
        if s.D < 0.40 and s.C > 0.85 and s.S > 0.80 and s.F < 0.95:
            return "focused"

        # Distracted regime
        if s.D > 0.95 or (s.C < 0.45 and s.F > 1.05):
            return "distracted"

        # Otherwise default to normal
        return "normal"


# ----------------------------
# Orchestrator
# ----------------------------

class Orchestrator:
    """
    Returns routing actions / control adjustments based on the current phase.
    Think of this as the controller policy for the agent graph.
    """

    def policy(self, phase: str) -> Dict[str, float]:
        if phase == "distracted":
            return {
                "U": 1.20,           # stronger attention input
                "Reg": 0.25,         # minimal coherence support
                "L": 0.20,           # learning / stabilization
                "SA": 0.35,          # self-alignment
                "Rm": 0.15,          # memory registration
                "Align": 0.20,       # alignment
                "Rel": 0.35,         # relevance filtering
                "Nov": 0.05,         # novelty low
                "Xa": 1.05,          # external distraction high
                "MW": 0.95,          # mind wandering high
                "Conflict": 0.90,    # conflict high
                "Ep": 1.00,          # prediction error high
                "Mm": 0.85,          # memory mismatch high
                "Noise": 1.00,
            }

        if phase == "normal":
            return {
                "U": 0.95,
                "Reg": 0.45,
                "L": 0.45,
                "SA": 0.50,
                "Rm": 0.40,
                "Align": 0.45,
                "Rel": 0.55,
                "Nov": 0.12,
                "Xa": 0.55,
                "MW": 0.45,
                "Conflict": 0.40,
                "Ep": 0.55,
                "Mm": 0.50,
                "Noise": 0.50,
            }

        if phase == "focused":
            return {
                "U": 1.00,
                "Reg": 0.80,
                "L": 0.85,
                "SA": 0.90,
                "Rm": 0.75,
                "Align": 0.85,
                "Rel": 0.85,
                "Nov": 0.15,
                "Xa": 0.20,
                "MW": 0.18,
                "Conflict": 0.16,
                "Ep": 0.22,
                "Mm": 0.20,
                "Noise": 0.18,
            }

        # breakthrough
        return {
            "U": 0.88,      # not maximal; preserve rather than force
            "Reg": 1.10,
            "L": 1.00,
            "SA": 1.10,
            "Rm": 0.95,
            "Align": 1.10,
            "Rel": 1.10,
            "Nov": 0.08,
            "Xa": 0.06,
            "MW": 0.05,
            "Conflict": 0.04,
            "Ep": 0.08,
            "Mm": 0.06,
            "Noise": 0.05,
        }


# ----------------------------
# Simulation engine
# ----------------------------

def clarity(s: State, cfg: SimConfig) -> float:
    base = (s.lam * s.C * s.S) / ((s.D + cfg.eps) ** 2 * (s.F + cfg.eps))

    # Nonlinear lift once distortion collapses below threshold.
    # This creates a genuine phase-transition-like jump.
    if s.D < cfg.breakthrough_distortion_threshold:
        collapse_factor = 1.0 + cfg.breakthrough_boost * (
            (cfg.breakthrough_distortion_threshold - s.D) / cfg.breakthrough_distortion_threshold
        ) ** 2
        base *= collapse_factor

    return float(base)


def step_dynamics(
    s: State,
    ctrl: Dict[str, float],
    cfg: SimConfig,
    shock: float = 0.0,
) -> State:
    dA = ctrl["U"] - cfg.gamma_A * s.A - cfg.rho_A * s.D
    dD = -cfg.kappa * s.A * s.D + cfg.xi1 * ctrl["Xa"] + cfg.xi2 * ctrl["MW"] + cfg.xi3 * ctrl["Conflict"] + shock
    dC = cfg.eta1 * s.A + cfg.eta2 * ctrl["Reg"] - cfg.mu_C * s.C - cfg.eta3 * s.D
    dF = cfg.phi1 * ctrl["Ep"] + cfg.phi2 * ctrl["Mm"] - cfg.phi3 * ctrl["L"] - cfg.phi4 * ctrl["SA"]
    dlam = cfg.c1 * ctrl["Rm"] + cfg.c2 * ctrl["Align"] - cfg.c3 * (0.30 + ctrl["Noise"])
    dS = cfg.s1 * ctrl["Rel"] + cfg.s2 * ctrl["Nov"] + cfg.s3 * s.C - cfg.s4 * ctrl["Noise"]

    s2 = State(
        A=s.A + cfg.dt * dA,
        D=s.D + cfg.dt * dD,
        C=s.C + cfg.dt * dC,
        F=s.F + cfg.dt * dF,
        lam=s.lam + cfg.dt * dlam,
        S=s.S + cfg.dt * dS,
    )
    s2.clamp()
    s2.psi = clarity(s2, cfg)
    return s2


def simulate_case(case: str, cfg: SimConfig) -> Dict[str, List[float]]:
    classifier = PhaseClassifier(cfg)
    orchestrator = Orchestrator()

    # Distinct initial conditions chosen to make the four trajectories visibly separate.
    initial_map = {
        "distracted":   State(A=0.18, D=1.55, C=0.25, F=1.45, lam=0.28, S=0.30),
        "normal":       State(A=0.55, D=0.88, C=0.55, F=0.92, lam=0.52, S=0.55),
        "focused":      State(A=0.95, D=0.48, C=0.82, F=0.72, lam=0.72, S=0.78),
        "breakthrough": State(A=1.08, D=0.32, C=0.92, F=0.65, lam=0.78, S=0.84),
    }
    s = initial_map[case]
    s.psi = clarity(s, cfg)

    hist: Dict[str, List[float]] = {k: [] for k in ["A", "D", "C", "F", "lam", "S", "psi"]}
    phases: List[str] = []

    # Starting policy anchored to the intended case for trajectory separation.
    base_phase = case

    for t in range(cfg.steps):
        # Blend classified phase with intended trajectory label to create
        # stable, interpretable trajectories for each regime.
        inferred_phase = classifier.classify(s)

        # Simple anchoring logic:
        # - distracted stays conservative
        # - normal gradually improves but not too far
        # - focused sustains strong routing
        # - breakthrough receives preservation routing after crossing threshold
        if case == "distracted":
            active_phase = "distracted" if t < int(cfg.steps * 0.85) else inferred_phase
            shock = 0.05 if (t % 27 in (0, 1, 2)) else 0.0

        elif case == "normal":
            active_phase = "normal"
            shock = 0.01 if (t % 55 == 0 and t > 0) else 0.0

        elif case == "focused":
            active_phase = "focused"
            shock = 0.0

        else:  # breakthrough trajectory
            # Allow focused buildup, then preserve breakthrough if reached.
            active_phase = "focused" if s.D > cfg.breakthrough_distortion_threshold else "breakthrough"
            shock = 0.0

        ctrl = orchestrator.policy(active_phase)
        s = step_dynamics(s, ctrl, cfg, shock=shock)

        # Record after update
        for k in hist.keys():
            hist[k].append(getattr(s, k))
        phases.append(classifier.classify(s))

    hist["phase"] = phases
    return hist


# ----------------------------
# Plotting and export
# ----------------------------

def plot_trajectories(results: Dict[str, Dict[str, List[float]]], cfg: SimConfig, out_png: str) -> None:
    x = np.arange(cfg.steps) * cfg.dt

    plt.figure(figsize=(11, 6))
    for label, hist in results.items():
        plt.plot(x, hist["psi"], linewidth=2, label=label.title())
    plt.xlabel("Time")
    plt.ylabel("Clarity Ψ(t)")
    plt.title("Divine Whisper Cognitive Regimes: Clarity Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def export_summary(results: Dict[str, Dict[str, List[float]]], out_json: str) -> None:
    summary = {}
    for label, hist in results.items():
        phases = hist["phase"]
        summary[label] = {
            "final_psi": hist["psi"][-1],
            "max_psi": max(hist["psi"]),
            "final_state": {
                "A": hist["A"][-1],
                "D": hist["D"][-1],
                "C": hist["C"][-1],
                "F": hist["F"][-1],
                "lam": hist["lam"][-1],
                "S": hist["S"][-1],
            },
            "phase_counts": {
                p: phases.count(p) for p in sorted(set(phases))
            },
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    cfg = SimConfig()
    results = {label: simulate_case(label, cfg) for label in ["distracted", "normal", "focused", "breakthrough"]}

    plot_trajectories(results, cfg, "divine_whisper_regimes.png")
    export_summary(results, "divine_whisper_regimes_summary.json")

    # Also print a compact console summary.
    print("=== Divine Whisper Regime Summary ===")
    for label, hist in results.items():
        print(
            f"{label:>12}: final Ψ={hist['psi'][-1]:8.3f} | "
            f"max Ψ={max(hist['psi']):8.3f} | "
            f"final phase={hist['phase'][-1]}"
        )


if __name__ == "__main__":
    main()
'''

py_path = out / "divine_whisper_regime_simulator.py"
py_path.write_text(code, encoding="utf-8")

# Execute the generated script to produce outputs
import subprocess, json, os, textwrap, sys
result = subprocess.run(
    ["python", str(py_path)],
    cwd=str(out),
    capture_output=True,
    text=True,
    check=True
)

print(result.stdout)

# Show the generated chart inline
from IPython.display import Image, display
display(Image(filename=str(out / "divine_whisper_regimes.png")))

print(f"Saved Python script: {py_path}")
print(f"Saved plot: {out / 'divine_whisper_regimes.png'}")
print(f"Saved summary JSON: {out / 'divine_whisper_regimes_summary.json'}")
