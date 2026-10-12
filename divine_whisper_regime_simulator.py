"""Deterministic, symbolic regime simulation; variables are not empirical measures.

Clarity Psi = lambda * C * S / ((D + eps)^2 * (F + eps)), with an
optional continuous amplification below a distortion threshold.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional


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
    phase_distortion_threshold: float = 0.14
    breakthrough_boost: float = 1.80  # continuous amplification below threshold

    def validate(self) -> None:
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(self.dt) or not 0 < self.dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(self.breakthrough_distortion_threshold) or not 0 < self.breakthrough_distortion_threshold < 5:
            raise ValueError("boost threshold must be finite and in (0, 5)")
        if not math.isfinite(self.phase_distortion_threshold) or not 0 < self.phase_distortion_threshold < 5:
            raise ValueError("phase threshold must be finite and in (0, 5)")
        if not math.isfinite(self.breakthrough_boost) or self.breakthrough_boost < 0:
            raise ValueError("boost must be finite and nonnegative")


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
        self.A = min(5.0, max(0.0, self.A))
        self.D = min(5.0, max(0.01, self.D))
        self.C = min(5.0, max(0.0, self.C))
        self.F = min(5.0, max(0.05, self.F))
        self.lam = min(5.0, max(0.0, self.lam))
        self.S = min(5.0, max(0.0, self.S))


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
            s.D < self.cfg.phase_distortion_threshold
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
        if phase not in {"distracted", "normal", "focused", "breakthrough"}:
            raise ValueError(f"unknown policy phase: {phase}")
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

    # This piecewise factor is continuous at the threshold. The classifier's
    # categorical threshold is discrete; the clarity equation is not.
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


CASES = ("distracted", "normal", "focused", "high_start")


def simulate_case(case: str, cfg: SimConfig, policy: str = "feedback") -> Dict[str, List]:
    cfg.validate()
    if policy not in {"feedback", "scripted"}:
        raise ValueError("policy must be feedback or scripted")
    classifier = PhaseClassifier(cfg)
    orchestrator = Orchestrator()

    # Case names describe initial conditions, not guaranteed final regimes.
    initial_map = {
        "distracted":   State(A=0.18, D=1.55, C=0.25, F=1.45, lam=0.28, S=0.30),
        "normal":       State(A=0.55, D=0.88, C=0.55, F=0.92, lam=0.52, S=0.55),
        "focused":      State(A=0.95, D=0.48, C=0.82, F=0.72, lam=0.72, S=0.78),
        "high_start":   State(A=1.08, D=0.32, C=0.92, F=0.65, lam=0.78, S=0.84),
    }
    if case not in initial_map:
        raise ValueError(f"unknown initial condition: {case}")
    s = initial_map[case]
    s.psi = clarity(s, cfg)

    hist: Dict[str, List] = {k: [] for k in ["A", "D", "C", "F", "lam", "S", "psi", "phase", "policy_phase"]}

    for t in range(cfg.steps):
        inferred_phase = classifier.classify(s)
        if policy == "feedback":
            # The classifier decides routing from the current state.
            active_phase, shock = inferred_phase, 0.0
        elif case == "distracted":
            active_phase = "distracted" if t < int(cfg.steps * 0.85) else inferred_phase
            shock = 0.05 if t % 27 in (0, 1, 2) else 0.0
        elif case == "normal":
            active_phase, shock = "normal", 0.01 if t % 55 == 0 and t > 0 else 0.0
        elif case == "focused":
            active_phase, shock = "focused", 0.0
        else:
            active_phase = "focused" if s.D > cfg.breakthrough_distortion_threshold else "breakthrough"
            shock = 0.0

        ctrl = orchestrator.policy(active_phase)
        s = step_dynamics(s, ctrl, cfg, shock=shock)

        for k in ("A", "D", "C", "F", "lam", "S", "psi"):
            hist[k].append(getattr(s, k))
        hist["phase"].append(classifier.classify(s))
        hist["policy_phase"].append(active_phase)

    return hist


# ----------------------------
# Plotting and export
# ----------------------------

def plot_trajectories(results: Dict[str, Dict[str, List]], cfg: SimConfig, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib: python -m pip install matplotlib") from exc
    x = [(i + 1) * cfg.dt for i in range(cfg.steps)]

    plt.figure(figsize=(11, 6))
    for label, hist in results.items():
        plt.plot(x, hist["psi"], linewidth=2, label=label.replace("_", " ").title())
    plt.xlabel("Simulation time (arbitrary units)")
    plt.ylabel("Modeled clarity Ψ")
    plt.title("Symbolic clarity trajectories by initial condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def summarize(results: Dict[str, Dict[str, List]], cfg: SimConfig, policy: str) -> dict:
    cases = {}
    for label, hist in results.items():
        phases = hist["phase"]
        first = next((i + 1 for i, phase in enumerate(phases) if phase == "breakthrough"), None)
        cases[label] = {
            "final_psi": hist["psi"][-1],
            "max_psi": max(hist["psi"]),
            "final_phase": phases[-1],
            "first_breakthrough_step": first,
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
            "policy_counts": {
                p: hist["policy_phase"].count(p) for p in sorted(set(hist["policy_phase"]))
            },
        }
    return {
        "scope": "symbolic deterministic model; not a measure of cognition",
        "policy": policy,
        "config": asdict(cfg),
        "cases": cases,
    }


def export_csv(results: Dict[str, Dict[str, List]], cfg: SimConfig, out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        keys = ("A", "D", "C", "F", "lam", "S", "psi", "phase", "policy_phase")
        writer.writerow(("case", "step", "time", *keys))
        for case, hist in results.items():
            for i in range(cfg.steps):
                writer.writerow((case, i + 1, (i + 1) * cfg.dt, *(hist[key][i] for key in keys)))


def run(cfg: SimConfig, policy: str = "feedback") -> Dict[str, Dict[str, List]]:
    return {case: simulate_case(case, cfg, policy) for case in CASES}


def boost_comparison(cfg: SimConfig, policy: str, original: dict) -> dict:
    without = summarize(run(replace(cfg, breakthrough_boost=0), policy), replace(cfg, breakthrough_boost=0), policy)
    return {
        "intervention": "Set only breakthrough_boost to zero; same initial conditions and policy",
        "cases": {
            case: {
                "max_psi_with_boost": data["max_psi"],
                "max_psi_without_boost": without["cases"][case]["max_psi"],
                "first_breakthrough_step_with_boost": data["first_breakthrough_step"],
                "first_breakthrough_step_without_boost": without["cases"][case]["first_breakthrough_step"],
            }
            for case, data in original["cases"].items()
        }
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--boost", type=float, default=1.8)
    parser.add_argument("--boost-threshold", type=float, default=0.16, help="distortion threshold for clarity amplification")
    parser.add_argument("--phase-threshold", type=float, default=0.14, help="separate distortion gate for breakthrough label")
    parser.add_argument("--policy", choices=("feedback", "scripted"), default="feedback")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--plot", action="store_true", help="write PNG; requires matplotlib")
    parser.add_argument("--compare-boost", action="store_true", help="write a paired boost=0 ablation")
    args = parser.parse_args(argv)
    cfg = SimConfig(steps=args.steps, dt=args.dt, breakthrough_boost=args.boost,
                    breakthrough_distortion_threshold=args.boost_threshold,
                    phase_distortion_threshold=args.phase_threshold)
    try:
        cfg.validate()
    except ValueError as exc:
        parser.error(str(exc))

    results = run(cfg, args.policy)
    summary = summarize(results, cfg, args.policy)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    export_csv(results, cfg, args.out_dir / "trajectories.csv")
    if args.compare_boost:
        comparison = boost_comparison(cfg, args.policy, summary)
        (args.out_dir / "boost_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        try:
            plot_trajectories(results, cfg, args.out_dir / "trajectories.png")
        except RuntimeError as exc:
            parser.error(str(exc))

    print(f"=== Symbolic regime simulation ({args.policy} policy) ===")
    for label, data in summary["cases"].items():
        print(
            f"{label:>12}: final Ψ={data['final_psi']:8.3f} | "
            f"max Ψ={data['max_psi']:8.3f} | "
            f"final phase={data['final_phase']} | "
            f"first breakthrough step={data['first_breakthrough_step']}"
        )
    print(f"Results: {args.out_dir}")


if __name__ == "__main__":
    main()
