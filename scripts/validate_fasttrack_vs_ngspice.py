from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.circuits import Circuit
from src.data.schema import ComponentSpec
from src.data.spice_runner import run_ac_analysis_with_ngspice
from src.physics import FastTrackEngine


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    scenario: str
    z0: float
    fc_hz: Optional[float]
    freq_hz: np.ndarray
    components: List[ComponentSpec]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Fast Track vs NGSPICE on dataset circuits.")
    p.add_argument("--data", required=True, type=Path, help="Path to dataset jsonl.")
    p.add_argument("--num", type=int, default=100, help="Number of circuits to evaluate.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument(
        "--components",
        choices=["discrete", "ideal"],
        default="discrete",
        help="Which component set to simulate.",
    )
    p.add_argument(
        "--ensure-scenarios",
        type=str,
        default="general,anti_jamming,coexistence,wideband_rejection,random_basic",
        help="Comma-separated scenarios to include if available.",
    )
    p.add_argument(
        "--min-per-scenario",
        type=int,
        default=1,
        help="Minimum count to include per scenario when available.",
    )
    p.add_argument("--q", type=float, default=None, help="Q value for L/C (default: None, disable Q).")
    p.add_argument("--q-l", type=float, default=None, help="Override Q for inductors.")
    p.add_argument("--q-c", type=float, default=None, help="Override Q for capacitors.")
    p.add_argument(
        "--q-model",
        type=str,
        choices=["none", "freq_dependent", "fixed_ref"],
        default="freq_dependent",
        help="Q modeling mode. 'none' disables Q; freq_dependent is high fidelity.",
    )
    p.add_argument("--max-plots", type=int, default=1, help="Max number of overlay plots to save.")
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to save overlay plots (optional).",
    )
    p.add_argument(
        "--plot-mode",
        choices=["worst", "random"],
        default="worst",
        help="Which samples to plot when plot-dir is set.",
    )
    p.add_argument("--dump", type=Path, help="Optional JSONL dump of per-circuit errors.")
    return p.parse_args()


def _component_from_dict(d: dict) -> ComponentSpec:
    return ComponentSpec(
        ctype=str(d["ctype"]),
        role=str(d["role"]),
        value_si=float(d["value_si"]),
        std_label=d.get("std_label"),
        node1=str(d["node1"]),
        node2=str(d["node2"]),
    )


def _load_samples(path: Path, *, components_key: str) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            freq = row.get("freq_hz")
            comps = row.get(components_key)
            if not freq or not comps:
                continue
            try:
                freq_hz = np.asarray(freq, dtype=float).reshape(-1)
            except Exception:
                continue
            if freq_hz.size == 0:
                continue
            comp_list = [_component_from_dict(c) for c in comps if c]
            if not comp_list:
                continue
            scenario = row.get("scenario") or "general"
            sample_id = str(row.get("sample_id") or row.get("circuit_id") or row.get("spec_id") or "unknown")
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    scenario=str(scenario),
                    z0=float(row.get("z0", 50.0)),
                    fc_hz=row.get("fc_hz"),
                    freq_hz=freq_hz,
                    components=comp_list,
                )
            )
    return records


def _ref_freq_hz(freq_hz: np.ndarray, fc_hz: Optional[float]) -> float:
    if fc_hz is not None and np.isfinite(fc_hz):
        return float(fc_hz)
    f_min = float(np.min(freq_hz))
    f_max = float(np.max(freq_hz))
    return float(np.sqrt(f_min * f_max))


def _select_samples(
    records: List[SampleRecord],
    *,
    num: int,
    ensure_scenarios: Sequence[str],
    min_per_scenario: int,
    seed: int,
) -> List[SampleRecord]:
    rng = random.Random(seed)
    remaining = list(records)
    selected: List[SampleRecord] = []
    for scenario in ensure_scenarios:
        for _ in range(int(min_per_scenario)):
            idxs = [i for i, r in enumerate(remaining) if r.scenario == scenario]
            if not idxs:
                break
            pick = rng.choice(idxs)
            selected.append(remaining.pop(pick))
    if len(selected) < int(num):
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, int(num) - len(selected))])
    return selected[: int(num)]


def _maybe_warn_q_mismatch(q_model: str, q_l: float | None, q_c: float | None) -> None:
    if q_model == "freq_dependent" and (q_l is not None or q_c is not None):
        print("[warn] NGSPICE uses fixed_ref Q; expect mismatch when q-model=freq_dependent.")


def _summarize(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "max": float(np.max(arr))}


def _save_plots(results: List[dict], *, plot_dir: Path, mode: str, max_plots: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not available; skipping plots.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    if not results or max_plots <= 0:
        return
    if mode == "worst":
        ranked = sorted(results, key=lambda r: float(r["max_err_db"]), reverse=True)
    else:
        ranked = list(results)
        random.shuffle(ranked)
    for row in ranked[: int(max_plots)]:
        freq = row.get("freq_hz")
        s21_fast = row.get("s21_fast_db")
        s21_spice = row.get("s21_spice_db")
        if freq is None or s21_fast is None or s21_spice is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(freq, s21_fast, label="Fast Track", color="tab:blue")
        ax.plot(freq, s21_spice, label="NGSPICE", color="tab:orange", linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("S21 (dB)")
        ax.set_title(f"sample={row.get('sample_id')} scenario={row.get('scenario')}")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend()
        out_path = plot_dir / f"fasttrack_vs_ngspice_{row.get('sample_id')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_path = args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    components_key = "discrete_components" if args.components == "discrete" else "ideal_components"
    records = _load_samples(data_path, components_key=components_key)
    if not records:
        raise RuntimeError(f"No valid samples found in {data_path}")

    ensure_scenarios = [s.strip() for s in str(args.ensure_scenarios).split(",") if s.strip()]
    selected = _select_samples(
        records,
        num=int(args.num),
        ensure_scenarios=ensure_scenarios,
        min_per_scenario=int(args.min_per_scenario),
        seed=int(args.seed),
    )
    if not selected:
        raise RuntimeError("No samples selected for evaluation.")

    if args.q_model == "none":
        q_l = None
        q_c = None
        q_model_ft = "freq_dependent"
    else:
        q_l = args.q if args.q_l is None else args.q_l
        q_c = args.q if args.q_c is None else args.q_c
        q_model_ft = str(args.q_model)
    _maybe_warn_q_mismatch(str(args.q_model), q_l, q_c)

    engine: Optional[FastTrackEngine] = None
    results: List[dict] = []
    skipped = 0
    all_errs: List[float] = []
    per_mae: List[float] = []
    per_max: List[float] = []

    for rec in selected:
        if engine is None or float(engine.z0) != float(rec.z0):
            engine = FastTrackEngine(z0=float(rec.z0), device="cpu", dtype=torch.float64)
        ref_freq_hz = _ref_freq_hz(rec.freq_hz, rec.fc_hz) if q_model_ft == "fixed_ref" and (q_l is not None or q_c is not None) else None
        s21_fast = engine.simulate_s21_db(
            rec.components,
            rec.freq_hz,
            q_L=q_l,
            q_C=q_c,
            q_model=q_model_ft,
            ref_freq_hz=ref_freq_hz,
        )
        circuit = Circuit(rec.components, z0=rec.z0, in_port=("in", "gnd"), out_port=("out", "gnd"))
        try:
            s21_spice, _ = run_ac_analysis_with_ngspice(
                circuit,
                rec.freq_hz,
                float(rec.z0),
                q_L=q_l,
                q_C=q_c,
                ref_freq_hz=_ref_freq_hz(rec.freq_hz, rec.fc_hz),
            )
        except RuntimeError:
            skipped += 1
            continue

        err = np.abs(s21_fast - s21_spice)
        mae = float(np.mean(err))
        maxe = float(np.max(err))
        per_mae.append(mae)
        per_max.append(maxe)
        all_errs.extend(err.tolist())
        row = {
            "sample_id": rec.sample_id,
            "scenario": rec.scenario,
            "mae_db": mae,
            "max_err_db": maxe,
        }
        if args.plot_dir is not None:
            row.update(
                {
                    "freq_hz": rec.freq_hz,
                    "s21_fast_db": s21_fast,
                    "s21_spice_db": s21_spice,
                }
            )
        results.append(row)

    print(f"Selected: {len(selected)}")
    print(f"Simulated: {len(results)} (skipped={skipped})")
    mae_stats = _summarize(per_mae)
    max_stats = _summarize(per_max)
    global_mae = float(np.mean(all_errs)) if all_errs else float("nan")
    global_max = float(np.max(all_errs)) if all_errs else float("nan")
    print(f"MAE(dB):   mean={mae_stats['mean']:.6g} median={mae_stats['median']:.6g} max={mae_stats['max']:.6g}")
    print(f"MaxErr(dB): mean={max_stats['mean']:.6g} median={max_stats['median']:.6g} max={max_stats['max']:.6g}")
    print(f"Global MAE(dB): {global_mae:.6g}")
    print(f"Global MaxErr(dB): {global_max:.6g}")

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as f:
            for row in results:
                out = {
                    "sample_id": row["sample_id"],
                    "scenario": row["scenario"],
                    "mae_db": float(row["mae_db"]),
                    "max_err_db": float(row["max_err_db"]),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"Wrote per-circuit metrics to {args.dump}")

    if args.plot_dir is not None:
        _save_plots(results, plot_dir=args.plot_dir, mode=str(args.plot_mode), max_plots=int(args.max_plots))


if __name__ == "__main__":
    main()
