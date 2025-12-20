from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.torch_dataset import FilterDesignDataset
from src.data.vact_codec import make_vact_syntax_prefix_allowed_tokens_fn
from src.data.dsl_codec import make_vactdsl_prefix_allowed_tokens_fn
from src.data.dsl_v2 import VALUE_SLOTS, make_dslv2_prefix_allowed_tokens_fn
from src.eval.simulate_and_score import (
    build_label_value_map,
    decode_components_from_token_ids,
    refine_component_values_to_match_s21,
    simulate_s21,
)
from src.models import VACTT5


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best-of-K evaluation with simulation verifier (VACT/VACT-DSL).")
    p.add_argument("--data", required=True, type=Path, help="Path to dataset jsonl (val/test).")
    p.add_argument("--ckpt", required=True, type=Path, help="Checkpoint dir (trainer save).")
    p.add_argument("--tokenizer", type=Path, help="Tokenizer path (defaults to --ckpt).")
    p.add_argument("--t5-name", type=str, default="t5-small", help="Base T5 model name (for raw state_dict load).")
    p.add_argument("--repr", choices=["vact", "vactdsl", "dslv2", "action"], default="vactdsl", help="Target representation to decode.")
    p.add_argument("--num", type=int, default=200, help="Number of samples to eval.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sample selection.")
    p.add_argument("--use-wave", default="real", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"])
    p.add_argument("--wave-norm", action="store_true", help="Normalize waveforms (must match training if enabled).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # generation
    p.add_argument("--kmax", type=int, default=16, help="Generate this many candidates per sample.")
    p.add_argument("--k-eval", type=str, default="1,4,16", help="Comma-separated K values for best-of-K curves.")
    p.add_argument("--do-sample", action="store_true", help="Use sampling instead of beam search.")
    p.add_argument("--num-beams", type=int, default=4, help="Beam size (used when --do-sample is off).")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling p (used when --do-sample).")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (used when --do-sample).")
    p.add_argument("--max-new", type=int, default=256, help="Max new tokens.")
    p.add_argument("--syntax-mask", action="store_true", help="Apply representation grammar mask during decoding.")
    p.add_argument("--value-mode", choices=["standard", "precision"], default="precision", help="Numeric inference mode for DSL v2 slots.")

    # simulation + metric
    p.add_argument("--sim", choices=["nodal", "abcd"], default="nodal", help="Simulator backend.")
    p.add_argument("--error", choices=["mae_lin", "rmse_lin", "mae_db", "rmse_db", "maxe_lin", "maxe_db"], default="mae_lin")
    p.add_argument("--taus", type=str, default="0.01,0.02,0.05", help="Comma-separated τ thresholds for success@τ.")

    # refinement (topology fixed)
    p.add_argument("--refine-steps", type=int, default=0, help="Few-step differentiable refinement steps (0 disables).")
    p.add_argument("--refine-top", type=int, default=1, help="Refine top-N candidates per sample.")
    p.add_argument("--refine-lr", type=float, default=5e-2, help="Refinement learning rate.")
    p.add_argument("--refine-max-ratio", type=float, default=2.0, help="Clamp value range: v in [v0/r, v0*r].")

    p.add_argument("--dump", type=Path, help="Optional JSONL dump of per-sample best candidate + score.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok_path = args.tokenizer or args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset for waveform/spec conditioning + raw targets
    ds = FilterDesignDataset(str(args.data), tokenizer, use_wave=args.use_wave, normalize_wave=args.wave_norm, use_repr="vact")
    idxs = random.sample(range(len(ds)), min(int(args.num), len(ds)))

    # model
    in_channels = ds[0]["wave"].shape[0]
    def _build_value_token_info(tok):
        vocab = tok.get_vocab()
        val_ids = []
        slot_map = {}
        slot_order = {t: i for i, t in enumerate(VALUE_SLOTS)}
        for t, tid in vocab.items():
            if t.startswith("<VAL_"):
                val_ids.append(int(tid))
                if t in slot_order:
                    slot_map[int(tid)] = int(slot_order[t])
        return val_ids, slot_map

    value_token_ids, slot_type_map = _build_value_token_info(tokenizer)

    model = VACTT5(
        t5_name=args.t5_name,
        waveform_in_channels=in_channels,
        vocab_size=len(tokenizer),
        value_token_ids=value_token_ids,
        slot_type_token_to_idx=slot_type_map,
    )
    state_path = args.ckpt / "pytorch_model.bin"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        # Be forgiving to config drift (e.g., type_vocab_size changes).
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped.append(k)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[warn] skipped {len(skipped)} mismatched keys (e.g., config drift): {skipped[:6]}")
        if missing or unexpected:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.t5.config.eos_token_id = tokenizer.eos_token_id
    model.t5.config.pad_token_id = tokenizer.pad_token_id
    model.t5.config.decoder_start_token_id = tokenizer.pad_token_id
    model.to(args.device).eval()

    label_map = build_label_value_map(tokenizer)

    taus = _parse_csv_floats(args.taus)
    k_eval = sorted(set(_parse_csv_ints(args.k_eval)))
    kmax = int(args.kmax)
    if kmax < max(k_eval):
        raise ValueError(f"--kmax ({kmax}) must be >= max(--k-eval) ({max(k_eval)})")

    prefix_allowed = None
    if args.syntax_mask:
        if args.repr == "vact":
            prefix_allowed = make_vact_syntax_prefix_allowed_tokens_fn(tokenizer)
        elif args.repr == "vactdsl":
            prefix_allowed = make_vactdsl_prefix_allowed_tokens_fn(tokenizer)
        elif args.repr == "dslv2":
            prefix_allowed = make_dslv2_prefix_allowed_tokens_fn(tokenizer)

    # metrics accumulators
    total = 0
    valid_any = 0  # at least one candidate decoded
    sim_any = 0  # at least one candidate simulated
    passive_ok_any = 0  # at least one simulated candidate is passive (within tol)
    success_counts = {(k, tau): 0 for k in k_eval for tau in taus}

    dump_rows = []

    for idx in idxs:
        total += 1
        sample = ds[idx]
        raw = ds.samples[idx]
        freq = np.asarray(raw.get("freq_hz", []), dtype=float)
        target_s21 = np.asarray(raw.get("real_s21_db") or raw.get("ideal_s21_db") or [], dtype=float)
        if freq.size == 0 or target_s21.size == 0:
            continue

        wave = sample["wave"].unsqueeze(0).to(args.device)
        scalars = sample["scalar"]
        filter_type = scalars[0:1].long().to(args.device)
        fc_hz = scalars[1:2].to(args.device)

        gen_kwargs = dict(
            wave=wave,
            filter_type=filter_type,
            fc_hz=fc_hz,
            max_new_tokens=int(args.max_new),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed,
            num_return_sequences=kmax,
        )
        if args.do_sample:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    num_beams=1,
                    top_p=float(args.top_p),
                    temperature=float(args.temperature),
                )
            )
        else:
            gen_kwargs.update(dict(do_sample=False, num_beams=max(int(args.num_beams), kmax)))

        with torch.no_grad():
            outs = model.generate(**gen_kwargs)
        seqs = outs.cpu().tolist()

        slot_values_seqs = None
        if args.repr == "dslv2":
            # Second pass: predict numeric values from decoder hidden states.
            seq_lens = [len(s) for s in seqs]
            max_len = max(seq_lens) if seq_lens else 0
            if max_len > 0:
                pad_id = int(tokenizer.pad_token_id)
                seq_tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=args.device)
                for i, s in enumerate(seqs):
                    seq_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=args.device)
                wave_rep = wave.repeat(len(seqs), 1, 1)
                filter_rep = filter_type.repeat(len(seqs))
                fc_rep = fc_hz.repeat(len(seqs))
                with torch.no_grad():
                    pred_vals = model.predict_values(wave_rep, filter_rep, fc_rep, seq_tensor, mode=args.value_mode)
                pred_vals = pred_vals.detach().cpu().tolist()
                slot_values_seqs = [pred_vals[i][: seq_lens[i]] for i in range(len(seqs))]

        # Candidate list aligned with generation order (length == kmax).
        # Each entry: (err_value, WaveformError|None, comps|None, tokens|None, passivity_violation_max|None)
        cand_records = []
        any_decoded = False
        any_simulated = False
        any_passive = False
        for i_seq, seq in enumerate(seqs):
            try:
                slot_values = slot_values_seqs[i_seq] if slot_values_seqs is not None else None
                comps, toks = decode_components_from_token_ids(
                    seq,
                    tokenizer,
                    repr_kind=args.repr,
                    label_to_value=label_map,
                    slot_values=slot_values,
                )
                if not comps:
                    cand_records.append((float("inf"), None, None, None, None))
                    continue
                any_decoded = True
                sim0 = simulate_s21(comps, freq, z0=float(raw.get("z0", 50.0)), sim_kind=args.sim)
                any_simulated = True
                from src.eval.metrics import waveform_error

                e0 = waveform_error(sim0.s21_db, target_s21, kind=args.error)
                vpass = sim0.passivity.violation_max if sim0.passivity is not None else None
                if vpass is not None and vpass <= 1e-6:
                    any_passive = True
                cand_records.append((float(e0.value), e0, comps, toks, vpass))
            except Exception:
                cand_records.append((float("inf"), None, None, None, None))

        if any_decoded:
            valid_any += 1
        if any_simulated:
            sim_any += 1
        if any_passive:
            passive_ok_any += 1

        # refine top-N candidates, then re-rank by refined error.
        if args.refine_steps and int(args.refine_steps) > 0:
            refined_records = list(cand_records)
            # pick top-N by current verifier error (excluding inf failures)
            top_n = max(0, int(args.refine_top))
            scored = [(score0, j) for j, (score0, err_obj, comps0, toks0, vpass0) in enumerate(refined_records) if np.isfinite(score0) and comps0]
            scored.sort(key=lambda x: float(x[0]))
            refine_idxs = [j for _, j in scored[:top_n]]
            for j in refine_idxs:
                score0, err_obj, comps0, toks0, vpass0 = refined_records[j]
                if comps0 is None:
                    continue
                    try:
                        comps_r = refine_component_values_to_match_s21(
                            comps0,
                            freq_hz=freq,
                            target_s21_db=target_s21,
                            z0=float(raw.get("z0", 50.0)),
                            steps=int(args.refine_steps),
                            lr=float(args.refine_lr),
                            max_ratio=float(args.refine_max_ratio),
                            device=args.device,
                        )
                        sim_r = simulate_s21(comps_r, freq, z0=float(raw.get("z0", 50.0)), sim_kind=args.sim)
                        from src.eval.metrics import waveform_error

                        err_r = waveform_error(sim_r.s21_db, target_s21, kind=args.error)
                        vpass_r = sim_r.passivity.violation_max if sim_r.passivity is not None else None
                        refined_records[j] = (float(err_r.value), err_r, comps_r, toks0, vpass_r)
                    except Exception:
                        pass
            cand_records = refined_records

        # compute best-of-K success in *generation order*
        err_values = [float(r[0]) for r in cand_records]
        best_k_val = {}
        for k in k_eval:
            kk = min(int(k), len(err_values))
            best_k_val[k] = float(np.min(err_values[:kk])) if kk > 0 else float("inf")

        for k in k_eval:
            for tau in taus:
                if np.isfinite(best_k_val[k]) and best_k_val[k] <= float(tau):
                    success_counts[(k, tau)] += 1

        # dump only the best candidate
        best_idx = int(np.argmin(err_values)) if err_values else 0
        best = cand_records[best_idx]
        if args.dump:
            dump_rows.append(
                {
                    "idx": idx,
                    "sample_id": raw.get("sample_id"),
                    "filter_type": raw.get("filter_type"),
                    "fc_hz": float(raw.get("fc_hz", 0.0)),
                    "error_kind": best[1].kind if best[1] is not None else args.error,
                    "best_error": float(best[0]),
                    "num_components": len(best[2] or []),
                    "passivity_violation_max": best[4],
                    "gen_tokens": (best[3] or [])[:200],
                }
            )

    print(f"Samples evaluated: {total}")
    print(f"Validity@Kmax:    {valid_any}/{total}")
    print(f"Simulated@Kmax:   {sim_any}/{total}")
    print(f"Passive@Kmax:     {passive_ok_any}/{total} (σ_max(S)<=1)")
    for k in k_eval:
        for tau in taus:
            num = success_counts[(k, tau)]
            den = max(total, 1)
            print(f"success@{tau:g} best-of-{k}: {num}/{den} = {num/den:.3f}")

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as f:
            for row in dump_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Dumped to {args.dump}")


if __name__ == "__main__":
    main()
