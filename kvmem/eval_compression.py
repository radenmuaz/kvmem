"""
kvmem/eval_compression.py — test-time compression-quality diagnostics for any
HMNModel checkpoint (kvmem/hmn.py), including checkpoints trained ONLY on
random bytes that have never seen structured data. The point of these
diagnostics is to separate three things people conflate under "the model
compresses well": genuinely storing a compressed representation in STATE,
memorizing via static weights (rule-inference bypassing memory), and simply
not having learned anything useful (underfitting) or not generalizing past
training-specific instances (overfitting). See CLAUDE.md's "Structured-data
track" section for the full design discussion this implements.

Four diagnostics, meant to be run IN ORDER — each gates the next, since a
"good compression number" is meaningless if earlier gates fail:

  1. state_ablation_gate   — does recall even depend on the encoding-pass
                              STATE at all? (necessary, not sufficient, for
                              any later claim about compression IN STATE)
  2. floor_comparison       — is loss below the no-compression floor (~8
                              bits/byte) and the trivial-classical-compressor
                              floor (zlib, via measure_bits_per_byte)?
                              (underfitting check)
  3. train_test_gap         — does loss on fresh (never-generated-before)
                              rule/seed draws match the training distribution's
                              typical loss at the same target_bits? (overfitting
                              check — approximate, see docstring)
  4. compression_sensitivity_curve — does loss vary systematically with TRUE
                              compressibility (target_bits) at FIXED
                              chunk_len/state_len? This is the main "is
                              compression happening" signal, and the one
                              that's safe to run zero-shot on ANY checkpoint,
                              including one trained only on random bytes —
                              a flat curve (no sensitivity to target_bits)
                              on such a model is itself an informative
                              negative result (no OOD compression ability
                              emerges for free).

An OPTIONAL, secondary max_recallable_length sweep (varying chunk_len itself,
to get a direct effective-compression-ratio number) is also implemented, but
flagged as needing chunk_len values the model wasn't necessarily trained on —
treat its results with more caution than diagnostics 1-4, which all hold
chunk_len/state_len fixed at whatever the checkpoint was actually trained
with (no extrapolation confound).

Usage:
    python3 -m kvmem.eval_compression --ckpt kvmem/logs/hmn_stage0_round0_single/checkpoints/stage0_best.pt --device mps
    python3 -m kvmem.eval_compression --ckpt <path> --device mps --kinds ca,chaotic,fractal
    python3 -m kvmem.eval_compression --ckpt <path> --device mps --sweep-length   # optional diagnostic 5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from kvmem.hmn import (
    build_model,
    chunk_positions_chained,
    chunk_mask_fb,
    make_test_sequences,
    _cyclic_state_ids,
    _positional_ls_nll,
)
from kvmem.structured_data import generate_structured_chunks, measure_bits_per_byte


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _load(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt['hp']
    hp_model = dict(
        V=hp['V'], d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
        block_type=hp.get('block_type', 'single_attn'),
        rope=hp.get('rope', True), yarn=hp.get('yarn', True),
        null_kv=hp.get('null_kv', True), rmsnorm=hp.get('rmsnorm', False),
        chunk_attn=0,
    )
    model = build_model(hp_model, device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, hp


def _build_layout(hp: dict, chain_steps=None):
    """pos_content/pos_mask/mask_t/tags for hp's own (n_chunks, chunk_len,
    state_len, chain_steps) config — reused by every diagnostic below so
    they're all evaluated under the checkpoint's OWN trained layout, no
    extrapolation unless explicitly asked for (see sweep_max_recallable_length)."""
    state_len = hp.get('state_len', 8)
    state_vocab_size = hp.get('state_vocab_size', 2)
    warmup_len = hp['warmup_len']
    stage_cfg = hp['curriculum'][0]
    n_chunks, chunk_len = stage_cfg['n_chunks'], stage_cfg['chunk_len']
    chain_steps = chain_steps or stage_cfg['chain_steps']
    n_refine = stage_cfg.get('n_refine', 0)

    built = chunk_positions_chained(n_chunks, chunk_len, state_len, warmup_len,
                                    chain_steps, n_refine=n_refine,
                                    state_vocab_size=state_vocab_size)
    return built['pos_content'], built['pos_mask'], built['tags'], n_chunks, chunk_len, state_len, state_vocab_size


def _fill_tokens(pos_content: dict, tags: list, chunks_list, sids: np.ndarray,
                 warmup_len: int) -> np.ndarray:
    """Ground-truth teacher-forced token fill (same pattern as attn_viz.py's
    _fill_tokens) — deterministic, no argmax feedback loop needed since these
    diagnostics only need a single forward pass's loss, not AR decode."""
    L = pos_content['L']
    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    wl = warmup_len
    for rb in pos_content['rec_blocks']:
        span_s, span_e = rb['span']
        gt_span = np.concatenate([np.array(chunks_list[i]) for i in range(span_s, span_e)])
        if rb['type'] == 'iq':
            if 'queue0' in rb:
                tok[rb['queue0']:rb['queue1']] = sids
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = np.array(gt_span[:wl], dtype=np.int64)
            tok[rb['c0']:rb['c1']] = np.array(gt_span[wl:wl + rb['out_len']], dtype=np.int64)
        else:
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = np.array(gt_span[:wl], dtype=np.int64)
            tok[rb['c0']:rb['c1']] = np.array(gt_span[wl:wl + rb['out_len']], dtype=np.int64)

    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[tag_pos] = tag_ids
    return tok


def _teacher_forced_bits(model, pos_content: dict, mask_t: torch.Tensor, tags: list,
                         chunks_list, sids: np.ndarray, warmup_len: int,
                         device: torch.device, h_inject: dict | None = None) -> float:
    """One forward pass, ground-truth-filled, NLL in bits/byte over every
    'is_clean' rec_block's output region — the actual bits/byte diagnostic
    number used by every check below."""
    tok = _fill_tokens(pos_content, tags, chunks_list, sids, warmup_len)
    tok_t = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tok_t, mask_t, h_inject=h_inject)
        if isinstance(logits, tuple):
            logits = logits[0]

    total_nll, total_n = 0.0, 0
    for rb in pos_content['rec_blocks']:
        if not rb['is_clean']:
            continue
        lp = F.log_softmax(logits[:, rb['c0'] - 1:rb['c1'] - 1], dim=-1)
        tgt = tok_t[:, rb['c0']:rb['c1']]
        nll = _positional_ls_nll(lp, tgt, 0.0)
        total_nll += float(nll.sum())
        total_n += nll.numel()
    return (total_nll / max(total_n, 1)) / np.log(2.0)  # nats -> bits


# ---------------------------------------------------------------------------
# Diagnostic 1 — state ablation gate
# ---------------------------------------------------------------------------

def state_ablation_gate(model, hp: dict, device: torch.device,
                        chunks_list, n_probe: int = 1) -> dict:
    """
    Does recall depend on the encoding-pass STATE at all? Runs the SAME
    teacher-forced forward twice: once normally, once with every encoding
    block's STATE region overridden (via h_inject — the same mechanism
    STATE_QUEUE uses, reused here for ablation instead of chaining) with
    fresh random noise instead of the model's own computed encoding.

    A large gap (ablated bits/byte >> normal bits/byte) means recall
    genuinely depends on STATE. A small/no gap means the model isn't using
    STATE for this task — it's either predicting from something else
    entirely (e.g. a degenerate shortcut) or just failing outright; either
    way, none of diagnostics 2-4's "compression quality" numbers mean
    anything until this gap is confirmed large.
    """
    pos_content, pos_mask, tags, n_chunks, chunk_len, state_len, state_vocab_size = _build_layout(hp)
    mask_np = chunk_mask_fb(pos_mask)
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    warmup_len = hp['warmup_len']

    d = hp['d']
    normal_bits = _teacher_forced_bits(model, pos_content, mask_t, tags, chunks_list,
                                       sids, warmup_len, device, h_inject=None)

    h_inject = {}
    for b in pos_content['enc_blocks']:
        noise = torch.randn(1, b['sl1'] - b['sl0'], d, device=device) * 0.5
        h_inject[(b['sl0'], b['sl1'])] = noise
    ablated_bits = _teacher_forced_bits(model, pos_content, mask_t, tags, chunks_list,
                                        sids, warmup_len, device, h_inject=h_inject)

    return dict(normal_bits=normal_bits, ablated_bits=ablated_bits,
               gap=ablated_bits - normal_bits,
               depends_on_state=(ablated_bits - normal_bits) > 1.0)  # >1 bit/byte gap = clear dependence


# ---------------------------------------------------------------------------
# Diagnostic 2 — floor comparison (underfitting check)
# ---------------------------------------------------------------------------

CHANCE_FLOOR_BITS = 8.0  # uniform-random-byte prediction, i.e. "storing nothing"


def floor_comparison(model, hp: dict, device: torch.device, chunks_list) -> dict:
    """L_model vs (a) chance floor ~8 bits/byte, (b) zlib's practical floor
    on the SAME sequence. L_model should be well below both for the model to
    be doing anything better than a trivial classical compressor."""
    pos_content, pos_mask, tags, n_chunks, chunk_len, state_len, state_vocab_size = _build_layout(hp)
    mask_np = chunk_mask_fb(pos_mask)
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    warmup_len = hp['warmup_len']

    l_model = _teacher_forced_bits(model, pos_content, mask_t, tags, chunks_list,
                                   sids, warmup_len, device)
    flat = np.concatenate(chunks_list)
    zlib_floor = measure_bits_per_byte(flat)

    return dict(l_model=l_model, chance_floor=CHANCE_FLOOR_BITS, zlib_floor=zlib_floor,
               beats_chance=l_model < CHANCE_FLOOR_BITS - 0.5,
               beats_zlib=l_model < zlib_floor)


# ---------------------------------------------------------------------------
# Diagnostic 3 — train/test generalization gap (approximate)
# ---------------------------------------------------------------------------

def train_test_gap(model, hp: dict, device: torch.device, kind: str,
                   target_bits: float, n_seeds: int = 8, base_seed: int = 10_000) -> dict:
    """
    APPROXIMATE overfitting check. This script has no access to the
    checkpoint's actual per-example training loss log, so it can't do a
    literal train-vs-test comparison. What it CAN do: generate n_seeds fresh
    (never-before-generated, since base_seed is far outside any plausible
    training RNG stream) sequences at the SAME target_bits and report the
    spread of L_model across them. High variance across
    held-out-rule/seed draws at a FIXED target_bits is itself suggestive of
    inconsistent generalization (the model does well on some rule instances
    and badly on others, rather than uniformly learning "decode whatever
    rule you're given") — a real, if indirect, overfitting signal. Low
    variance doesn't PROVE no overfitting (would need actual training-loss
    logs for that), but high variance is informative on its own.
    """
    bits_list = []
    for i in range(n_seeds):
        rng = np.random.default_rng(base_seed + i)
        stage_cfg = hp['curriculum'][0]
        chunks = generate_structured_chunks(rng, kind, stage_cfg['n_chunks'],
                                            stage_cfg['chunk_len'], target_bits=target_bits)
        chunks_list = [chunks[k] for k in range(chunks.shape[0])]
        r = floor_comparison(model, hp, device, chunks_list)
        bits_list.append(r['l_model'])
    bits_arr = np.array(bits_list)
    return dict(kind=kind, target_bits=target_bits, seeds_tested=n_seeds,
               mean_bits=float(bits_arr.mean()), std_bits=float(bits_arr.std()),
               min_bits=float(bits_arr.min()), max_bits=float(bits_arr.max()),
               per_seed_bits=bits_list)


# ---------------------------------------------------------------------------
# Diagnostic 4 — compression sensitivity curve (the main signal)
# ---------------------------------------------------------------------------

def compression_sensitivity_curve(model, hp: dict, device: torch.device, kind: str,
                                  target_bits_grid=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
                                  seed: int = 20_000) -> dict:
    """
    Sweep target_bits at FIXED chunk_len/state_len (the checkpoint's own
    trained layout — no extrapolation confound) and measure L_model at each
    point, plus a pure-random-byte reference point (target_bits ~8,
    equivalent to this project's existing random-byte eval). A model with
    genuine OOD compression capability should show L_model decreasing as
    target_bits decreases (more compressible content -> lower loss, since
    the SAME fixed STATE capacity now has to represent less true information).
    A flat curve (L_model roughly constant regardless of target_bits) means
    no compression sensitivity emerged — informative even for (especially
    for) a checkpoint trained only on random bytes, since it's a genuine
    zero-shot generalization test.
    """
    stage_cfg = hp['curriculum'][0]
    n_chunks, chunk_len = stage_cfg['n_chunks'], stage_cfg['chunk_len']
    curve = []
    for tb in target_bits_grid:
        rng = np.random.default_rng(seed + int(tb * 1000))
        chunks = generate_structured_chunks(rng, kind, n_chunks, chunk_len, target_bits=tb)
        chunks_list = [chunks[k] for k in range(chunks.shape[0])]
        r = floor_comparison(model, hp, device, chunks_list)
        curve.append(dict(target_bits=tb, achieved_bits=r['l_model'],
                          zlib_floor=r['zlib_floor']))

    rng_rand = np.random.default_rng(seed)
    val_seqs = make_test_sequences(n_chunks * chunk_len)
    rand_bytes = list(val_seqs.values())[0]
    rand_chunks = [rand_bytes[k * chunk_len:(k + 1) * chunk_len] for k in range(n_chunks)]
    r_rand = floor_comparison(model, hp, device, rand_chunks)
    curve.append(dict(target_bits=8.0, achieved_bits=r_rand['l_model'],
                      zlib_floor=r_rand['zlib_floor'], note='pure random reference'))

    bits_vals = [c['achieved_bits'] for c in curve[:-1]]  # exclude random reference
    monotonic_ish = all(bits_vals[i] <= bits_vals[i + 1] + 0.3 for i in range(len(bits_vals) - 1))
    return dict(kind=kind, curve=curve, roughly_monotonic=monotonic_ish)


# ---------------------------------------------------------------------------
# Optional diagnostic 5 — max-recallable-length sweep (extrapolation risk)
# ---------------------------------------------------------------------------

def sweep_max_recallable_length(model, hp: dict, device: torch.device, kind: str,
                                target_bits: float, chunk_lens=(8, 16, 24, 32, 48, 64),
                                match_threshold: float = 95.0, seed: int = 30_000) -> dict:
    """
    OPTIONAL, secondary. Varies chunk_len itself (holding state_len fixed) to
    find the longest chunk the model still recalls near-perfectly, separately
    for random bytes and for `kind` at `target_bits`. The ratio
    max_len(structured)/max_len(random) is the Effective Compression Ratio
    (ECR); compare against the theoretical ceiling 8/target_bits for an
    efficiency estimate.

    Also reports two capacity normalizations, since raw max_len numbers alone
    conflate two different things:
      - capacity_bits_per_state_token = (max_len_random * 8) / state_len —
        max_len_random*8 is the EMPIRICAL number of bits losslessly stored
        for INCOMPRESSIBLE content (the only content type where "bits stored"
        is unambiguous — Shannon leaves no other explanation for successful
        recall of random bytes). Dividing by state_len ties that measured
        capacity to the register WIDTH hyperparameter directly.
      - capacity_bits_per_million_params = (max_len_random * 8) / (n_params/1e6)
        — normalizes by total model size, not just state_len. This matters
        because a bigger model could show a higher raw max_len_random simply
        by having more WEIGHT capacity to hard-code patterns (the same
        rule-in-weights contamination risk flagged elsewhere in this file,
        and the exact concern docs/MDL_MODEL_SIZE.md already raises: model
        size should track algorithm complexity, not be a free source of
        apparent capability). This normalization does NOT resolve that
        ambiguity by itself (state_ablation_gate is still the tool for "is
        this coming from STATE or from weights") — it only lets two
        DIFFERENT-SIZED models' capacity numbers be compared fairly instead
        of the bigger one automatically looking better regardless of cause.

    CAVEAT: chunk_len values other than what the checkpoint was actually
    trained with require the model to extrapolate (RoPE/position
    generalization), which is a SEPARATE capability from compression this
    script isn't trying to isolate — a low ECR here could mean either "no
    compression" or "can't extrapolate to this chunk_len at all," and this
    function can't tell you which. Treat results here with more caution than
    diagnostics 1-4, which all stay at the checkpoint's own trained chunk_len.
    """
    def _max_len(gen_kind, tb):
        state_len = hp.get('state_len', 8)
        state_vocab_size = hp.get('state_vocab_size', 2)
        warmup_len = hp['warmup_len']
        best = 0
        for cl in chunk_lens:
            rng = np.random.default_rng(seed + cl)
            if gen_kind == 'random':
                seqs = make_test_sequences(cl)
                seq = list(seqs.values())[0]
                chunks_list = [np.array(seq, dtype=np.int64)]
            else:
                chunks = generate_structured_chunks(rng, gen_kind, 1, cl, target_bits=tb)
                chunks_list = [chunks[0]]
            built = chunk_positions_chained(1, cl, state_len, warmup_len, [(0, 1)],
                                            n_refine=0, state_vocab_size=state_vocab_size)
            pos_content, pos_mask, tags = built['pos_content'], built['pos_mask'], built['tags']
            mask_t = torch.tensor(chunk_mask_fb(pos_mask), dtype=torch.float32, device=device)
            sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
            tok = _fill_tokens(pos_content, tags, chunks_list, sids, warmup_len)
            tok_t = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(tok_t, mask_t)
            rb = pos_content['rec_blocks'][0]
            pred = logits[0, rb['c0'] - 1:rb['c1'] - 1].argmax(-1).cpu().numpy()
            tgt = tok[rb['c0']:rb['c1']]
            match = 100.0 * float((pred == tgt).sum()) / max(len(tgt), 1)
            if match >= match_threshold:
                best = cl
            else:
                break
        return best

    max_random = _max_len('random', None)
    max_structured = _max_len(kind, target_bits)
    ecr = max_structured / max(max_random, 1)
    ceiling = 8.0 / target_bits

    state_len = hp.get('state_len', 8)
    n_params = model.count_params()
    capacity_bits = max_random * 8.0  # empirical: only unambiguous for INCOMPRESSIBLE (random) content
    bits_per_state_token = capacity_bits / max(state_len, 1)
    bits_per_million_params = capacity_bits / max(n_params / 1e6, 1e-9)

    return dict(kind=kind, target_bits=target_bits, max_len_random=max_random,
               max_len_structured=max_structured, ecr=ecr, theoretical_ceiling=ceiling,
               efficiency=ecr / ceiling if ceiling > 0 else float('nan'),
               state_len=state_len, n_params=n_params,
               capacity_bits_estimate=capacity_bits,
               capacity_bits_per_state_token=bits_per_state_token,
               capacity_bits_per_million_params=bits_per_million_params)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Compression-quality diagnostics for an HMNModel checkpoint')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--kinds', default='ca,chaotic,fractal')
    p.add_argument('--sweep-length', action='store_true', help='also run the optional chunk_len sweep (diagnostic 5)')
    args = p.parse_args()

    device = torch.device(args.device)
    model, hp = _load(args.ckpt, device)
    kinds = args.kinds.split(',')

    val_seqs = make_test_sequences(hp['curriculum'][0]['n_chunks'] * hp['curriculum'][0]['chunk_len'])
    rand_bytes = list(val_seqs.values())[0]
    n_chunks, chunk_len = hp['curriculum'][0]['n_chunks'], hp['curriculum'][0]['chunk_len']
    rand_chunks_list = [rand_bytes[k * chunk_len:(k + 1) * chunk_len] for k in range(n_chunks)]

    print('=== Diagnostic 1: STATE ablation gate (random bytes) ===')
    g = state_ablation_gate(model, hp, device, rand_chunks_list)
    print(f'  normal={g["normal_bits"]:.2f} bits/byte  ablated={g["ablated_bits"]:.2f} bits/byte  '
         f'gap={g["gap"]:.2f}  depends_on_state={g["depends_on_state"]}')
    if not g['depends_on_state']:
        print('  GATE FAILED: recall does not clearly depend on STATE — diagnostics below may not be meaningful.')

    print('\n=== Diagnostic 2: floor comparison (random bytes) ===')
    f = floor_comparison(model, hp, device, rand_chunks_list)
    print(f'  L_model={f["l_model"]:.2f}  chance_floor={f["chance_floor"]:.2f}  '
         f'zlib_floor={f["zlib_floor"]:.2f}  beats_chance={f["beats_chance"]}  beats_zlib={f["beats_zlib"]}')

    for kind in kinds:
        print(f'\n=== Diagnostic 3: train/test gap ({kind}, target_bits=3.0) ===')
        tt = train_test_gap(model, hp, device, kind, target_bits=3.0)
        print(f'  mean={tt["mean_bits"]:.2f}  std={tt["std_bits"]:.2f}  '
             f'range=[{tt["min_bits"]:.2f}, {tt["max_bits"]:.2f}]  (n={tt["seeds_tested"]} held-out seeds)')

        print(f'\n=== Diagnostic 4: compression sensitivity curve ({kind}) ===')
        c = compression_sensitivity_curve(model, hp, device, kind)
        for point in c['curve']:
            note = point.get('note', '')
            print(f'  target_bits={point["target_bits"]:.1f}  achieved={point["achieved_bits"]:.2f} bits/byte  '
                 f'zlib_floor={point["zlib_floor"]:.2f}  {note}')
        print(f'  roughly_monotonic={c["roughly_monotonic"]}')

        if args.sweep_length:
            print(f'\n=== Diagnostic 5 (optional): max-recallable-length sweep ({kind}, target_bits=2.0) ===')
            s = sweep_max_recallable_length(model, hp, device, kind, target_bits=2.0)
            print(f'  max_len_random={s["max_len_random"]}  max_len_structured={s["max_len_structured"]}  '
                 f'ECR={s["ecr"]:.2f}  theoretical_ceiling={s["theoretical_ceiling"]:.2f}  '
                 f'efficiency={s["efficiency"]:.2f}')
            print(f'  state_len={s["state_len"]}  n_params={s["n_params"]:,}  '
                 f'capacity_bits(empirical,random)={s["capacity_bits_estimate"]:.0f}  '
                 f'bits/state_token={s["capacity_bits_per_state_token"]:.1f}  '
                 f'bits/M_params={s["capacity_bits_per_million_params"]:.1f}')


if __name__ == '__main__':
    main()
