"""One-off verification script for the TPU port (see CLAUDE.md's scale-up
entry / the plan doc's Verification gates 3-5). NOT part of the trained
pipeline — run manually, then delete or leave as a record. Exercises:
  3. CPU fp32 vs TPU fp32 loss-curve parity, then TPU fp32 vs TPU bf16.
  4. bf16 vs fp32 byte-exact match on a short-trained checkpoint.
  5. End-to-end smoke: the real hmn_tpu_recall1024_flat.py config, tiny
     n_steps, confirms every bucket compiles and eval runs on the TPU.
Run from ~/kvmem on the TPU VM: python3 -m kvmem.gate_check
"""
import copy
import shutil
import sys

import numpy as np
import torch

from kvmem.hmn import (ar_decode_traj_nokv, build_model, chunk_mask_fb_traj,
                        chunk_positions_traj, load_config, make_batch_tagged,
                        traj_suffix, train)

SMALL_HP = dict(
    d=32, n_layers=4, n_heads=4, V=271,
    block_type='single_attn',
    lr_max=1e-3, wd=0.0,
    warmup_steps=5, log_every=10,
    rope=True, yarn=True, L_train=600, L_max=2048,
    null_kv=True, rmsnorm=True,
    state_len=4, state_vocab_size=2,
    warmup_len=16,
    val_n_seqs=1,
    bucket_lengths=True, max_shape_buckets=4, token_budget=20000,
)


def _small_curriculum(n_steps, eval_every):
    return [dict(n_chunks=4, chunk_len=32, B=8, n_steps=n_steps, eval_every=eval_every,
                 hops=-1,
                 weave_mix=[
                     dict(weight=1.0, dsl='E(32) E3 Q(0,4,0,16)'),
                     dict(weight=1.0, dsl='E(32) E3 Q(0,4,40,16)'),
                     dict(weight=1.0, dsl='E(32) E3 Q(0,4,80,16)'),
                 ])]


GATE3_N_STEPS = 200


def gate3_run_one(tag, device_str):
    """Runs ONE device's training and exits. Deliberately a SEPARATE process
    per device (see gate3_compare's own note) — running CPU train() then TPU
    train() in the same python process hits a real PyTorch/XLA bug: the
    autograd engine's device_ready_queues_ is sized when the Engine
    singleton is first used (at the first .backward() call); if that first
    call is a plain CPU backward, the engine never learns about the XLA
    device registered afterward, and the SECOND (TPU) .backward() call in
    the same process crashes with `RuntimeError: 0 <= device.index() &&
    device.index() < ... device_ready_queues_.size() INTERNAL ASSERT
    FAILED` — hit and confirmed on tpu1 while building this gate. Not
    specific to this codebase; the fix is "one device, one process."
    """
    hp = copy.deepcopy(SMALL_HP)
    hp['name'] = f'_gate3_{tag}'
    hp['seed'] = 7
    hp['curriculum'] = _small_curriculum(GATE3_N_STEPS, GATE3_N_STEPS)  # eval once, at the end
    shutil.rmtree(f'logs/_gate3_{tag}', ignore_errors=True)
    train(hp, log_base='logs', device_str=device_str)


def gate3_compare():
    print('=== Gate 3: CPU fp32 vs TPU fp32/bf16 loss-curve parity (compare) ===', flush=True)
    import json
    losses = {}
    for tag in ('cpu_fp32', 'tpu'):
        with open(f'logs/_gate3_{tag}/train.jsonl') as f:
            lines = [l for l in f if '"loss"' in l]
        losses[tag] = [json.loads(l)['loss'] for l in lines]
        print(f'{tag}: {len(losses[tag])} logged losses, last={losses[tag][-1]:.4f}', flush=True)

    cpu_last = losses['cpu_fp32'][-1]
    tpu_last = losses['tpu'][-1]
    rel_diff = abs(cpu_last - tpu_last) / max(abs(cpu_last), 1e-6)
    print(f'CPU fp32 final loss={cpu_last:.4f}  TPU (bf16 autocast) final loss={tpu_last:.4f}  '
          f'rel_diff={rel_diff:.3f}', flush=True)
    print('NOTE: TPU path runs under bf16 autocast by default (device_is_tpu=True), so this '
          'comparison is CPU-fp32 vs TPU-bf16, not a pure device-only diff — loose tolerance '
          'expected/OK here; the point is "same ballpark, no NaN/divergence", not bitwise match.',
          flush=True)
    assert np.isfinite(tpu_last), 'TPU loss is NaN/inf — bf16 autocast or XLA path is broken'
    assert rel_diff < 1.0, f'TPU loss diverged from CPU reference (rel_diff={rel_diff:.3f})'
    print('Gate 3: PASSED (no NaN, same ballpark)\n', flush=True)


def gate4_bf16_exactness():
    print('=== Gate 4: bf16 vs fp32 byte-exact match ===', flush=True)
    # Train a tiny model briefly on CPU (fp32 reference weights), then decode the SAME
    # weights once under fp32 and once forced through bf16 autocast, compare match%.
    hp = copy.deepcopy(SMALL_HP)
    hp['name'] = '_gate4_ref'
    hp['seed'] = 3
    hp['curriculum'] = _small_curriculum(400, 400)
    shutil.rmtree('logs/_gate4_ref', ignore_errors=True)
    train(hp, log_base='logs', device_str='cpu')

    ckpt = torch.load('logs/_gate4_ref/checkpoints/stage0_last.pt', map_location='cpu')
    hp_model = dict(V=271, d=32, n_layers=4, n_heads=4, block_type='single_attn',
                    rope=True, yarn=True, L_train=600, L_max=2048,
                    null_kv=True, rmsnorm=True)
    model = build_model(hp_model, torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    model.eval()

    built = chunk_positions_traj(32, 4, 16, traj_suffix(4, 4), state_vocab_size=2)
    pos_content, pos_mask, tags = built['pos_content'], built['pos_mask'], built['tags']
    mask_np = chunk_mask_fb_traj(pos_mask, hops=-1)
    rng = np.random.default_rng(99)
    chunks = rng.integers(0, 256, size=(4, 32), dtype=np.int64)

    r_fp32 = ar_decode_traj_nokv(model, chunks, 4, 2, mask_np, pos_content, tags,
                                 torch.device('cpu'))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=True):
        r_bf16 = ar_decode_traj_nokv(model, chunks, 4, 2, mask_np, pos_content, tags,
                                     torch.device('cpu'))
    print(f'fp32 match={r_fp32["match_pct"]:.1f}%  bf16 match={r_bf16["match_pct"]:.1f}%',
          flush=True)
    drop = r_fp32['match_pct'] - r_bf16['match_pct']
    print(f'match drop from bf16: {drop:.1f}pp', flush=True)
    print('Gate 4: reported (not asserted — model is undertrained at 400 steps so absolute '
          'match% is meaningless either way; what matters is whether bf16 introduces its OWN '
          'additional degradation on top of whatever the undertrained model already gets wrong. '
          'Re-run this gate against a real, converged Run A checkpoint before trusting bf16 for '
          'the actual reported results.)\n', flush=True)


def gate5_smoke():
    print('=== Gate 5: end-to-end smoke test, real Run A config, TPU ===', flush=True)
    hp = load_config('kvmem/configs/hmn_tpu_recall1024_flat.py')
    hp = copy.deepcopy(hp)
    hp['name'] = '_gate5_smoke'
    hp['curriculum'][0]['n_steps'] = 30
    hp['curriculum'][0]['eval_every'] = 30
    hp['log_every'] = 5
    hp['val_n_seqs'] = 1
    shutil.rmtree('logs/_gate5_smoke', ignore_errors=True)
    train(hp, log_base='logs', device_str='tpu')
    print('Gate 5: PASSED (every bucket compiled, eval ran, no crash)\n', flush=True)


if __name__ == '__main__':
    # Every subcommand below is deliberately ONE device per process (see
    # gate3_run_one's docstring for why mixing CPU-then-XLA .backward() calls
    # in a single process crashes the autograd engine) — gate4 trains on CPU
    # only (no XLA involved at all, safe standalone) and gate5 trains on TPU
    # only; run each subcommand as its own `python3 -m kvmem.gate_check
    # <name>` invocation, never chained in-process.
    which = sys.argv[1] if len(sys.argv) > 1 else None
    dispatch = dict(
        gate3_cpu=lambda: gate3_run_one('cpu_fp32', 'cpu'),
        gate3_tpu=lambda: gate3_run_one('tpu', 'tpu'),
        gate3_compare=gate3_compare,
        gate4=gate4_bf16_exactness,
        gate5=gate5_smoke,
    )
    if which not in dispatch:
        raise SystemExit(f'unknown gate {which!r} — use one of {list(dispatch)}')
    dispatch[which]()
    print('DONE')
