"""
kvmem/stage0_ocd.py — Recall training with OCD (Optimal Completion Distillation).

Like stage0.py recall-corpus but replaces teacher-forcing with OCD rollouts
to eliminate the train/inference mismatch (exposure bias).

Key difference from stage0 recall:
  - stage0:  loss on teacher-forced Y (gold prefix at every step)
  - stage0_ocd: AR rollout first, then OCD soft-targets (optimal completions)
                No hyperparameters, directly optimises edit distance.

Usage:
    python -m kvmem.stage0_ocd [--test-files ...] [--seg-len N] [--steps N]
                                [--ocd-every K] [--tf-warmup K]

Architecture, optimizer, and checkpoint format identical to stage0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from kvmem.data import DATA_LO, ETX, NUL, STX, make_slot_ids
from kvmem.ocd import (
    optimal_next_tokens_copy,
    ocd_sequence_loss,
)
from kvmem.stage0 import (
    KVMemModel,
    build_model,
    count_params,
    init_opt_state,
    lr_schedule,
    clip_grads,
    adam_update,
    grok_adam_update,
    make_mask_stage0,
    save_checkpoint,
    setup_run_dir,
    _eval_recall_on_file,
    _make_synthetic_recall_batch,
    _sample_varied_seg,
    RECALL_HPARAMS,
)

# ---------------------------------------------------------------------------
# OCD-specific hyperparams (extend RECALL_HPARAMS)
# ---------------------------------------------------------------------------

OCD_HPARAMS = {
    **RECALL_HPARAMS,
    # After tf_warmup steps of teacher-forcing, switch to OCD
    'tf_warmup'   : 5_000,
    # Every ocd_every steps do a full OCD rollout batch (expensive);
    # in-between steps use teacher-forcing for speed.
    'ocd_every'   : 1,    # 1 = pure OCD after warmup (slower but better)
    # Temperature for AR rollout during OCD (0.0 = greedy)
    'ocd_temp'    : 0.0,
}


# ---------------------------------------------------------------------------
# OCD rollout + loss (single example, CPU numpy)
# ---------------------------------------------------------------------------

def _ocd_rollout_one(model: KVMemModel, x_S: list[int], N: int,
                     seg_len: int, temperature: float = 0.0,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    AR rollout from KV memory for one example, compute per-step OCD targets.

    Returns:
      tokens_input : (seg_len + 2 + N + seg_len,) int32 — full sequence
                     with AR-generated Y pasted in (for loss computation)
      ocd_targets  : (seg_len + 2 + N + seg_len - 1, 256) float32
                     OCD soft-target distributions; zero rows for non-Y positions
    """
    L_S       = seg_len
    mem_block = [STX] + make_slot_ids(N) + [ETX]
    ETX_pos   = L_S + 1 + N          # index of ETX token
    L_total   = L_S + 2 + N + L_S   # full sequence length

    # Build mask (reused each step — expensive to rebuild, but needed for exact mask)
    # We use a fixed full-sequence mask and trust that positions before ETX
    # attend correctly.
    mask_jnp = jnp.array(make_mask_stage0(L_S, N, L_S))

    # Greedy AR rollout
    generated: list[int] = []
    key = jax.random.PRNGKey(seed)

    for step in range(L_S):
        cur_len  = L_S + 2 + N + step  # length of current sequence fed
        cur_tok  = jnp.array(x_S + mem_block + generated, dtype=jnp.int32)
        # Pad to L_total and use the full mask (causally masked, so future tokens
        # are attended away — padding at end is never attended to)
        pad_n  = L_total - len(cur_tok)
        cur_padded = jnp.concatenate([cur_tok, jnp.zeros(pad_n, dtype=jnp.int32)])
        logits = model(cur_padded, mask_jnp)   # (L_total, V)
        logit_last = logits[cur_len - 1]       # (V,) — predict next token

        if temperature == 0.0:
            nb = int(jnp.argmax(logit_last))
        else:
            key, sk = jax.random.split(key)
            nb = int(jax.random.choice(sk, 256,
                                       p=jax.nn.softmax(logit_last / temperature)))
        generated.append(nb)

    # Now compute OCD targets for the Y region
    # y_gen = generated sequence (length L_S)
    # x_ref = x_S (length L_S, the target to copy)
    y_gen = generated          # what model generated
    x_ref = x_S                # ground truth to copy

    # Build full token sequence with AR-generated Y
    tokens_np = np.array(x_S + mem_block + generated, dtype=np.int32)  # (L_total,)

    # Build OCD targets: shape (L_total - 1, 256)
    ocd_targets = np.zeros((L_total - 1, 256), dtype=np.float32)

    for k in range(L_S):
        # Position in full sequence = ETX_pos + k
        # y_gen prefix so far = generated[:k]
        # target next token distribution = OCD over x_ref given y_gen[:k]
        dist = optimal_next_tokens_copy(y_gen[:k], x_ref, vocab_size=256)
        pos  = ETX_pos + k           # position in tokens_np
        ocd_targets[pos] = dist      # targets[pos] is what we predict at pos+1

    return tokens_np, ocd_targets


# ---------------------------------------------------------------------------
# Batched OCD step (JAX, jitted)
# ---------------------------------------------------------------------------

def _make_ocd_step(hp: dict, mask_jnp: jnp.ndarray):
    """Return jitted OCD train step.

    The OCD targets are computed on CPU (numpy) before this call.
    This function just computes the loss and gradient from precomputed targets.
    """
    L_S      = hp['seg_len']
    N        = hp['N']
    ETX_pos  = L_S + 1 + N
    wd       = hp['wd']
    optimizer = hp.get('optimizer', 'adamw')
    use_grok  = (optimizer == 'grokadamw')

    @jax.jit
    def _step(model, opt_state, tokens_b, targets_b, step, lr):
        """
        tokens_b : (B, L_total) int32
        targets_b: (B, L_total-1, 256) float32 — OCD soft targets
        """
        def _loss(m):
            B   = tokens_b.shape[0]
            L   = tokens_b.shape[1]
            # Forward pass: get logits for full sequence
            logits = jax.vmap(lambda tok: m(tok, mask_jnp))(tokens_b)  # (B, L, V)
            logits_shift = logits[:, :-1, :]   # (B, L-1, V)
            # OCD sequence loss — only Y positions have nonzero targets
            # Sum over B and mean over active positions
            loss = jnp.mean(jax.vmap(ocd_sequence_loss)(logits_shift, targets_b))
            return loss

        params = eqx.filter(model, eqx.is_array)
        loss, grads = jax.value_and_grad(_loss)(model)
        grads_arr = eqx.filter(grads, eqx.is_array)
        grads_arr = clip_grads(grads_arr, max_norm=hp['grad_clip'])

        if use_grok:
            new_params, new_opt = grok_adam_update(
                params, grads_arr, opt_state, lr,
                rho=hp.get('grok_rho', 0.95), wd=wd,
                step=step, batch_size=hp['B'])
        else:
            new_params, new_opt = adam_update(
                params, grads_arr, opt_state, lr, wd=wd, step=step)

        deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
        new_model = eqx.apply_updates(model, deltas)
        return new_model, new_opt, loss

    return _step


def _make_tf_step(hp: dict, mask_jnp: jnp.ndarray):
    """Return jitted teacher-forcing step (fast, for warmup and in-between)."""
    L_S      = hp['seg_len']
    N        = hp['N']
    ETX_pos  = L_S + 1 + N
    wd       = hp['wd']
    optimizer = hp.get('optimizer', 'adamw')
    use_grok  = (optimizer == 'grokadamw')
    B        = hp['B']

    @jax.jit
    def _step(model, opt_state, tokens_b, step, lr):
        L = tokens_b.shape[1]

        def _loss(m):
            logits = jax.vmap(lambda tok: m(tok, mask_jnp))(tokens_b)
            lp     = jax.nn.log_softmax(logits[:, :-1], axis=-1)
            tgts   = tokens_b[:, 1:]
            nll    = -lp[jnp.arange(B)[:, None], jnp.arange(L-1)[None, :], tgts]
            pos       = jnp.arange(L - 1)
            Y_end     = ETX_pos + L_S
            mask_cont = ((pos >= ETX_pos) & (pos < Y_end)).astype(jnp.float32)
            return jnp.sum(nll * mask_cont[None, :]) / (mask_cont.sum() * B + 1e-8)

        params = eqx.filter(model, eqx.is_array)
        loss, grads = jax.value_and_grad(_loss)(model)
        grads_arr = eqx.filter(grads, eqx.is_array)
        grads_arr = clip_grads(grads_arr, max_norm=hp['grad_clip'])

        if use_grok:
            new_params, new_opt = grok_adam_update(
                params, grads_arr, opt_state, lr,
                rho=hp.get('grok_rho', 0.95), wd=wd,
                step=step, batch_size=B)
        else:
            new_params, new_opt = adam_update(
                params, grads_arr, opt_state, lr, wd=wd, step=step)

        deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
        new_model = eqx.apply_updates(model, deltas)
        return new_model, new_opt, loss

    return _step


# ---------------------------------------------------------------------------
# Main OCD training loop
# ---------------------------------------------------------------------------

def train_recall_ocd(hp: dict, test_files: list[str], log_base: str = 'logs'):
    """
    OCD recall training:
      Phase 1 (tf_warmup steps): teacher-forcing on varied synthetic data
      Phase 2 (remaining steps): OCD rollout + optimal soft targets

    OCD eliminates exposure bias: model trains on its own generated prefixes,
    supervised by the set of ground-truth continuations that minimize edit distance.
    No hyperparameters for the curriculum — OCD adapts automatically.
    """
    seg_len      = hp['seg_len']
    N            = hp['N']
    B            = hp['B']
    n_steps      = hp['n_steps']
    warmup_bytes = hp['warmup_bytes']
    val_every    = hp['val_every']
    ckpt_every   = hp['ckpt_every']
    log_every    = hp['log_every']
    tf_warmup    = hp.get('tf_warmup', 5_000)
    ocd_every    = hp.get('ocd_every', 1)
    ocd_temp     = hp.get('ocd_temp', 0.0)

    key = jax.random.PRNGKey(hp['seed'])
    key, mkey = jax.random.split(key)
    model     = build_model(hp, mkey)
    optimizer = hp.get('optimizer', 'adamw')
    opt_state = init_opt_state(model, optimizer=optimizer)

    run_dir  = setup_run_dir(log_base, 'recall_ocd')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    raw_log_f   = open(os.path.join(run_dir, 'train.log'),   'w', buffering=1)
    train_log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', buffering=1)

    def _log(msg):
        tqdm.write(msg)
        raw_log_f.write(msg + '\n')
        raw_log_f.flush()

    def _jlog(rec):
        train_log_f.write(json.dumps(rec) + '\n')
        train_log_f.flush()

    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump({**hp, 'test_files': test_files}, f, indent=2)

    pcount   = count_params(model)
    L_total  = seg_len + 2 + N + seg_len
    mask_jnp = jnp.array(make_mask_stage0(seg_len, N, seg_len))

    _log(f'\n=== Recall+OCD training | run_dir={run_dir} ===')
    _log(f'  Training: SYNTHETIC varied distributions, NO real text')
    _log(f'  Phase 1: teacher-forcing for {tf_warmup} steps (warmup)')
    _log(f'  Phase 2: OCD rollout every {ocd_every} steps')
    _log(f'  Test files: {test_files}')
    _log(f'  Params: {pcount["total"]:,}')
    _log(f'  seg_len={seg_len}  N={N}  KV_floats={2*hp["n_layers"]*N*hp["d"]:,}')
    _log(f'  Steps={n_steps}  Batch={B}  Optimizer={optimizer}')

    rng = np.random.default_rng(hp['seed'] + 1)
    t0  = time.time()

    ocd_step = _make_ocd_step(hp, mask_jnp)
    tf_step  = _make_tf_step(hp, mask_jnp)

    pbar = tqdm(range(1, n_steps + 1), desc='recall_ocd', unit='step',
                dynamic_ncols=True, file=sys.stdout)

    for step in pbar:
        lr = lr_schedule(step, hp)

        if step <= tf_warmup or (step % ocd_every != 0):
            # Teacher-forcing step (fast)
            np_tokens = _make_synthetic_recall_batch(rng, B, seg_len, N)
            tokens_b  = jnp.array(np_tokens)
            model, opt_state, loss = tf_step(model, opt_state, tokens_b, step, lr)
            loss_f = float(loss)
            mode   = 'tf'
        else:
            # OCD rollout step
            # 1. Generate B random source segments
            segments = [list(_sample_varied_seg(rng, seg_len)) for _ in range(B)]
            # 2. Per-example OCD rollout (CPU numpy)
            all_tokens = []
            all_targets = []
            for i, x_S in enumerate(segments):
                tok_np, tgt_np = _ocd_rollout_one(
                    model, x_S, N, seg_len,
                    temperature=ocd_temp, seed=step * B + i)
                all_tokens.append(tok_np)
                all_targets.append(tgt_np)
            tokens_b  = jnp.array(np.stack(all_tokens,  axis=0))   # (B, L_total)
            targets_b = jnp.array(np.stack(all_targets, axis=0))   # (B, L_total-1, 256)
            model, opt_state, loss = ocd_step(model, opt_state, tokens_b, targets_b, step, lr)
            loss_f = float(loss)
            mode   = 'ocd'

        pbar.set_postfix(loss=f'{loss_f:.4f}', lr=f'{lr:.1e}', mode=mode, refresh=False)

        if step % log_every == 0:
            elapsed = time.time() - t0
            _log(f'  step={step:5d}/{n_steps}  loss={loss_f:.4f}  lr={lr:.2e}  [{mode}]  {elapsed:.0f}s')
            _jlog(dict(step=step, loss=loss_f, lr=lr, mode=mode, elapsed=elapsed))

        if step % val_every == 0:
            eval_rng = np.random.default_rng(step)
            for fpath in test_files:
                if not os.path.exists(fpath):
                    continue
                acc = _eval_recall_on_file(model, fpath, seg_len, N,
                                           warmup_bytes, eval_rng)
                fname = os.path.basename(fpath)
                _log(f'  [val]  step={step:5d}  {fname}  recall={acc:.1f}%  '
                     f'(warmup={warmup_bytes}B  tgt={seg_len-warmup_bytes}B)')
                _jlog(dict(step=step, file=fpath, recall_pct=acc))

        if step % ckpt_every == 0 or step == n_steps:
            ckpt_path = os.path.join(ckpt_dir, f'recall_ocd_step{step}')
            save_checkpoint(ckpt_path, model, step, hp)
            _log(f'  [ckpt] {ckpt_path}')

    pbar.close()
    _log(f'\nDone. Total time: {time.time()-t0:.0f}s')
    train_log_f.close()
    raw_log_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(prog='kvmem.stage0_ocd')
    parser.add_argument('--test-files', type=str, nargs='+',
                        default=['datasets/1.txt', 'datasets/suratalfatihah.txt'])
    parser.add_argument('--seg-len',   type=int,   default=64)
    parser.add_argument('--N',         type=int,   default=32)
    parser.add_argument('--steps',     type=int,   default=None)
    parser.add_argument('--log-dir',   type=str,   default='logs')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--warmup-bytes', type=int, default=4)
    parser.add_argument('--optimizer', type=str,   default='adamw',
                        choices=['adamw', 'grokadamw'])
    parser.add_argument('--wd',        type=float, default=None)
    parser.add_argument('--tf-warmup', type=int,   default=5_000,
                        help='Steps of teacher-forcing before switching to OCD')
    parser.add_argument('--ocd-every', type=int,   default=1,
                        help='Do OCD rollout every N steps (1=always after tf_warmup)')
    args = parser.parse_args()

    hp = dict(OCD_HPARAMS)
    hp['seg_len']      = args.seg_len
    hp['N']            = args.N
    hp['seed']         = args.seed
    hp['warmup_bytes'] = args.warmup_bytes
    hp['optimizer']    = args.optimizer
    hp['tf_warmup']    = args.tf_warmup
    hp['ocd_every']    = args.ocd_every
    if args.wd is not None:
        hp['wd'] = args.wd
    if args.steps:
        hp['n_steps'] = args.steps

    train_recall_ocd(hp, test_files=args.test_files, log_base=args.log_dir)


if __name__ == '__main__':
    main()
