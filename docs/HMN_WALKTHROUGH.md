# HMN Walkthrough — train & eval, c64 → weave

A hands-on run-through: what commands to type, in what order, and what to
look at in the output. For *what the architecture is and why* (STATE, the
relay, the E/S/Q DSL), see [`HMN_RECIPE.md`](HMN_RECIPE.md) — this doc
assumes that context and focuses on the mechanics of running `kvmem/hmn.py`.

All commands run from the repo root. Only ever run **one** training job at
a time — MPS can't share a device between processes.

---

## 0. What you're running

Everything goes through one entry point:

```
python3 -m kvmem.hmn --config <config.py> --device mps [--pretrained <ckpt.pt>] [--log-dir logs]
```

A config file (`kvmem/configs/*.py`) is just a Python module defining an
`hp` dict — hyperparameters plus a `curriculum` list of training stages.
`train()` (`kvmem/hmn.py`) reads it, builds the model, and writes to
`<log-dir>/<hp['name']>/`:

```
logs/<name>/
  train.log          # tqdm progress + periodic eval blocks (human-readable)
  train.jsonl        # same data, one JSON record per line (machine-readable)
  train_status.log   # latest-status snapshot, overwritten each log_every
  checkpoints/
    stage<N>_last.pt  # most recent checkpoint for stage N
    stage<N>_best.pt  # best-val checkpoint for stage N (use this to warm-start later stages)
    stage<N>_end.pt   # final checkpoint when stage N completes
```

---

## 1. Simplest case: `hmn_single_recall_c64.py` (one chunk, no routing, no relay)

`kvmem/configs/hmn_single_recall_c64.py` — the most basic version of the
task there is: `n_chunks=1`, so there's nothing to route across and no
relay to learn, just "compress 64 bytes into one STATE register, then
recall them from an 8-byte warmup." This is the right first run — it
validates that STATE-compression works on the current vocab at all,
before any multi-chunk or cross-chain-step complexity.

```bash
python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall_c64.py --device mps
```

`chain_steps=[(0, 1)]` — one query, spanning the single chunk. 100,000
steps, trained from scratch (no `_pretrained_ckpt`).

**Watch during training** (`tail -f logs/hmn_single_recall_c64/train.log`):

```
stage0:  20%|##        | 20000/100000 [...]  loss=1.55, lr=1.4e-04
--- stage=0 step=40000/100000  g=40000  loss=0.28  ---
  val/srs/up_counter      per-span=[...]  stitched=...%
  val/srs/MEAN               match=...%
```

`loss` should fall from ~3-4 (near-random over a 256-byte alphabet) toward
0. `val/srs/MEAN` is the number that matters — a converged run reaches
100% well before the full 100,000 steps (measured: 67% at step 40000, 100%
by step 60000). The checkpoint for the next step is
`logs/hmn_single_recall_c64/checkpoints/stage0_best.pt`.

### Scaling up: `hmn_single_recall_c128.py`

Same single-chunk task, `chunk_len` doubled to 128 — still the simplest
architecture, just testing whether one STATE register holds more bytes.
Warm-started from c64 (`_pretrained_ckpt` already set inside the config —
no `--pretrained` flag needed):

```bash
python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall_c128.py --device mps
```

This is the start of a `chunk_len` ladder — each stage warm-starts from the
previous one's best checkpoint (c64 → c128 → c256 → c512, if those configs
exist), not from c64 directly past the first hop.

---

## 2. Adding multi-chunk routing + the relay: `hop`

`kvmem/configs/hmn_recall_queue.py` jumps straight from the c64 base
above to `n_chunks=4` and chains 3 recall units
(`chain_steps=[(0,2),(1,3),(2,4)]`) with `hops=1` — each chain step's STATE
gets a narrow, learned attention permission to read the *previous* chain
step's STATE directly (the relay). It warm-starts from `c64`'s checkpoint
(same architecture — d/n_layers/n_heads/state_len/V unchanged, only
`n_chunks`/`chain_steps` differ, so weights transfer directly).

`hmn_routing_4to1_state.py` (`solo`) — an earlier, separate bootstrap stage
that also routed across 4 chunks but with no relay — is treated as an
archived experiment (no checkpoint for it exists on disk) and is no longer
part of this pipeline.

```bash
python3 -m kvmem.hmn --config kvmem/configs/hmn_recall_queue.py --device mps \
    --pretrained logs/hmn_single_recall_c64/checkpoints/stage0_best.pt
```

Now `STITCHED_MEAN` in the val output *is* meaningful (all 3 chain steps
together cover the full source) — that's the number to track, along with
per-chain-step `span/MEAN` broken out (chain step 2, the 2-hop case, is
historically the hardest).

---

## 3. Moving to `weave`: varied trajectory shapes

`kvmem/configs/hmn_weave_mix.py` — same architecture size as `hop`, but
instead of one fixed query schedule, every training step samples one of
three trajectory *shapes* (`batch`, `stream`, `interleave_delayed` —
uniform weight, see `HMN_RECIPE.md`'s DSL table). Warm-started from the
same `c64` checkpoint as `hop` (not from `hop` itself — see the config's
own docstring for why: `hop`'s checkpoint is a weaker, non-reproduced run).

```bash
python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_mix.py --device mps
```

(`_pretrained_ckpt` is already set inside this config, pointing at
`logs/hmn_single_recall_c64/checkpoints/stage0_best.pt` — no `--pretrained`
flag needed unless you want to override it.)

Same log format as before, 160,000 steps. Because every step draws a
different trajectory shape, per-pattern performance can diverge — a
checkpoint might do well on `batch` (the shape `hop` itself trained on) and
worse on `stream`/`interleave_delayed` early on.

### The RNN-forced variant: `hops=1`

`kvmem/configs/hmn_weave_mix_accum_rnn.py` — identical trajectory mix, but
sets `hops=1` explicitly. This blocks every query past the first from
attending to *any* encoding-pass STATE directly, so the single-hop relay
becomes the model's only channel to anything beyond its own local query —
a much harder, more RNN-like constraint (`state_t = f(state_{t-1},
query_t)`) than `weave_mix`'s default unbounded (`hops=-1`) routing.

```bash
python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_mix_accum_rnn.py --device mps
```

If you want to sanity-check this mechanism cheaply before committing to the
full 160,000-step run, `kvmem/configs/hmn_accum_rnn_sanity.py` is a
much smaller (`d=16, n_layers=4`), from-scratch, 20,000-step version of the
same `hops=1` + weave_mix setup — same relay mechanism under test, just
sized to finish in minutes instead of hours:

```bash
python3 -m kvmem.hmn --config kvmem/configs/hmn_accum_rnn_sanity.py --device mps
```

---

## 4. Held-out generalization eval: `eval_weave.py`

This is the real test of whether the relay generalizes, run *after*
training against a saved checkpoint — separate from the in-training val
blocks, and using trajectory patterns (`repeat_query`, `decay_curve`,
`long_hop_recovery`) that are deliberately **never trained on**.

```bash
python3 -m kvmem.eval_weave --ckpt logs/hmn_weave_mix/checkpoints/stage0_best.pt \
    --device mps --patterns batch,stream,interleave_delayed,repeat_query
```

Key flags (`kvmem/eval_weave.py`'s `main()`):

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | required | checkpoint path |
| `--patterns` | `batch,stream,interleave_delayed,repeat_query` | comma-separated; add `decay_curve` for the pure-decay probe |
| `--n-chunks` | 4 | use 8 for `long_hop_recovery` — stresses more hops than any current training used |
| `--chunk-len` | 16 | must match the checkpoint's training shape |
| `--n-seqs` | 3 | held-out test sequences per pattern |
| `--noop-hops` | `1,2,4,8` | hop counts to sweep for `decay_curve` |

**What to read in the output**: for `repeat_query`, look at the printed
`first=...% repeated=...% drop=...pp` line — a large drop means the
relay lost information between the first and repeated query of the same
span. This is the exact test that caught `hop`'s clean 0% recovery failure
(see `CLAUDE.md`'s results section) and the thing `weave_mix`/`accum_rnn`
are trying to improve.

```bash
# stress-test with more hops than any training run used
python3 -m kvmem.eval_weave --ckpt logs/hmn_weave_mix/checkpoints/stage0_best.pt \
    --device mps --patterns long_hop_recovery,decay_curve --n-chunks 8
```

---

## Summary: the progression

```
c64 (hmn_single_recall_c64.py)           — 1 chunk, no routing, no relay, simplest, from scratch
  -> c128 (hmn_single_recall_c128.py)    — same task, chunk_len doubled
  -> hop (hmn_recall_queue.py)           — 4 chunks, 3 fixed chain steps, hops=1 relay
  -> weave_mix (hmn_weave_mix.py)        — varied shapes, unbounded (hops=-1) routing
  -> weave_mix_accum_rnn(_sanity)        — varied shapes, hops=1 forced recurrence
```

`hop`, `weave_mix`, and `weave_mix_accum_rnn` all warm-start directly from
`c64`'s checkpoint, not from each other or from `c128` — `c128` is its own
side branch of the `chunk_len` ladder.
(`hmn_routing_4to1_state.py`/`solo`, an earlier bootstrap stage, is treated
as an archived experiment with no checkpoint on disk and is no longer part
of this pipeline.) `accum_rnn_sanity` is the only other from-scratch config.
Train with `python3 -m kvmem.hmn`, watch `logs/<name>/train.log` for
in-training val, then run `python3 -m kvmem.eval_weave` against the
finished checkpoint for the held-out generalization patterns training
never saw.

See [`CLAUDE.md`](../CLAUDE.md) for what's actually been run and the real
numbers each stage produced.
