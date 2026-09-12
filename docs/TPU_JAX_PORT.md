# TPU port history — torch_xla (abandoned) → JAX/Flax NNX (current)

**Moved verbatim from `CLAUDE.md`'s old "## TPU port" section (2026-09-06).** This is the full forensic narrative of porting kvmem's training loop onto TPU hardware: first via `torch_xla` (abandoned — see below), then via a from-scratch JAX/Flax NNX port (`kvmem/hmn_jax.py`), which is the current and only supported TPU path.

**Read this first if you're new to the TPU side of this project**: `torch_xla` is DEAD for this project — real, reproducible, never-root-caused instability (bugs 5/6/7 below: a data-dependent NaN independent of every hyperparameter ablated, and a separate RoPE-at-long-length NaN that persisted through every fix tried) is why the project moved to JAX entirely. `kvmem/hmn.py`'s own torch_xla-specific plumbing (`bucket_lengths`, `device_str='tpu'`, the bf16-autocast path) still exists in that file but should be treated as dead/frozen code kept for reference, not a path to extend further. All current and future TPU work goes through `kvmem/hmn_jax.py`. For day-to-day operational mechanics (ssh, gcloud commands, common failure modes) see [`docs/tpu_setup.md`](tpu_setup.md) and [`docs/tpu_direct_ssh.md`](tpu_direct_ssh.md) instead — this file is history, not a how-to.

---

## TPU port (2026-07-30) — status: infra confirmed working, first real training run not yet completed

**Standing rule: never create/recreate a TPU VM directly via `gcloud ... tpu-vm create` — even
under an autonomous-work mandate.** This project's TPU access is via **TRC (TPU Research Cloud)
free-tier quota**, which is consumed/tracked through `gcloud compute tpus queued-resources`, NOT
the direct `tpu-vm create` API — creating a VM directly bills normally instead of drawing on the
TRC quota. `tpu3` (`v6e-8`, `us-east1-d`, spot) was PREEMPTED (GCP spot reclaim) mid-run on
2026-07-31; `gcloud ... tpu-vm start` does not support restarting a PREEMPTED node, and an
attempt to `gcloud ... tpu-vm create tpu3 ...` to replace it was stopped by the user ("do not
create own tpu... must use `gcloud compute tpus queued-resources` else will be billed not using
TRC quota... not allowed to spin own vm"). Provisioning (even via the correct queued-resources
path) is the user's call — report the loss and wait, don't self-serve a replacement, regardless
of how much autonomy has otherwise been granted for the training work itself.

**Context**: a scale-up experiment (target: 1024-byte perfect recall from a warmup anchored at
any source index, `d=128/n_layers=16/n_heads=8`, ~1.12M params — see
`/Users/muaz/.claude/plans/dazzling-waddling-widget.md` for the full plan) needed far more
throughput than MPS/CPU could give, motivating the actual TPU port `docs/TRC_TPU.md` had
previously only estimated. `docs/TRC_TPU.md` now has the up-to-date, corrected version of
everything below (tier confirmation, the packing-recommendation reversal, grad_checkpoint
correction) — this entry is the CLAUDE.md-level summary of what's confirmed working and what
broke, for quick reference.

**Access**: `gcloud compute tpus tpu-vm ssh tpu1 --zone=europe-west4-b` — confirmed `tpu1` is
`v5litepod-1`, ONE v5e chip (not a `-8` slice), `torch 2.6.0`/`torch_xla 2.6.1` preinstalled.
SSH is flaky/slow to connect (sometimes several retries, occasionally outright fails) WHILE the
TPU process is mid-XLA-compile and pegging most of the host's 24 vCPUs — this is contention, not
a real connectivity problem; retry rather than assume the VM is down. **Use tmux for anything
that must survive a dropped SSH session** — every run in this project's TPU work goes through a
persistent tmux session (`tmux new-session -d -s kvmem_gate`, `tmux send-keys ... Enter`, `tmux
capture-pane -t kvmem_gate -p` to read output) rather than a bare `--command`, specifically
because a long-running training job must not die when a flaky SSH connection drops.

**Fix for the flakiness itself, for one-off status-check commands (separate from the tmux point
above, which is about the training job surviving a drop)**: each `gcloud compute tpus tpu-vm ssh
--command=...` invocation is a brand-new SSH handshake + gcloud auth/IAM/IAP round-trip from
scratch, which collides badly with the host being CPU-starved during a compile — this is why
repeated status-check calls fail far more often than the one persistent tmux session does. Get
the real `ssh` invocation gcloud would run via `gcloud compute tpus tpu-vm ssh tpu1
--zone=europe-west4-b --dry-run` (prints something like `/usr/bin/ssh -t -i
~/.ssh/google_compute_engine -o HostKeyAlias=... muaz@<external-ip>`), then open ONE multiplexed
master connection directly with plain `ssh` and reuse it for every subsequent command instead of
going through `gcloud`'s wrapper each time:
```
ssh -o ControlMaster=auto -o ControlPersist=1h -o ControlPath=/tmp/tpu1_ssh/cm \
    -o CheckHostIP=no -o HashKnownHosts=no -o HostKeyAlias=<from dry-run> -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=~/.ssh/google_compute_known_hosts \
    -i ~/.ssh/google_compute_engine muaz@<external-ip> "echo CONNECTED"
# every later command reuses the same authenticated socket, no new handshake:
ssh -o ControlPath=/tmp/tpu1_ssh/cm muaz@<external-ip> "ps aux | grep hmn"
```
Verified this resolves the repeated-connection-failure pattern in practice. `ssh -O check -o
ControlPath=... <host>` confirms the master is still alive if a later command behaves oddly.

**Standing rule: after launching any TPU training job, always give both of these two commands**
(full explicit form, substituting the real host/`ControlPath`/run name/session name — no shell
alias, since aliases require editing files outside this repo), so the user can immediately watch
the job either as a scrolling log or as the live terminal:
```
ssh -o ControlPath=/tmp/<tpuN>_ssh/cm muaz@<ip> "tail -f -n 50 ~/kvmem/logs/<run_name>/train.log"
ssh -o ControlPath=/tmp/<tpuN>_ssh/cm muaz@<ip> -t "tmux attach -t <session_name>"
```
**These `-o ControlPath=...` short forms only work from wherever the multiplexed master socket
was actually opened** (verified: when the master was opened inside this assistant's own sandboxed
shell, the user's own terminal got `Permission denied (publickey)` trying to reuse that same
`ControlPath` — the socket file isn't visible across that boundary). For the user's own terminal,
give the `gcloud` form instead, which handles auth itself and needs no pre-existing socket:
```
gcloud compute tpus tpu-vm ssh <tpuN> --zone=<zone> --command="tail -f -n 50 ~/kvmem/logs/<run_name>/train.log"
gcloud compute tpus tpu-vm ssh <tpuN> --zone=<zone> -- -t "tmux attach -t <session_name>"
```

**One sharp edge hit directly**: a command that kills a large, actively-compiling process on the
remote end (e.g. `pkill -9` against the training PID) can itself return a spurious immediate
`exit 255` on the multiplexed channel even though the master session survives and the kill
actually landed — don't read that as "the connection is broken," re-check with a plain command
(`ps aux`) on the same socket before concluding anything failed.

**Monitoring**: `~/.local/bin/tpu-info` (already installed) shows chip/PID/HBM-usage/duty-cycle —
useful for confirming which process holds the device and current HBM usage. **Caveat found
directly**: running it under `watch` in a second tmux session (`tmux new-session -d -s
tpu_monitor`) appears to STALL/freeze (stale timestamps, no refresh) while the training process
is mid-compile — contention between `tpu-info`'s own metrics query and the busy compile. Don't
trust its wall-clock freshness during a compile; use `ps aux`'s CPU-time field on the training
PID instead (climbing steadily = genuinely still working, not hung) as the reliable liveness
signal during that phase.

**Porting work landed in `kvmem/hmn.py`** (`train()`, opt-in via `hp['bucket_lengths']`/
`device_str='tpu'` — every existing CPU/MPS config unaffected): length bucketing + padding
(`_bucket_ceilings`/`_pad_mask_to`/`_pad_tok_to`, a weighted k-segment DP minimizing `L^2`-weighted
cost, since attention cost scales with `L^2` not `L`), per-bucket batch sizing from TWO memory
ceilings (`token_budget` for the `B*L` term, `attn_sq_budget` for the `B*L^2` attention-matrix
term — see the grad_checkpoint finding below for why the second one is load-bearing, not
optional), `torch_xla.sync()` + bf16 autocast, host-sync-throttled loss/logging (avoids a device
round-trip every step), a CPU eval replica (`_synced_eval_model()` — autoregressive decode is a
token-at-a-time Python loop over a growing shape, a recompile-per-token disaster on XLA, so eval
copies weights to a CPU copy of the model instead of porting decode), and a vectorized (no more
per-`b_idx` Python loop) `make_batch_tagged`. Verification harness: `kvmem/gate_check.py`
(gates 3/4/5 — CPU/TPU loss-curve parity, bf16-vs-fp32 byte-exact match, real-config end-to-end
smoke test), run as `python3 -m kvmem.gate_check <gate3_cpu|gate3_tpu|gate3_compare|gate4|gate5>`.

**Four real bugs found and fixed, plus a fifth still open (see below), all confirmed on `tpu1`
directly (not theoretical)**:
1. **Mixing a CPU `.backward()` call and a TPU `.backward()` call in the SAME Python process
   crashes the second one** — `RuntimeError: 0 <= device.index() && device.index() <
   ... device_ready_queues_.size() INTERNAL ASSERT FAILED`. PyTorch's autograd Engine singleton
   sizes `device_ready_queues_` when first used; if that first use is a CPU backward, it never
   learns about XLA registered afterward. Not specific to this codebase. **Fix: one device per
   process** — `gate_check.py`'s `gate3_cpu`/`gate3_tpu` are separate `python3 -m` invocations,
   compared only via their logged output on disk, never in-process.
2. **`torch.utils.checkpoint.checkpoint`'s default (`use_reentrant=False`) path is incompatible
   with XLA tensors** — `AttributeError: module 'torch' has no attribute 'xla'`, because it calls
   `getattr(torch, device_type)` to save/restore per-device RNG state, and `torch_xla` doesn't
   register itself under `torch.xla`.
3. **Gradient checkpointing is NOT optional at long `L`, regardless of how small the model is** —
   without checkpointing, training the ~1.12M-param model at `B=64, L=1232` hit a hard HBM OOM:
   `Used 52.85G of 15.75G hbm`. The requested amount matches `B*H*L^2*n_layers*4bytes`
   (`64*8*1232^2*16*4 ≈ 52.8G`) almost exactly — every layer's `O(B*H*L^2)` attention-score matrix
   was being retained simultaneously for backward. Recomputing one layer's activations at a time
   instead fixes it. The lesson generalizes: whether checkpointing matters is a function of `L`
   (quadratic term) vs. model size (linear term), NOT primarily a function of param count the way
   `docs/TRC_TPU.md`'s original (now-corrected) guidance assumed.
4. **`torch_xla.utils.checkpoint.checkpoint` (the initial fix for bug 2) silently breaks under bf16
   autocast — trains "successfully" all the way to `loss=NaN` from the very first logged step,
   never crashing.** It doesn't reapply the surrounding `torch.autocast` context during backward's
   recompute the way stock PyTorch's reentrant `CheckpointFunction` does. Found by direct A/B on
   `hmn_tpu_sanity_w25.py`: `grad_checkpoint='block'` (via `torch_xla.utils.checkpoint`) → NaN from
   step 1; `grad_checkpoint=False` (no checkpointing at all, same everything else) → loss
   5.33→5.39, finite, over 600 steps. A local CPU repro of the same architecture under forced bf16
   autocast — both with and without `torch.utils.checkpoint` — never produced NaN either, ruling
   out autocast or checkpointing individually and isolating the interaction specifically to
   torch_xla's implementation. **Real fix** (`kvmem/hmn.py`'s `_ckpt`): for XLA tensors, use stock
   PyTorch's REENTRANT path instead — `torch.utils.checkpoint.checkpoint(fn, *args,
   use_reentrant=True, preserve_rng_state=False)`. `CheckpointFunction.backward` explicitly
   reapplies `torch.amp.autocast(device_type=ctx.device_type, **ctx.device_autocast_kwargs)`
   around the recomputed forward — the handling torch_xla's version lacks. `preserve_rng_state=
   False` is required too: it's what gates the `_get_device_module`/`getattr(torch, 'xla')` call
   from bug 2 (safe here — no dropout/stochastic ops in any checkpointed block). One more wrinkle:
   even with `preserve_rng_state=False`, `torch.random.fork_rng` (called unconditionally inside
   `CheckpointFunction.backward`, before it checks its own `enabled` flag) still does `getattr(
   torch, 'xla', None)` and raises if that's `None` — fixed by `torch._register_device_module(
   'xla', torch_xla)` once at import time (any non-None object satisfies it; with `enabled=False`
   nothing downstream actually touches it). Verified: re-running `hmn_tpu_sanity_w25.py` with
   `grad_checkpoint='block'` restored and this fix in place reproduced the SAME finite loss values
   (5.33→5.39) as the no-checkpoint run, at the same speed (~3.4-4 it/s) — checkpointing is now
   free at this scale, not just avoided.

**A fifth issue, still OPEN and NOT root-caused despite extensive ablation** (2026-07-30):
bug 4's fix resolved `hmn_tpu_sanity_w25.py`'s stage 0 (`chunk_len=8`, finite loss 5.33→5.39,
`best=3.1%` match, clean eval) — but **stage 1 (`chunk_len=16`) hit `loss=NaN` again, from step 1**.
A long sequence of single-variable ablations followed, each built as a genuine positive/negative
pair on `tpu1` directly (never reproduced on CPU under any settings, including forced bf16
autocast and the exact reentrant-checkpoint code path) — **every one of the following was
individually ruled out as the sole cause**:
- **Real padding** (a bucket mixing different real `L` under one ceiling) — a config with a
  single bucket genuinely forced to pad (`max_shape_buckets=1`, 3 entries with real `L=19/20/21`
  merged into one `Lb=21`) NaN'd; the exact same 3 entries with `max_shape_buckets=3` (each gets
  its own exact bucket, verified `waste=0.0%` on every bucket) **also NaN'd** — padding is not
  necessary for the failure.
- **`grad_checkpoint='block'`** — set to `False` on an otherwise-identical config: still NaN'd.
- **bf16 autocast** — added `hp['no_autocast']` (forces fp32 via `torch.autocast(...,
  enabled=False)`, `kvmem/hmn.py`'s weave_mix forward) and reran: **still NaN'd even in fp32**.
  This alone rules out precision as the cause, contradicting the working hypothesis at the time.
- **`rope`/`state_vocab_size`** — swapping `rope=False, state_vocab_size=1` (the scale-up
  target's settings) for `rope=True, state_vocab_size=2` (every historically-proven-working
  config's settings) on the same shape: still NaN'd.
- **Batch size** — `B=4096` (the value used throughout `hmn_tpu_sanity_w25.py`) vs `B=64` on the
  ORIGINAL known-good 6-entry stage-0 weave_mix (unmodified from the config that trained cleanly
  earlier): `B=4096` finite and declining (confirmed twice, including a fresh re-run late in the
  investigation confirming the environment itself had not degraded from repeated `pkill -9`s),
  **`B=64` NaN'd from step 4** — the one result that looked like a real, single-variable
  correlation.

**Why even the batch-size result is not trustworthy as a root cause**: changing `B` changes how
many values `rng.integers`/`rng.beta` draws per batch-construction call in `make_batch_tagged`,
which shifts the ENTIRE subsequent NumPy RNG stream from the very first batch onward — `B=4096`
and `B=64` runs are not "the same data, fewer rows," they diverge into completely different
random draws immediately. Every ablation above has this same confound: each config edit was
also, unavoidably, a different RNG stream. **Net honest conclusion**: this looks like a rare,
data-dependent numerical edge case specific to real XLA/TPU execution (bf16 OR fp32 — precision
doesn't gate it) that no single hyperparameter reliably triggers or avoids — some specific random
batch draws hit it, others don't, across every setting tried. The next step that would actually
localize this (not yet done) is forward hooks checking each block's output for NaN/inf at a FIXED
seed, to find exactly which layer and which row first goes non-finite, rather than continued
hyperparameter-level ablation. **`tpu1` was shut down at the end of this investigation — no
further TPU work has happened since.** `kvmem/configs/hmn_tpu_sanity_w25_ablate*.py` (three
variants: `_ablate`, `_ablate_2`, `_ablate_3`) and `kvmem/configs/hmn_tpu_recall1024_flat.py`
are all still `rope=False`/`state_vocab_size=1`-based and untouched since. **Do not re-attempt
Run A until this is resolved** — its own buckets will mix real lengths, hitting the identical
open failure mode.

**JAX/Flax NNX port, and the finding that actually answers bug 5** (`kvmem/hmn_jax.py`,
2026-07-30): `torch_xla` is one bridge among several onto XLA; JAX is XLA's own first-party
frontend, built independently — a genuinely different data point on whether bug 5 is a
`torch_xla`-bridge-layer bug or something XLA itself does with this exact computation.
**Single file, fully self-contained** (no import of `kvmem.hmn`, no `torch` at all) —
`chunk_positions_traj`/`chunk_mask_fb_traj`/`parse_traj_dsl`/`make_batch_tagged` are copied
byte-for-byte (pure NumPy/Python, no torch involved in any of them) rather than imported, so the
file has zero PyTorch dependency. Scope: only `block_type='single_attn'` with `rope`+`yarn`/
`null_kv`/`rmsnorm` — `hmn_notags_w25_rope.py`'s exact feature set. `build_model(hp, rngs) ->
HMNModel` mirrors `kvmem.hmn.build_model`'s own signature; `train_jax(hp)` is a genuine (if
scope-limited — no refine rounds, no padding/bucketing, no label smoothing, no decode-eval)
optimization loop: weighted trajectory sampling, real gradient steps via
`nnx.value_and_grad`/`optax.adamw`, teacher-forced NLL loss only.

**One real bug caught while porting, not yet flagged elsewhere in this codebase**:
`kvmem.hmn.MHAttention`'s own docstring claims `null_kv`'s null K/V pair is "learnable," but the
actual `forward()` code constructs it as a fresh `torch.zeros(...)` every call, never wrapped in
`nn.Parameter` — it can never receive gradients and is permanently zero, contradicting the
docstring (now corrected in `kvmem/hmn.py`'s own docstring, behavior left unchanged since no
checkpoint has ever exercised a learned null slot). Caught by a 1024-param mismatch (166,400 vs
165,376) between the JAX port (which initially matched the docstring) and the real PyTorch model,
found by comparing param counts directly, not by inspection.

**Two flax-API version mismatches hit and fixed, both real portability bugs, not TPU-specific**:
newer flax (verified 0.12.8) requires wrapping a plain Python list of submodules in `nnx.List`
(a bare list now raises `ValueError: ... Static attributes should not contain data values`);
older flax (0.10.7 — the newest installable on a TPU VM still shipping Python 3.10, since
flax>=0.11 requires Python 3.11+, both verified directly) predates `nnx.List` entirely and just
accepts a bare list. `nnx.Optimizer.update`'s signature also changed — newer takes `(model,
grads)` positionally, older takes just `(grads)` with the model reference stored at `__init__`.
Both detected at runtime (`hasattr(nnx, 'List')`, `'model' in inspect.signature(...).parameters`)
rather than pinned to one version — **first attempt at the second one used a parameter-COUNT
check instead of a name check, which was wrong** (`**kwargs` inflates both signatures' arg count
equally, so count alone doesn't discriminate) and produced the exact same crash again on
`tpu2` — fixed by checking for the `'model'` parameter name specifically.

**`kvmem/setup_tpu_jax.sh`**: one-shot install script for a fresh TPU VM (`pip install
'jax[tpu]' -f <libtpu index> flax optax tpu-info`, no flax version pin — pinning one broke the
install outright on `tpu2`'s Python 3.10 instead of degrading gracefully) plus a self-check that
`jax.devices()` actually returns a TPU device. Verified on `tpu2` (`v6e-1`, Trillium,
`europe-west4-a`, a fresh VM with zero ML packages preinstalled — 44 vCPU/172GB host, notably
larger than `tpu1`'s 24/47) from a cold start.

**The actual finding**: with `kvmem/hmn_jax.py` running cleanly on `tpu2` (real gradient steps,
finite loss, both stage 0 no-padding and general training confirmed), `torch_xla` was ALSO
installed on `tpu2` (`pip install torch~=2.6.0 torch_xla[tpu]~=2.6.0`, same versions as `tpu1`)
and `hmn_tpu_sanity_w25_ablate_2.py` — the exact config that reliably produced bug 5's NaN on
`tpu1`/v5e (genuine padding via `max_shape_buckets=1` forcing 3 different real lengths into one
`Lb=21` bucket, `rope=False`, `state_vocab_size=1`, `grad_checkpoint='block'`, bf16 autocast) —
was run unchanged on `tpu2`/v6e. **It completed all 100 steps with ZERO non-finite loss values**
(`[stage 0] done.`, final losses in the 5.2-5.5 range throughout, vs. instant `loss=nan` from
step 1 on every v5e attempt). **This is strong evidence bug 5 is specific to the v5e chip
generation (or its particular libtpu/PJRT build), not a generic torch_xla bug, not this
architecture's masking/padding logic, and not any of the hyperparameters ablated earlier** (all
of which were tested on v5e only). Not yet fully conclusive — only one config variant has been
re-tested on v6e so far (not, e.g., `hmn_tpu_recall1024_flat.py`'s much longer `L`), and "clean
for 100 steps" is not as strong as the multi-thousand-step confirmation bug 5 itself needed to
surface reliably — but this is the first actionable lead after a full day of inconclusive
same-hardware ablation.

**A sixth, DIFFERENT bug found immediately after, on the SAME chip (`tpu2`/v6e) — `rope=True` +
bf16 autocast NaNs; fp32 fixes it cleanly.** Once `hmn_tpu_sanity_w25.py` (NoPE, `state_vocab_
size=1`, `lr_max` corrected to `1e-4` — see below) was training on `tpu2` with real, healthy
progress (match 21.7% at step 5000, loss monotonically declining past step 10000), a clone with
only `rope=True` changed (`hmn_tpu_sanity_w25_rope.py`, everything else identical: same `lr_max`,
`grad_checkpoint='block'`, bf16 autocast, `B=16`, no padding — every bucket `waste=0.0%`) hit
`loss=nan` on ALL 6 trajectories from step 1. Setting `hp['no_autocast']=True` (forces fp32,
the same escape hatch built for bug 5) on that exact config fixed it immediately — loss finite
and declining smoothly (4.627→2.981 over 4400 steps, no NaN anywhere). **This is mechanistically
distinct from bug 5**: bug 5 turned out to be a v5e-hardware/libtpu issue independent of
precision (fp32 didn't fix it there, and it disappeared on v6e regardless of precision); this one
is a genuine bf16-precision issue specific to RoPE (`rope=True`) that reproduces even on v6e
where bug 5 doesn't — plausible mechanism is accumulated phase/rotation error in bf16's ~8-bit
mantissa compounding across the `sin`/`cos` position-angle computation
(`kvmem.hmn.apply_rope`), a classically bf16-sensitive operation, unrelated to chip generation.
**Not yet deeply isolated beyond the fp32 fix** (didn't test whether `grad_checkpoint`/batch size
matter here the way they were ruled out for bug 5) — fp32 is a working, if unoptimized,
workaround; any future `rope=True` TPU run should set `no_autocast=True` until this gets a
proper mechanistic fix (e.g. computing `apply_rope`'s `cos`/`sin` in fp32 even under an
otherwise-bf16 autocast region, a much narrower and cheaper fix than disabling autocast
entirely).

**`lr_max` also needed correcting for `hmn_tpu_sanity_w25.py`'s real convergence attempt**:
carried over unexamined from Run A's large-batch √-scaled value (`6e-4`), it produced a fast
initial drop (loss 4.5→2.5 by step 1000) followed by plateau/oscillation (2.2-2.9 for the next
4000 steps, match=2.3% at step 5000) instead of continued convergence. Reverting to
`hmn_notags_w25.py`'s original `1e-4` (the value that config actually converged under, per
CLAUDE.md's own chunk_len-ladder results) fixed it — smooth monotonic loss decline, match=21.7%
at the same step-5000 checkpoint (vs. 2.3% at the wrong LR), continuing to decline past step
10000 (match wobbled 21.7%→17.4%, plausibly eval noise from the tiny `val_n_seqs=3` sample —
loss kept improving monotonically through that same window, and CLAUDE.md's own `weave_c64`
entry documents an identical wobble-not-degradation pattern elsewhere).

**Bug 6 turned out to be much bigger than the fp32 fix suggested — a SEVENTH issue, length-
dependent, isolated down to "XLA-compilation-specific" and still OPEN.** Once the fp32 fix looked
clean at sanity scale (`hmn_tpu_sanity_w25_rope.py`: match=50.1% at step 5000, more than double
NoPE's 21.7% at the same step — confirming RoPE's known advantage holds once precision is
handled), the natural next step was re-testing Run A's real config with `rope=True`. A direct
clone (`hmn_tpu_recall1024_flat_rope.py` — `rope=True, yarn=True, no_autocast=True,
L_train=2200, L_max=8192`, otherwise identical to `hmn_tpu_recall1024_flat.py` including the
OOM-driven `max_shape_buckets=4`/`attn_sq_budget=31_000_000` fix) hit **`loss=nan` across every
single entry** in its own 30-step gate-5-style smoke test — at Run A's real scale (`L=1232-2128`),
`no_autocast=True` did NOT fix it, unlike at sanity scale. A systematic single-variable ablation
followed, same pattern as bug 5's own investigation:
- **`yarn=False`** (removes YaRN's interpolation ramp entirely, plain unscaled RoPE frequencies)
  — still NaN, every entry. Rules out the YaRN ramp formula.
- **`grad_checkpoint=False`** (plus `attn_sq_budget` cut ~16x to `2_000_000` to compensate for no
  longer checkpointing) — still NaN, every entry. Rules out checkpointing.
- **Direct on-device component test**: ran `kvmem.hmn.apply_rope` and raw `torch.sin`/`torch.cos`
  directly on a real TPU tensor at the exact failing scale (`pos` up to 2127, the freq=1 channel
  — angle up to ~2127 radians) — **all finite, no NaN**. Rules out RoPE's own trig computation as
  the mechanism, even at this position magnitude.
- **CPU reproduction, the decisive test**: ran the EXACT same config (`rope=True`, `L` up to 2128,
  real data pipeline, `B=2`, 10 steps) via `device_str='cpu'` (eager PyTorch, no XLA at all) —
  **loss finite throughout** (5.56-5.59, zero NaN). The identical architecture, identical `L`,
  identical RoPE math trains cleanly off-XLA.

**Net conclusion (2026-07-30, at the time)**: real, reproducible, isolated to XLA's COMPILED
graph specifically — not RoPE's math (fine in isolation on-device AND in the full CPU pipeline),
not YaRN, not checkpointing, not batch size, not raw position magnitude. This was a different
flavor of "XLA does something CPU/component-testing can't catch" than bug 5 — it persisted on the
SAME chip (`tpu2`/v6e) that bug 5's own config trained cleanly on, so it was specifically about
`rope=True` at long `L` in torch_xla's compiled graph, not chip generation.

**Bug 7 RESOLVED — by switching frameworks, not by finding the torch_xla root cause.** Given how
long bug 5 and bug 7 both took to (partially) pin down on torch_xla, the next move was testing
whether `kvmem/hmn_jax.py` — independent XLA lowering, no torch_xla bridge layer — sidesteps this
family of bug entirely, rather than continuing to dig into torch_xla's compiler internals.
Sequence: (1) `hmn_tpu_sanity_w25_rope.py` run via `kvmem.hmn_jax` (plain fp32, no bf16 autocast
in this port at all) trained cleanly at sanity scale — expected, since bug 6 was already known to
be a bf16-specific issue. (2) The real test — `hmn_tpu_recall1024_flat_rope.py` (Run A's own
scale, `L=1232-2128`, the exact config that reliably NaN'd on torch_xla regardless of every lever
pulled) run via `kvmem.hmn_jax` (after fixing a real bug in `train_jax` itself: it hardcoded
`n_chunks=1` in its `make_batch_tagged` call, silently correct for every `_w25*`-style config
tested so far but wrong for Run A's `n_chunks=16` — fixed by threading `_build_trajectory`'s own
computed `len(pos_content['enc_blocks'])` through as `traj['n_chunks']`) — **hit an HBM OOM
first** (`Used 52.64G of 31.25G hbm`, since `hmn_jax.py` had no `grad_checkpoint`/bucketing yet at
that point, so it inherited Run A's `B=64` un-checkpointed), **then, at `B=4`, completed all 20
steps with FINITE loss throughout** (5.5752→5.5533, `[stage 0] done.`). **This is the decisive
result**: the identical `rope=True` config at the identical scale that reliably NaN'd on
torch_xla — including every yarn/checkpoint/precision variant tried — trains cleanly on JAX.
`kvmem/hmn_jax.py` is therefore the working path for `rope=True` at Run A's scale; torch_xla's
own root cause for bug 7 remains formally unexplained (not worth continuing to chase now that a
working alternative exists), but is functionally closed for this project's purposes.

**`kvmem/hmn_jax.py` brought to full feature parity with `kvmem.hmn`'s own `train()`, within this
file's existing scope (single non-refine Q per entry), same day** — previously loss-only:
- **`nnx.jit`-compiled training step** — one compiled step function per trajectory (built once,
  cached on `traj['step_fn']`, `w0`/`c1` closed over as Python constants so the loss slice uses
  plain indexing rather than `jax.lax.dynamic_slice`) — **~60x speedup** at sanity scale (1.4 → 85
  steps/sec) once the per-shape compile cache warms up; loss values track the pre-jit run almost
  exactly (5.5587→5.2666 vs 5.5585→5.2662 at the same steps), confirming jit changed only speed.
- **KV-cache** (`HMNModel.__call__`'s `past_kv`/`return_kv`/`offset` now mirrors `kvmem.hmn.
  HMNModel.forward`'s signature exactly) and **`remat`** (`nnx.remat`, JAX's gradient-checkpoint
  transform, the counterpart to `grad_checkpoint='block'`) — both added to `MHAttention`/
  `SingleAttnBlock`/`HMNModel`. One real bug caught immediately: `nnx.remat` traces ALL positional
  args as dynamic by default, but `offset` feeds `jnp.arange(offset, ...)` inside `apply_rope`
  (needs a concrete Python int) and `return_kv` gates a Python-level `if` — both need `static_
  argnums`; without it, `ConcretizationTypeError` on the very first backward pass. Fixed via
  `nnx.remat(_block_call, static_argnums=(4, 5))`. Verified on CPU: forward+backward through
  `remat` gives finite loss and finite grads; KV-cache first-call + incremental-call (with a
  `null_kv`-padded mask) both verified correct.
- **`ar_decode_traj_nokv`/`ar_decode_traj_kv`** — ported eval, restricted (like the rest of this
  file) to the single-non-refine-Q case. `_nokv` is the direct port of `kvmem.hmn.ar_decode_traj_
  nokv` (full recompute per generated byte, matches what `train()` itself uses for its own
  `val/weave/*` numbers — deliberately NOT jitted, since the growing-sequence-length loop would
  retrace every token). `_kv` is NEW (not a port — `kvmem.hmn`'s own KV-cached decoders target
  other position layouts, not `chunk_positions_traj`): encodes the fixed prefix once via
  `return_kv=True`, then grows the cache one token at a time — mathematically identical greedy-
  argmax result to `_nokv`, much faster for long generations. `train_jax`'s own periodic eval uses
  `_kv`.
- **`make_test_sequences`** (copied verbatim) and **`save_checkpoint`/`load_checkpoint`**
  (pickle + numpy, not `torch.save`/orbax — no new dependency, and the two frameworks' checkpoints
  were never going to be interchangeable regardless of format) round out the `stage{i}_last/
  best/end.pt` pattern and `val/weave/*` + `MEAN` + `by_chunk_len` logging, matching `train()`'s
  own format line-for-line.
- **Verified end-to-end on both CPU and real TPU hardware (`tpu2`)**: training (jit+remat) + eval
  (KV-cached decode, real match% output) + checkpoint save, all in one run, no errors, checkpoint
  files confirmed written and independently reloadable.
- **Log format brought to parity with `kvmem.hmn`'s own `train()` a second time (2026-07-31)**:
  `train_jax()` now writes `train.log` (renamed from `train_jax.log` — matches `train()`'s own
  filename exactly, so tooling/aliases don't need a JAX-specific path), `train.jsonl` (per-
  `log_every`-step `{step, stage, loss, lr, entry}` record), and a live-updating
  `train_status.log` via a copied `_StatusWriter` (truncate-and-rewrite so `tail -f` shows a
  single live-updating tqdm line rather than growing unboundedly). Training loop now drives a
  real `tqdm` progress bar (`file=status_file`, `dynamic_ncols=True`) instead of a bare Python
  `range()` loop, with `pbar.set_postfix(loss=..., lr=..., entry=...)` updated every `log_every`
  steps and `str(pbar)` appended to `train.log` — same pattern `train()` itself uses. Scope note:
  unlike `train()`'s own jsonl (which logs one `traj_loss` value per trajectory in the mix),
  this file's jsonl logs only the single trajectory actually sampled that step (`entry`) — no
  per-trajectory EMA-loss bookkeeping was added, since nothing in this file's current scope reads
  it back.

**Given this, `kvmem/hmn_jax.py` is now the recommended path for any `rope=True` work at Run A's
scale** — `hmn_tpu_recall1024_flat.py`'s original `rope=False` torch_xla config remains a valid,
still-untested-post-OOM-fix fallback (`max_shape_buckets=4`/`attn_sq_budget=31_000_000`, found
necessary via a real `RESOURCE_EXHAUSTED` at `Lb=1744/B=32`, never re-verified since), but is no
longer the only option. `hmn_tpu_recall1024_flat_rope.py`/`_noyarn.py`/`_noyarn_nockpt.py`
(the torch_xla ablation trio) stay in the repo as the investigation record.

**`kvmem/hmn_jax.py` gained length bucketing, a persistent XLA compilation cache, and a
re-derived segmented forward — all opt-in, none change existing configs' behavior (2026-07-31).**
- **Length bucketing** (`hp['bucket_lengths']`) — direct port of `kvmem.hmn`'s own
  `_bucket_ceilings`/`_assign_bucket`/`_pad_mask_to`/`_pad_tok_to`/`_pow2_floor` (pure NumPy,
  copied verbatim). Unlike the torch version's per-trajectory-but-padded design, the JAX port
  shares ONE `nnx.jit`-compiled step function across every trajectory in a bucket
  (`_make_train_step_bucket`) — different trajectories sharing a bucket have different real
  `c1` (out_len varies with anchor), so the loss can't be a fixed-size Python slice; instead
  it's computed over the full static `[w0, Lb)` range and weighted by a per-trajectory
  `loss_mask` (traced array, 1.0 for real `w0<=pos<c1`, else 0.0) passed in at call time. This
  requires every trajectory in a stage to share the same `w0` — true for this file's
  single-query suffix-recall shape (asserted at setup, verified directly against
  `hmn_tpu_recall1024_flat_rope.py`: all 16 entries have `w0=1104`, 4 buckets from 8 distinct
  `L` values). Reduces Run A's distinct-shape compile count from 16 to `max_shape_buckets`.
  Per-bucket batch size is `b_cap = min(B, pow2_floor(token_budget/Lb), pow2_floor(attn_sq_
  budget/Lb**2))` — `token_budget` caps the linear-in-L cost (embeddings/FFN/residual-stream
  activations), `attn_sq_budget` caps the quadratic-in-L attention-score-matrix cost, the term
  that actually dominates at long `L`. **Worked numerical example against real v6e HBM** (31.25
  GiB usable per chip, confirmed via `tpu-info`): the shared torch config's own `attn_sq_
  budget=31_000_000` would cap `B` down to 4 at `Lb=2128` (`pow2_floor(31_000_000/2128**2)=4`)
  — but a real run of `hmn_tpu_recall1024_flat_rope_jax.py`'s exact architecture (`d=128/
  n_layers=16/n_heads=8`, ~1.12M params, `grad_checkpoint='block'`, fp32) at `B=64, Lb=2128`
  measured only 29.51/31.25 GiB HBM (steady state) — so that torch-era budget was calibrated
  for a different (more conservative, possibly torch_xla-specific) memory profile, not what
  JAX actually needs here. `hmn_tpu_recall1024_flat_rope_jax.py` recalibrates both:
  `token_budget=200_000` (> `64*2128=136,192`), `attn_sq_budget=320_000_000` (> `64*2128**2=
  289,816,576`) — both set just above the real calibration point so no bucket shrinks below the
  already-verified-safe `B=64`, leaving the remaining ~1.7 GiB as deliberate headroom (eval/
  checkpoint-save need extra memory too). **Caveat**: the 29.5 GiB figure includes params/
  optimizer-state/embedding overhead, not purely the attention matrix, so it is NOT a clean
  per-unit conversion factor to reuse for a different architecture — treat it as a single-point
  anchor, and re-verify via `tpu-info` after raising either budget rather than trusting the
  arithmetic alone (full worked-example docstring: `kvmem/hmn_jax.py`, right above
  `_bucket_ceilings`).
- **Persistent compilation cache** (`jax.config.update('jax_compilation_cache_dir', ...)`,
  defaults to `/tmp/jax_cache`, overridable via `JAX_CACHE_DIR`) — compiled executables survive
  process restarts, so a killed/relaunched run hitting the same bucket shapes again skips
  recompilation entirely. `jax_persistent_cache_min_compile_time_secs=1` avoids caching trivial
  sub-second compiles. Verified end-to-end (cache files written, ~2.2x speedup on a warm rerun
  at toy scale — real savings will be much larger at Run A's actual per-shape compile cost).
- **Segmented forward** (`hp['forward_granularity']`/`hp['segment_checkpoint']`) — JAX port
  AND re-derivation, not a straight port, of `kvmem.hmn`'s own `_iter_forward_segments`/
  `_forward_segmented` (TIME-axis gradient checkpointing across STATE-bounded segments, walking
  the packed sequence in groups with a carried KV cache instead of one dense forward pass —
  separate from and orthogonal to `grad_checkpoint`'s model-DEPTH checkpointing). **The torch
  version is currently unconditionally `NotImplementedError`-guarded** — its segment-boundary
  logic assumed the old pre-end-of-turn-STATE layout and breaks for any rec_block with its own
  trailing STATE commit (`sl0 is not None`, the non-terminal/relay case). This file's own scope
  (`_build_trajectory`'s assertion: exactly one terminal, non-refine 'initial' rec_block) never
  hits that case — a terminal query's `sl0` is always `None` — so `_iter_forward_segments_jax`
  is a narrower, independently-correct re-derivation: it segments ONLY the encode portion
  `[0, w0)` (verified contiguous, and verified `enc_blocks[-1]['sl1'] == rec_blocks[0]['w0']`
  and `rec_blocks[0]['c1'] == L`, i.e. no gap on either side); the query itself always gets its
  own dedicated final forward pass, never merged into an encode group. `_make_train_step_
  segmented` reconstructs the full loss by concatenating the LAST local position of the final
  encode group's own output (predicts token `w0`) with all-but-the-last of the query group's own
  output (predicts `w0+1..end-1`) — mathematically identical to what a single dense
  `model(tokens[:,:end], mask[:end,:end])` call's `logits[:, w0-1:end-1]` slice would give, since
  the KV-cache-carrying grouped calls are just that same dense computation split into pieces.
  `segment_checkpoint` wraps every group's own `model(...)` call (encode groups AND the final
  query call, uniformly) in `nnx.remat` — same `static_argnums` requirement as the existing
  model-depth remat (`offset` must be static, a Python int closed over per call site, never
  traced data). **Verified bit-exact against the dense path** (`loss` diff `0.00e+00` across
  granularity `1`/`4`/`16`/`1.0` and both `segment_checkpoint` settings) and gradients match to
  float32 noise (`2.98e-08` max abs diff) — the project's own standing rule ("verify masking
  changes against the actual attention-mask matrix, not just 'does it run'") applied here as a
  direct numerical comparison against the already-trusted dense computation. **Not currently
  needed for Run A** (`hmn_tpu_recall1024_flat_rope.py`'s HBM usage sits at ~29.5/31.25 GiB with
  bucketing alone, not OOMing) — available as an opt-in lever if a future config needs it, and
  asserted mutually exclusive with `bucket_lengths` for now (combining bucket-padding's variable
  `Lb`/`loss_mask` with segmented forward's own loss reconstruction wasn't needed yet and would
  add real complexity — not attempted).

**Depth-axis remat granularity, adaptive-mix sampling, compile-time instrumentation, and a
decode-jit rewrite that fixed a real ~35-minute eval bottleneck — all landed in `kvmem/hmn_jax.py`
the same session (2026-07-31), same "verify against the dense/eager baseline before trusting it"
discipline throughout.**
- **Depth-axis `grad_checkpoint` granularity** — mirrors `forward_granularity`'s own int/float
  duality onto the model-DEPTH checkpointing axis: `False` (none), `True`/`'block'` (per-layer,
  unchanged default), an int >=1 (exact layer-group size), or a float in `(0,1]` (fraction of
  `n_layers` per checkpoint group — `1.0` = whole stack as one group, still saves memory vs no
  checkpointing since only that group's own input is retained, just with the fewest/largest
  remat call-sites). `_grad_checkpoint_groups`/`_group_call`/`_group_call_remat`. **A real bug
  caught and fixed during this**: `build_model` was doing `grad_checkpoint=bool(hp.get(...))` —
  `bool(2)`/`bool(0.25)`/`bool('block')` are all `True`, silently collapsing every numeric/
  string granularity down to the coarsest per-layer grouping regardless of what was configured.
  Fixed by passing the raw value through unchanged. Verified bit-exact (loss AND full gradient
  tree) against the no-checkpoint baseline across `False`/`True`/`'block'`/`1`/`2`/`4`/`8`/
  `0.25`/`0.5`/`1.0` on a local CPU test, same seed/init every time.
- **Adaptive weave_mix reweighting** — JAX port of `kvmem.hmn`'s own `_adapt_reweight`/
  `_temp_softmax_rescale` (identical formula: harder-than-average trajectories scaled up, easier
  ones down, floor-blended so nothing drops below `adapt_floor`'s relative share). Motivated by a
  real gap: `hmn_tpu_recall1024_jax_adaptive_mix.py`'s 60-entry mixed-difficulty weave_mix (see
  below) was originally sampled at STATIC uniform weight forever — mixing easy-and-hard entries
  without ever shifting sampling effort toward whichever ones are still failing isn't meaningfully
  a curriculum, just wider static coverage. `adapt_signal='val_match'` (default) skips adapting
  until the 2nd eval (first reading is the noisiest). Verified the rescaling formula directly
  (uniform difficulty stays uniform, a harder entry gets upweighted, sum preserved) before
  deploying.
- **`weights`/`traj_loss` now logged to both `train.log` and `train.jsonl` every `log_every`
  step** (not just the sampled trajectory's own loss) — for plotting weight/loss evolution per
  entry over training. Per-trajectory loss materialized every step now (small added host-sync
  cost, traded for having the curve at all).
- **Per-shape compile-time instrumentation** (`[compile] step_fn for L=...` lines, keyed by
  `id(step_fn)` since bucketed/grouped step functions are SHARED across trajectories) — real
  compile times turned out much noisier than expected: same 60-shape training-step compile set
  measured 154s in one run and 994s in another, with individual shapes alternating between ~2s
  and ~20s in no clean pattern (ruled out monotonic cache-size growth as the explanation — fast
  and slow compiles interleave). Not root-caused; logged so future runs have the data rather than
  guessing.
- **`hmn_tpu_recall1024_jax*.py` config family renamed and made self-contained** —
  `hmn_tpu_recall1024_flat_rope.py` (the torch base config the JAX chain used to `load_config`
  from) was deleted from the working tree outside this session's own actions (found via `git
  status` showing it and several other torch configs as uncommitted deletions); `hmn_tpu_
  recall1024_jax.py` now inlines its own `hp` directly instead of depending on a file that may not
  exist. Chain: `hmn_tpu_recall1024_jax.py` (base, `B=64`) -> `_smallbatch.py` (`B=8, lr_max=1e-4`
  — the original `B=64, lr_max=6e-4` run was still at the random-baseline loss after 1000 steps,
  projected 46-60 HOURS to finish; `lr_max=6e-4` was √-scaled for a B≈256 target that was never
  actually deployed, `6e-4` is too high for the real `B=64`, and large `B` alone means far fewer
  updates/minute — this exact wrong-LR failure mode already happened once before, see the
  `hmn_tpu_sanity_w25.py` entry above) -> `_adaptive_mix.py` (extends the weave_mix to `n_chunks`
  in `{2,4,8,12,16}`, 60 entries total, `bucket_lengths=False` since `w0` scales with `n_chunks`
  — 5 distinct values, violating the bucket path's shared-`w0` requirement — plus `adaptive=True`).
  Renamed from `..._curriculum.py` mid-session: "curriculum" was misleading (no staged/sequential
  difficulty progression, just one stage with adaptive reweighting over a static mixed-difficulty
  set) — "adaptive_mix" names what it actually is.
- **Decode-jit rewrite — the big one, fixed a real ~35-minute eval bottleneck.** `ar_decode_
  traj_kv`'s eager, token-at-a-time Python loop scales badly once `weave_mix` has 60 heterogeneous
  entries (`out_len` up to ~2000): a live run's eval pass ran for over 35 minutes and still hadn't
  finished when checked. Root cause understood via a local CPU benchmark (tiny model, one
  `out_len=352` entry) comparing four approaches: **(A) eager baseline** 85-92s; **(B) naively
  jit the per-step forward call as-is** (growing `past_kv` via `concat`) ~62-68s extrapolated —
  barely better, because a DIFFERENT input shape every step forces a full XLA recompile every
  token, and B's "advantage" is just compiled matmuls beating eager dispatch despite constantly
  recompiling; **(C) fixed-size KV buffer** (`jax.lax.dynamic_update_slice` instead of `concat`,
  keeps every step's shapes IDENTICAL) **+ forward-pass-only jit, Python loop**: 0.73-0.39s
  (~118-236x); **(D) same fixed buffer + the WHOLE loop jitted via `jax.lax.fori_loop`** (zero
  Python-level dispatch inside the loop at all): 0.24-0.09s (~358-1000x, cached calls fastest).
  All four verified byte-identical match% against the eager baseline. Two real bugs caught while
  building this: (1) `apply_rope`'s `jnp.arange(offset, offset+L)` requires a CONCRETE `offset`,
  which would force a recompile every step even with a fixed buffer — rewritten to `jnp.arange(L)
  + offset` (mathematically identical, but `offset` can now be a TRACED scalar since only `L`,
  always static from shape, needs to be concrete) — existing static-offset callers verified
  bit-exact afterward; (2) the first `write_pos` (fixed-buffer) implementation on `MHAttention.
  __call__` returned the KV buffer AFTER `null_kv`'s extra column got concatenated onto it,
  silently growing the "fixed" buffer by one column every call and breaking the static-shape
  guarantee entirely — caught by a shape-mismatch crash on the SECOND decode step, not silently
  wrong; fixed by capturing the buffer before the `null_kv` branch. `MHAttention.__call__`/
  `SingleAttnBlock.__call__`/`HMNModel.__call__` all gained an optional `write_pos` param (default
  `None`, byte-identical to before — regression-checked via a bit-exact forward-pass comparison
  before AND after each of the two bug fixes above). `ar_decode_traj_kv_jit` (variant D) is now
  wired into `train_jax`'s own eval loop, replacing `ar_decode_traj_kv`; compiled decode programs
  are cached module-level (`_decode_jit_cache`, keyed by `(id(model), prefix_end, out_len)`) and
  reused across BOTH different trajectories sharing a shape AND repeated eval calls within one
  run — confirmed on a local smoke test (2.3s first eval including 6 compiles, 0.1s second eval,
  same 6 shapes, zero new compiles) and then on the real 60-entry/1.12M-param run: **first eval
  279.8s total** (compiling all 60 decode shapes for the first time) vs. the eager path's 35+
  minutes and still not done — roughly 7-8x faster even INCLUDING first-time compile cost, with
  every later eval expected to be dramatically faster still once nothing new needs compiling.
  `val/weave/decode_time_total` (+ per-entry `decode=Xs` alongside each match% line, + `train.
  jsonl`'s `eval_decode_total_s`/`traj_decode_s`) now logged every eval specifically to track this.
  Confirmed the caching itself is correct (not silently serving stale weights) via a direct local
  test: decode before vs. after 200 real training steps on the same cached compiled function gave
  different match% (0.57% -> 0.28%), proving `nnx.jit`'s per-call state-splitting reads current
  params every time — only the compiled PROGRAM is cached by `(id(model), prefix_end, out_len)`,
  never the weight values.

**Staged curriculum (`hmn_tpu_recall1024_jax_curriculum_staged.py`, 5 sequential stages `n_chunks`
2→4→8→12→16, each gated by `early_stop_mean=90.0`) — first real run surfaced two genuine bugs in
`train_jax` itself, both fixed (2026-07-31).** Stage 0 (n_chunks=2, easiest) peaked at val
MEAN=34.7% around step 16000-17000, then collapsed to ~1% over the next several evals with NO
warning in the logged training loss (which kept declining smoothly the whole time) — confirmed via
a direct decode comparison that this was NOT a decode-jit bug (eager vs jit agreed exactly on the
collapsed checkpoint). Root cause investigation found two real gaps:
- **No gradient clipping anywhere in `hmn_jax.py`** — `kvmem.hmn`'s own `train()` clips every step
  (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, immediately after `backward()`); the
  JAX port had plain `optax.adamw` with nothing composed in front of it. An unclipped outlier
  gradient is a textbook cause of exactly this failure signature (loss looks fine, generation
  quality collapses). Fixed: `hp['grad_clip_norm']` (default `1.0`, matching torch's hardcoded
  value) composes `optax.clip_by_global_norm` in front of `adamw` via `optax.chain`; set to
  `None`/`0` to reproduce the old unclipped behavior. Verified the composition actually clips
  (a `[100,100,100]` gradient reduces to global norm ~1.0) before deploying.
- **Curriculum stages warm-continued from whatever the live model held, not from that stage's own
  best checkpoint** — so stage 1 inherited stage 0's COLLAPSED final weights (not its 34.7%-best
  state) and, unsurprisingly, never recovered (finished its own full 30000 steps at MEAN=0.5%,
  worse than the collapse it inherited) — a real cascading-failure design gap, not a training
  instability question at all. Fixed: `hp['warm_start_from_best']` (default `True`) loads the
  previous stage's own `stage{i-1}_best.pt` at the start of each new stage instead of continuing
  from wherever training happened to land; set to `False` to reproduce the old always-continue
  behavior. Verified locally (multi-stage smoke test confirms the "warm-started from stage N's own
  best checkpoint" log line fires and the loaded weights are actually used).

Both fixes verified independently before combining (grad-clip composition math checked in
isolation; warm-start log line + checkpoint load confirmed in a 2-stage local smoke test) and
together (same smoke test, both features active, no errors). Curriculum relaunched from scratch
with both fixes rather than resumed from the now-poisoned checkpoints.

- **`hmn_tpu_recall1024_jax_adaptive_mix.py` — first full run, done.** 20000 steps, `B=8`,
  `lr_max=1e-4`, all fixes above combined (small batch, corrected LR, adaptive reweighting, jit
  decode eval). Compile: 60 training shapes in 154-994s across different runs (noisy, not
  root-caused — see compile-time-instrumentation entry above). Loss declined from the random
  baseline (~5.545) down into the 5.25-5.35 range with real per-entry differentiation by step
  4000-6000 (previously stuck flat at baseline for 2000+ steps before the small-batch/LR fixes).
  **val MEAN**: 0.2% (step 2000) -> 0.5% (4000) -> 19.1% (6000) -> 33.7% (8000) -> **plateaued at
  33.7-33.9% for the remaining 12000 steps** (evals at 10000/12000/14000/16000/18000/20000 all
  landed within 0.2pp of each other) despite train loss continuing to decline the whole time —
  a genuine generalization plateau, not an artifact (loss-still-falling-but-match-flat is exactly
  the signature CLAUDE.md's own `hmn_single_recall_c128` entry already documents as "undertrained"
  ELSEWHERE, but here 12000 steps of flat match against a still-declining loss reads more like a
  real ceiling at this model scale/step budget than simple undertraining — not conclusively
  settled either way). **Final: MEAN=33.9%, best checkpoint=33.9%** (same value, `stage0_end.pt`).
  Adaptive weights stayed close to uniform throughout (0.01-0.02 range) — the difficulty spread
  across entries was never large enough for the reweighting to meaningfully concentrate effort on
  a specific subset. Natural next steps, not yet done: longer budget to see if the plateau is
  truly a ceiling or would eventually break: a bigger model; or inspecting whether specific
  entries (e.g. the hardest near-start-anchor ones, per this project's own recurring "near-end
  anchor easy / near-start anchor hard" pattern) are the ones actually capping the MEAN.


