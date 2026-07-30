# kvmem

Fast-weight language model — HashMemNet (HMN). **Current focus: `kvmem/hmn.py`**, a from-scratch consolidated single-file rewrite (single-attn blocks + shared chat-tag vocab + bounded cross-chain-step relay), replacing the prior multi-file `kvmem/`+`experiments/` stack. All prior code/docs/checkpoints are preserved verbatim under `archive_v1/` (old `kvmem/`, old `experiments/`, and `archive_v1/CLAUDE_v1.md` — the previous version of this file) — nothing was deleted, just superseded. `archive_v1/` code still runs standalone via `PYTHONPATH=archive_v1`.

**Why the rewrite**: a design review caught that the old chat-tag vocabulary assigned a separate tag token per window (`HMN_QUERY_A_OPEN`..`_G_OPEN`) — backwards for a chat-formatted design (a real LLM reuses the same role tokens every turn; turn identity comes from position, not a turn-numbered vocab entry). Fixing it required retraining anyway, so this was also the point to build in bounded, persistent chain memory from the start instead of bolting it on later. Full rationale, worked examples, and every original naming decision: [`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`](/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md) (the approved plan — read this first for historical *why*; some of its terminology, notably `STATE_QUEUE`/`h_inject`, describes a mechanism since superseded — see below).

---

## Terminology (read this before the code — "step" alone is retired, it was overloaded 4 ways)

| Concept | Term |
|---|---|
| token index within the packed sequence | **position** |
| schedule position in a chained schedule (old: "window A/B/C/G") | **chain step** (always two words) |
| byte range being recalled (e.g. `span=(0,2)`) | **span** |
| initial vs refine pass within one chain step | **round** — round 0 = initial, round *k*>0 = refine (unified into one `_emit_round(round_idx, ...)`, not two separate block types) |
| SGD/optimizer iteration (`global_step`/`local_step`) | **training step** |
| chat_tags-style weighted trajectory sampling | **trajectory** |
| the compressed per-chunk/per-round register (old: "SLOT") | **STATE** (`HMN_STATE_0..N-1`, `state_len`, `state_vocab_size`, `_cyclic_state_ids`) |
| bounded cross-chain-step memory channel | **relay** — currently the `hop` mechanism (single-hop attention permission, see below); the original `STATE_QUEUE`/`h_inject` design (forced feature-vector copy) is deleted, see "Deleted mechanisms" |

**Vocab layout** (reordered this session — chat tags now come first, STATE occupies the tail): `HMN_SRC_OPEN/CLOSE`=256/257, `HMN_QUERY_OPEN/CLOSE`=258/259, `HMN_RESPONSE_OPEN/CLOSE`=260/261 (three generic pairs, reused identically at every chain step, no per-position variants, no `<mem>` wrapper — STATE regions are self-identifying via their placeholder tokens). `HMN_STATE_0`=262 onward — the only region expected to grow, so growth is always a pure tail-append (`hp['V']` must be bumped accordingly past `state_vocab_size=12`, which is free under the default `HMN_TAG_VOCAB_SIZE=274`). See `kvmem/hmn.py`'s vocab-section docstring for the full mechanics and why this ordering replaced the original one.

**Masking rule names** (renamed this session from a legacy "Rule 2/3/3b/3b'/4a/4b/5-8" numbering to descriptive names — see `kvmem/hmn.py`'s `chunk_mask_fb`/`chunk_mask_fb_hop`/`chunk_mask_fb_traj`): **encoding isolation** (encoding STATE_k blocked from other chunks), **chunk blackout** (recall STATE blocked from all raw chunks), **nochain blackout** (recall STATE blocked from all prior chain steps' content — the core invariant), **relay exception** (the `hops`-controlled carve-out from the nochain blackout — see below), **warmup/output bottleneck** (round-0 warmup/output rows restricted to own content only), **refine feedback isolation** (a refine round's `state`/argmax/`feedback_state` blocked from chunks + other outputs), **refine output bottleneck** (refine warmup/output rows restricted to own content only).

**`hops` semantics** (`chunk_mask_fb_hop`/`chunk_mask_fb_traj`, redesigned 2026-07-15 — no separate flag, `hops` alone controls both the relay window width and whether recurrent mode is on, since those are the same question): **`0` is invalid** (raises `ValueError`) — **`-1` (default)** is unbounded/routing-style: every chain step/op sees the union of ALL earlier chain steps'/ops' own STATE, AND keeps permanent unrestricted attention to every encoding-pass STATE directly (this is what `hmn_routing_4to1_state.py`'s single-chain-step case already does, generalized across multiple chain steps) — **`N>=1`** is a genuine bounded recurrence: the union of only the last N chain steps'/ops' STATE, AND every chain step/op past the first (chain step/op 0 is always exempt — the entry point, no predecessor to relay from, same role as an RNN's `h_0=f(x_0)`) is ADDITIONALLY blocked from every encoding-pass STATE directly, leaving the relay window as its ONLY channel for anything beyond its own local query/warmup — literally `h_t=f(h_{t-1..t-N}, x_t)`, not routing with an optional bonus channel. All four combinations (hops=0 invalid, hops=-1, hops=1, hops=2) verified by direct mask-matrix inspection for both `chunk_mask_fb_hop` (chain-step path) and `chunk_mask_fb_traj` (op_idx path, `stream` pattern). This closes a real gap: encoding-pass access used to remain permanently open regardless of `hops`, making the relay optional rather than load-bearing — `hop`'s pre-2026-07-15 measured results (below) were produced under that leaky version and may not reproduce under this corrected masking. `hmn_recall_queue.py` (already sets `hops=1`) and `hmn_weave_mix_accum_rnn.py` (new, sets `hops=1` explicitly) now exercise the corrected, genuinely-recurrent masking automatically — no separate `_accum_rnn`-suffixed hop config is needed (an earlier `hmn_accum_rnn.py` was built with a since-removed `strict_accumulation` flag and deleted once it became identical to `hmn_recall_queue.py`).

**Refine-round STATE naming**: a refine round has two STATE registers, both filled with placeholder tokens like every other STATE region — `state` (built first, functionally identical to every other STATE register: encoding STATE, round-0 STATE) and `feedback_state` (built after incorporating the round's argmax feedback — the register that actually seeds the fresh `<query>`/generation, and what `hop`'s relay reads as "that chain step's last-round STATE"). `feedback_state` also has its own dedicated tail vocab family (`HMN_FEEDBACK_STATE_FAMILY`, `kvmem/hmn.py`) — a distinct, role-based placeholder alphabet reused identically at every refine round (same pattern as `<query>`/`<response>`, not a position-indexed tag) — added since no `n_refine>0` experiment has ever been trained, so there was no checkpoint compatibility to preserve.

---

## Architecture in plain terms

**The task** (unchanged from every prior architecture in this project): memorize a byte sequence, then recall it from a short seed (warmup), byte-exact.

**Block types** — one unified `HMNModel` class, selected via `block_type` hp:

| `block_type` | Structure | Role |
|---|---|---|
| `attn_mlp` | `x = x + attn(norm1(x)); x = x + ffn(norm2(x))` | standard architecture, for comparison |
| `dual_attn` | `x = x + attn1(norm1(x)); x = x + attn2(norm2(x))` (paired, no MLP) | kept as an available ablation option (byte-identical port of the prior architecture) |
| `single_attn` | `x = x + attn(norm(x))` (one attn, one norm, no MLP) | **the default going forward** — same block repeated `n_layers` times; use `n_layers` = 2× the equivalent `dual_attn` config to match total attention-op count |

**Cross-chain-step relay (`hop`)**: each chain step after the first gets its own round-0 STATE row a narrow **attention permission** (the relay exception, width controlled by `hops` — see the `hops` semantics entry above) to read the last `hops` preceding chain steps' own last-round STATE directly — resolved entirely by mask permissions within one packed-sequence forward pass, no sequential per-chain-step orchestration. This replaced the original `STATE_QUEUE`/`h_inject` design (see "Deleted mechanisms" below), which forced a `.detach()`'d feature-vector copy instead of a learned attention path. The nochain blackout (nothing in the mask lets one chain step attend directly into another's raw content) still holds — the relay exception is the *only* sanctioned cross-chain-step channel, scoped to the STATE row alone (never warmup/response rows). At `hops>=1` this is also the ONLY channel period (encoding-pass access is blocked for chain steps past the first); at `hops=-1` (default) it's additive on top of permanent encoding-pass access.

**Deleted mechanisms**: the original `STATE_QUEUE`/`h_inject` relay (`chunk_positions_chained`, `HMNModel.forward`'s `h_inject` param, `train()`'s `chain=True` sequential per-chain-step training loop) was deleted after `hop` (the attention-permission alternative) was shown to massively outperform it — see "Results" below. `kvmem/configs/hmn_stage1_round0_chained.py` (the deleted mechanism's config) has also been removed — nothing in the codebase can execute a `chain=True` stage anymore.

---

## Results

- **`chunk_len` ladder** (`kvmem/configs/hmn_single_recall_c64.py` → `hmn_single_recall_c128.py`, single chunk/single STATE, no routing, no relay — `n_chunks=1, chain_steps=[(0,1)]`) — **c64: Done**, 100000/100000 steps, trained from scratch, val MEAN=100.0%, loss=0.012 (converged by step ~60000). **c128: Done**, warm-started from c64's checkpoint, 100000/100000 steps, val MEAN plateaued at 47.5% (best checkpoint 52.5%), loss=0.759. **Do not read this as a proven `state_len=8` capacity ceiling** — train loss was still declining at the very end (0.804 at step 90000 → 0.759 at step 100000, noisy but not flat) while `lr` had already annealed to ~1e-6 (`cosine_T0=100000` matches `n_steps`, so the schedule gave it no room to keep pushing) — the likelier read is **undertrained**, not capacity-limited. A longer schedule (larger `n_steps`/`cosine_T0`) at the same `chunk_len=128` is the natural next step before concluding anything about capacity at this `state_len`.
- **`repeat_batch=4` ablation on `hmn_single_recall_c64.py`** (`kvmem/configs/hmn_single_recall_c64_repeat4.py`, otherwise identical to the c64 baseline above) — **Done, negative result**: reached the same val MEAN=100.0% ceiling, but first hit it at step **80000** vs. the baseline's step **60000** — slower by 20000 steps, not faster. Unlike the `weave_mix_accum_rnn` case below (where the baseline was genuinely underfitting/plateaued and `repeat_batch=8` fixed it), `hmn_single_recall_c64` was already converging cleanly with no plateau — there was no underfitting problem to fix, so `repeat_batch>1` here just means fewer distinct random sequences seen per wall-clock step, a pure cost with no offsetting benefit. **Takeaway: `repeat_batch` is not a universal win** — helps when training is stuck/underfitting, neutral-to-harmful when it isn't.
- **`hmn_routing_4to1_state.py` (`solo`) — archived.** No checkpoint for it exists on disk (it was never actually (re)trained under the current vocab in this working tree — the numbers below are preserved from before, not reproducible from a file on disk today). `hmn_recall_queue.py` (`hop`), `hmn_weave_mix.py`, and `hmn_weave_mix_accum_rnn.py` have all been repointed to warm-start from `hmn_single_recall_c64.py`'s checkpoint instead (same architecture — `d`/`n_layers`/`n_heads`/`state_len`/`V` unchanged, only `n_chunks`/schedule differ, so weights transfer directly) — see `docs/HMN_WALKTHROUGH.md` for the current pipeline. The `solo`/`hop`/`weave_mix` numbers below describe what those stages measured under the OLD `hmn_routing_4to1_state` warm-start and are historical record, not reproducible as-is from the configs on disk now.
- **`solo`** (`kvmem/configs/hmn_routing_4to1_state.py`, renamed from `hmn_single_recall.py` — the recall STATE routes across all n_chunks encoding STATEs simultaneously via attention, verified zero-blocked in the mask; matches the `hops=-1` default's own unbounded/routing behavior, since a single chain step has no predecessor to relay from either way) — one chain step, round 0 only, no relay. **Done**: 160000/160000 steps, val per-span MEAN=94.4% (best 97.2% at step 150000), test=100%, loss=0.017 — matches the historical ~100% single-window initial-round ceiling.
- **`relay`** (config file and logs both deleted — the now-removed `STATE_QUEUE`/`h_inject` mechanism) — three chain steps (`[(0,2),(1,3),(2,4)]`), warm-started from `solo`. **Done, everything removed** (old vocab, superseded mechanism). Final numbers (preserved here since the run itself is gone): val MEAN=45.8% (STITCHED=44.6%), test MEAN=48.6% (STITCHED=44.6%). Chain step 2 (the 2-hop case) closed at 12.5%/12.5%, never exceeding its step-90000 peak of 11.1%/25.0% across the final 70000 steps despite loss continuing to decline (1.448→1.051). Motivated `hop`.
- **`hop`** (`kvmem/configs/hmn_recall_queue.py`, the attention-permission relay) — identical hyperparameters/schedule to `relay`, warm-started from `solo`. **Done, run twice, with a real discrepancy between the two runs** (see below) — the checkpoint currently on disk is from the SECOND run, weaker than what the recovery-probe result below was measured against.
  - **First run** (logs since deleted, old vocab): val = 100.0%/95.8%/72.2% (STITCHED=88.1%), test = 100.0%/95.8%/70.8% (STITCHED=85.7%), loss=0.603, best checkpoint 88.7%. Massively outperformed `relay` on every metric (chain step 2 test 70.8% vs relay's 12.5%, 5.7x) — strong evidence the gradient-flow fix (full backprop vs. `.detach()`-truncated copy) matters. This is the run the recovery-probe result immediately below was measured against.
  - **Second run** (current checkpoint, post vocab-reorder — same config, warm-started from the reordered-vocab `solo`): val/test STITCHED=71.4%/71.4%, loss=1.851 — substantially worse, loss plateaued flat around 1.84-1.86 from step 50000 onward, never broke out like the first run did. Mask/relay mechanism independently verified correct in both runs (byte-identical mask regardless of vocab ID relabeling); the discrepancy is attributed to warm-start sensitivity, not a code defect — two `solo` checkpoints can both hit ~100% on solo's own near-trivial task while differing enough in underlying weight configuration to matter for `hop`'s much harder relay-learning objective. Re-running `hop` is not guaranteed to reproduce either result exactly.
- **`accum_rnn` masking fix** (`kvmem/hmn.py`, 2026-07-15 — see the `hops` semantics entry above for the full mechanism) — closed a real gap found by direct mask inspection of `hop`: every chain step had PERMANENT, unblocked attention access to all n_chunks encoding-pass STATEs regardless of chain step (verified: chain step 2 showed 0/64 blocked against every one of the 4 encoding STATEs), so the relay exception was layered on top of that, never the only channel — `hop` could always bypass the relay and re-derive an answer straight from the encoding pass. `hops=1` now automatically forces genuine `state_t=f(state_{t-1},query_t)` recurrence (no separate flag). Since `hmn_recall_queue.py` already sets `hops=1`, it now exercises this corrected masking with no config change — its already-measured results above were produced under the OLD leaky version and are not guaranteed to reproduce. **`hmn_weave_mix_accum_rnn.py`** (new, `hops=1` explicit — `hmn_weave_mix.py` itself is unaffected, since it never sets `hops` and keeps the `-1`/unbounded default) is the trajectory-generalization counterpart: mirrors `weave_mix`'s test (does forced accumulation generalize across `batch`/`stream`/`interleave_delayed`) but under the corrected masking — **done, see its own entry below** (plateaued/underfit, fixed by the `repeat_batch` ablation immediately following it). If `hmn_recall_queue.py` collapses relative to its previously-measured numbers once re-run, that's direct evidence those numbers were substantially propped up by the encoding-pass bypass rather than genuine relay use.
- **`hmn_weave_mix_accum_rnn.py`** — **Done**. `hops=1` (forced single-hop recurrence, corrected masking) + `weave_mix` (`batch`/`stream`/`interleave_delayed`), warm-started from `hmn_single_recall_c64`. 160000/160000 steps. Loss and val both **plateaued from step ~50000 onward and never recovered**: loss (10k-step rolling avg) sits flat at 2.57-2.94 for the entire second half of training (no downward trend at all, not just a slow one), val MEAN 42.7-44.8% across steps 60000-160000, final=43.8%, best checkpoint=44.8%. This is a genuine **training-loss plateau** (underfitting), not merely a generalization ceiling.
- **`repeat_batch` ablation** (`hp['repeat_batch']`, `kvmem/hmn.py` — takes N gradient steps on the same sampled batch before resampling a new one, default 1 = no change from prior behavior; see `kvmem/configs/hmn_weave_mix_accum_rnn_repeat8.py`, identical config to `hmn_weave_mix_accum_rnn.py` above except `repeat_batch=8`) — **fixes the plateau**. Direct comparison (10k-step rolling-avg loss, val MEAN, both configs otherwise identical):

  | step | baseline loss | repeat8 loss | baseline val MEAN | repeat8 val MEAN |
  |---|---|---|---|---|
  | 10000 | 3.896 | 4.013 | 34.0% | 27.6% |
  | 30000 | 3.058 | 3.078 | 40.7% | 36.4% |
  | 50000 | 2.782 | 3.006 | 41.4% | 45.8% |
  | 60000 | 2.775 | **2.660** | 41.4% | **50.2%** |
  | 70000 | 2.842 | **2.360** | 43.8% | **52.2%** |
  | 80000 | 2.872 | **2.434** | 43.4% | **53.7%** |
  | 90000 | 2.935 | **2.499** | 42.7% | **53.2%** |
  | 100000 | 2.707 | **2.374** | 42.9% | 50.6% |

  `repeat_batch=8` starts *behind* the baseline through step ~40000 (fewer distinct batches seen per wall-clock step), crosses over decisively at step 50000, and from step 60000 onward has BOTH a lower (still-declining) loss AND 6-10pp higher val MEAN than the baseline ever reached at any step in its 160000-step run — the baseline's own peak val (44.8% best checkpoint) is beaten by repeat8's step-60000 checkpoint alone, at 3/8 of the training budget. Loss trajectory is the more important signal here: baseline's loss is flat-noisy (no trend) past step 50000 while repeat8's keeps trending down through at least step 100000, i.e. the baseline wasn't failing to generalize from an already-fit training signal — it was failing to fit the training data itself, and taking multiple gradient steps per batch is enough to unstick that.

  **Done** (was "in progress" — finished at step 160000/160000: loss=1.821, val `batch`=50.5%/`stream`=44.9%/`interleave_delayed`=48.1%/MEAN=47.8%, **best checkpoint=53.7%** at step 80000; came down slightly off that peak over the last 60000 steps — 53.7%→53.2%→50.6%→...→47.8% — but never dropped back to baseline's 42.9-44.8% range).

  **Qualitative comparison** (`ar_decode_traj_nokv`, pattern `batch`, seq `up_counter`, generated bytes vs ground truth — script preserved in this session's scratchpad, not checked into the repo): baseline's checkpoint degrades from readable-but-wrong to complete non-printable noise almost immediately —
  ```
  op0 (45.8% match): ()*+,-./0Q2S4;VW/YT5\]X_        <- trails off after ~9 correct bytes
  op1 (0.0% match):  \xd9H(}1-11F\x9d\xcb\xed]...     <- complete non-printable garbage
  op2 (0.0% match):  ()*+\xd1-./6#\xce_&\xfe...        <- complete non-printable garbage
  ```
  repeat8's best checkpoint (53.7%) is qualitatively different in kind, not just degree — `stream` op0 hits 100.0% EXACT match; `interleave_delayed` op0 hits 70.8% (`HIJKLMNO` — 8 exact bytes before diverging); `batch` op2 (the 2-hop case) goes from 0.0%/pure-noise to 25-29% with genuinely plausible printable characters mixed into the wrong bytes, instead of uniform escape sequences. The one op that stays weak in every pattern is **op1** (the 1-hop relay case specifically) — 4.2-8.3% match, still mostly non-printable — a real, specific remaining gap, not a uniform improvement across the board.
- **`hmn_stitch_src1024.py` — REDESIGNED, multi-hop relay-chain version abandoned before completing a full run.** Original attempt: src=1024 (`chunk_len=64, n_chunks=16`), true-continuous-decode via a CHAIN of ~30-126 relay hops (`chunk_positions_stitch`/`make_batch_stitch`/`ar_decode_stitch`, all new code in `kvmem/hmn.py`). Got as far as a working, verified pipeline (two real bugs found and fixed via smoke testing: (1) the original "_nokv" full-recompute decode OOM'd on MPS during eval at L~4900, fixed by making it genuinely KV-cached like `ar_decode_iq_global_rw_tagged`, byte-identical to a brute-force reference; (2) an off-by-one in KV-cache bookkeeping, the `</response>` closing tag's own token was never cached, fixed by explicitly caching that position too) and a real memory/speed fix (`segment_checkpoint` hp flag — TIME-axis/segment gradient checkpointing via `torch.utils.checkpoint.checkpoint`, separate from `HMNModel`'s own model-depth `grad_checkpoint`, verified mathematically equivalent to the non-checkpointed path — combined with `B=2, forward_granularity=0.125-0.25` this got a 100-step smoke test through cleanly at 1.4-4 it/s on an 8GB-RAM machine that had OOM'd at more aggressive settings). Launched twice (once killed by an apparent machine reboot at step 199, restarted) but abandoned before completing any real training — decided the whole multi-hop-chain design was more complex and expensive than needed for the actual question being asked.
  - **New design** (current): no relay chain at all. Encode n_chunks chunks, then a SINGLE query recalling the SUFFIX of the source — warmup anchors partway through (8 ground-truth bytes), response must generate everything after that anchor through the true end, whatever length that is. New `traj_suffix` trajectory constructor (`kvmem/hmn.py`, registered in the `weave_mix` dispatch as `pattern='suffix'`) builds this as `Q(n_chunks-window_chunks, n_chunks)` — `window_chunks` here means "how many chunks back from the end the anchor sits," not a sliding window; enforced `>=2` so there's always a non-trivial response (the degenerate "warmup right at the end" case is excluded by construction). Since there's only ever one query (`op_idx=0`, always exempt from the `hops`-bounded relay restriction), none of `hops`/`forward_granularity`/`segment_checkpoint` are needed — a plain dense forward pass, and `L` is dramatically smaller (1452 at n_chunks=16 vs. 2800-4900 for the abandoned chain versions).
  - **Curriculum**: 3 stages in one run (continuing the same model/optimizer) ramping `n_chunks` up — {2,4} → {2,4,8} → {2,4,8,16} — each stage additionally mixing several `window_chunks` values so the warmup anchor lands at varying distances from the end (the practical stand-in for "warmup from any byte index," since each distinct shape needs its own fixed packed-sequence layout). Warm-started from `hmn_weave_c64`'s checkpoint (chunk_len=64-matched, see that config's own docstring).
  - **Extrapolation check planned** (held out from training entirely): evaluate at n_chunks=24/32, sizes never seen during training (capped at 16), using the existing generic `ar_decode_traj_nokv` — verified directly (offline, tiny CPU model) that all planned training shapes AND both extrapolation sizes build/forward/decode correctly before trusting this design, per this file's own "verify before trusting mid-run" rule.
- **Chain-memory recovery probe** (`eval_weave.py --patterns repeat_query`, run against `hop`'s FIRST-run checkpoint, since deleted) — **failed cleanly**. Query span (0,2)→(1,3)→(2,4)→re-query (0,2): first occurrence 100% (trivial), repeated occurrence (reachable only through the accumulated 3-hop relay chain, since direct attention back to chunk 0/1 is blocked) **0.0% across all 3 test sequences** — complete, not partial, failure. Caveat: `repeat_query` is a trajectory shape `hop` was never trained on (only the fixed 3-query schedule), so this could reflect either "the relay doesn't preserve information across 3 hops" or "the model can't generalize to this novel trajectory shape at all" — the total (not gradual) failure leans toward the latter, but this test alone can't cleanly separate the two. `long_hop_recovery` (n_chunks=8) scored near-zero including first occurrences — the known length-extrapolation confound (trained at L=236), not an additional signal. This motivated `weave_mix` (below) — re-running the probe against a `weave_mix`-trained checkpoint is the direct follow-up test.
- **Vocab reorder** — mechanism verified correct (mask byte-identical regardless of vocab ID relabeling, batch construction confirmed correct under the new IDs). Chat tags now occupy IDs 256-261 (fixed, small), STATE occupies the tail from 262 (pure append-growth). Old-vocab logs/checkpoints (original `solo`/`relay`/`hop`) have been deleted, and the `_vreorder`-suffixed configs/logs were renamed to drop that suffix now that the reordered vocab is simply *the* vocab (no more old-vocab comparison to distinguish against) — `kvmem/configs/hmn_routing_4to1_state.py` (renamed from `hmn_single_recall.py`) and `hmn_recall_queue.py` (renamed from `hmn_flow.py`) are now the reordered-vocab versions. See the `hop` entry above for the reproducibility-check numbers themselves (kept there, not duplicated here).
- **`squeeze`** (`kvmem/configs/hmn_squeeze_markov_n4.py` + `hmn_squeeze_random_n4.py`) — **not currently running; earlier partial progress (paused at step 17998/60000) deleted** as stale/low-value (superseded by the sweet-spot pair below, see `nominal_capacity_accounting`) — both configs are queued from scratch if resumed. Dedicated compression-capacity test (Markov-structured vs. random-byte paired control) at `chunk_len=96, state_len=8, d=64` — nominal capacity headroom is large (KV-cache view 682.7x the true content), so this pair alone doesn't force genuine compression, only demonstrates it's tractable. Switched from `data_kind='ca'` to `data_kind='markov'` (the earlier `hmn_squeeze_ca_n4.py`, never trained, is superseded/deleted) — `gen_ca`'s `target_bits` calibration is zlib measure-and-search (seed-dependent, imprecise); `gen_markov`'s is an exact closed-form bisection against the true entropy rate. See `docs/HISTORY.md` §10 for the full design rationale, including the `chunk_len` capacity-pressure correction.
- **`squeeze` sweet-spot pair** (`kvmem/configs/hmn_squeeze_sweetspot_n4.py` + `hmn_squeeze_sanity_bigmodel_n4.py`) — deliberately sized so success is neither trivial (STATE smaller than the raw file, ruling out byte-for-byte copying) nor information-theoretically impossible (STATE still bigger than the data's true Shannon content), the only window where a result actually means something (see `nominal_capacity_accounting`). `hmn_squeeze_sweetspot_n4.py` (`chunk_len=1024, state_len=2, d=8, n_layers=4`, 5,304 params, KV-cache/true-content ratio=2.0x, KV-cache/raw ratio=0.5x — genuinely can't trivial-copy) — **queued, not yet run** (~7.0 hrs measured for 60000 steps). `hmn_squeeze_sanity_bigmodel_n4.py` (same dataset, `state_len=8, d=8, n_layers=4`, 128x more nominal headroom — a pure "does recall even work at this chunk_len/L~2000 sequence length at all, independent of any capacity question" reference ceiling) — **not currently running; earlier minimal progress (stopped step 5999/60000, near-random loss=5.541) deleted** as stale — queued from scratch if resumed.
- **`weave_mix`** (`kvmem/configs/hmn_weave_mix.py`) — **Done**. Uniform mix of `batch`/`stream`/`interleave_delayed`, 160000/160000 steps, warm-started from `solo`'s checkpoint (NOT `hop`'s — `hop`'s current checkpoint is the weaker, non-reproduced second run; `solo` has no such ambiguity, see the `hop` entry above). Final: `batch`=63.4%, `stream`=95.8%, `interleave_delayed`=63.9%, MEAN=74.4% (best checkpoint 74.7%). `batch` (byte-shape-identical to what `hop` trained on) landed well below `hop`'s own 88.7% on that shape — since this run had to learn the relay exception AND generalize across trajectory shapes simultaneously (no pre-learned relay to transfer from), that gap suggests a weaker relay, not just weaker generalization.
  - **Recovery probe re-run** (`eval_weave.py --patterns repeat_query,long_hop_recovery,decay_curve`, against this checkpoint) — **partial, not clean, improvement over `hop`'s 0.0%**: `repeat_query`'s repeated occurrence of span (0,2) after 3 intervening queries now recovers 0.0-4.2% (vs `hop`'s uniform 0.0% across all 3 test sequences), average drop 80.6pp. `decay_curve` shows the same pattern — recovery degrades with hop distance (80.6pp drop @1 hop, 98.6pp @2-4 hops, 94.4pp @8 hops, noisy at the tail). Not a clean confirmation OR refutation of the original hypothesis ("was `repeat_query`'s failure pure generalization gap, since `hop` never trained on varied orderings?") — the weaker relay (from the `solo` warm-start, see above) confounds the result; a cleaner test would warm-start from a strong `hop` checkpoint (would need re-running `hop` first, see its own entry's caveat) so relay-strength and generalization-gap aren't both varying at once.
- **Positional shortcut in `batch`/`interleave_delayed` — three RoPE-mechanism fixes failed, anchor variation (a completely different attack) succeeded so far.** `kvmem/probe_positional_shortcut.py` found the root cause of `batch`/`interleave_delayed`'s persistent 8-20% ceiling (`hmn_adaptive_trainer.py`'s reweighting couldn't fix it either): the model resolves these two shapes' shared queries via pure attention POSITION, not warmup content (91.1% match to the wrong-but-positionally-usual chunk vs. 0.4% to the chunk whose real bytes were actually given). Three RoPE-mechanism fixes tried under the old tagged design — dual-clock RoPE, `rope_state_scale`, `relpos` — all failed or were abandoned (`docs/HISTORY.md` §12-13); all three are now marked **deprecated** (see `_dual_positions`/`_scaled_state_positions` docstrings, `kvmem/hmn.py`). A live head-to-head under the NEW opcode/no-tags design (`hmn_notags_w25` vs `hmn_notags_w25_rope`, otherwise identical) then found RoPE itself converges dramatically faster/higher than NoPE at every comparable stage/step — reframing the question: maybe RoPE wasn't the problem, and the fix should attack the shortcut a different way instead of changing the position encoding. `hmn_notags_weave_anchor(_rope).py` (`_grid_shapes` sweep of chunk_len x non-zero-biased recall anchors, on top of `hmn_weave_c64.py`'s curriculum) tests exactly that. Both the original behavioral swap test AND a new mechanistic check (attention-mass + gradient-saliency, `kvmem/probe_mechanistic_addressing.py` + `MHAttention.capture_attn`, `kvmem/hmn.py`) confirm content-addressed (not position-addressed) recall at every length where training has converged so far (chunk_len 8/16/32); chunk_len=64 needs more training before it can be tested (the run finished — cl64 plateaued flat all stage, cl8/16/32 kept climbing — undertrained there, not capacity-limited; the cl64 mechanistic question stays open). Full detail: `docs/HISTORY.md` §16.
- **Refine-round redesign: uniform `[STATE][w][content]` primitive, within-op bottleneck relay, `am` now exposure-bias-trained.** Every round (round-0 and every refine round) now reduces to one repeating shape instead of three visually different ones; the post-response commit-STATE between rounds is now mandatory (not just claimed-by-relay), within-op round transitions are masked exactly like a between-op relay hop (no more free raw lookback across a whole op's history), and the argmax feedback (`am`) is now NLL-scored against the SAME ground truth every response uses (not against its own content) — genuine exposure-bias training. Caught and fixed a real bug during the pass: the `'S'`-claim logic still referenced a field (`end_sl0/end_sl1`) removed by the field-unification, which would have silently broken every claimed refine-relay. Verified via direct mask-matrix inspection plus `train()` smoke tests (single-op multi-round, cross-op relay with refine on both sides, the new loss terms) — no config sets `n_refine>0` yet, so nothing live was disrupted. Full detail: `docs/HISTORY.md` §17.
- **`hmn_locate_nope_curriculum_dense` — is the architecture or the dataset a limiting factor? Both: no.** `kvmem/probe_signal_propagation.py` (`--mode signal`/`--mode ambiguity`, two real bugs found/fixed in the diagnostic itself before trusting any result — see its own docstring) found no vanishing gradients (grad norm is actually LARGER in early layers, the healthy residual-network pattern), no exploding activations (RMSNorm re-normalizes each block's input regardless of residual-stream growth), and attention genuinely sharpens with training (entropy 0.73-0.90 uniformly at random init vs. 0.30 in layer 1 post-training) — no architectural/signal-propagation ceiling evident. Separately, empirical genuine-ambiguity rate (does the TRUE warmup excerpt's exact bytes recur elsewhere in the same random chunk) is under 0.1% even at the hardest tested `chunk_len=64`/`warmup_len=2` — the `data_kind='random'` training data is not a meaningful limiting factor either; uniform random is close to the best-case distribution for this task (structured/repetitive data would introduce MORE genuine duplicate substrings, not fewer). Full detail: `docs/HISTORY.md` §14.
- **`kvmem/hmn.py` promoted to the opcode+shared-value-alphabet, no-chat-tags design — STATE-role ambiguity fixed.** Without chat tags, an encode's claim-STATE and a query's own recall-STATE share the same token family — no vocab-level signal distinguishes them, only local order. Fix: one **opcode token** (`update`/`noop`/`feedback`) per STATE emission + a single **shared value alphabet** across all three roles. Separately found the query's own STATE emission is **redundant for the terminal op** (nothing relays from it) but **load-bearing for any non-terminal op** (it's the `h_t=f(h_{t-1},x_t)` transition output itself — masking can't substitute for a value that was never computed). STATE moved to *end-of-turn* (`[warmup][response][STATE]` instead of `[STATE][warmup][response]`), claimed by a trailing `'S'`, omitted for terminal ops. Hand-verified across single-recall/batch/stream/3-query-chain trajectories before writing code; the stream+`hops=1` case reproduces the exact empirical weak spot already on record (`hmn_weave_mix_accum_rnn_repeat8`'s persistent "op1" ceiling) from first principles. **Fully implemented, ported, and promoted**: all four stage-dispatch paths (`weave_mix`/`chunk_positions_traj` native; `chain_steps`/`chunk_positions_hop` now a thin delegating wrapper; the legacy global-window path/`chunk_positions_iq_global_rw_tagged`; `stitch_mix`/`chunk_positions_stitch`) rewritten around one positive `allowed_state` allowlist per op, refine-round support added (`OP_FEEDBACK` placed before the argmax content, fixing a real boundary ambiguity between feedback content and the prior round's response), and two critical bugs caught during pre-promotion verification: (1) the named pattern constructors `traj_batch`/`traj_stream`/`traj_interleave_delayed`/`traj_repeat_query` never inserted a trailing `'S'` between consecutive queries, making every query terminal (no end-of-turn STATE at all) and silently breaking the relay chain for every config using these patterns; (2) `_relay_source` mishandled `'noop'`-type blocks (`KeyError: 'end_sl0'` on `decay_curve`). Both fixed and re-verified via direct mask-matrix inspection (including confirming under `hops=1` that a later query sees ONLY its predecessor's end-of-turn STATE, not the encoding-pass STATEs directly) plus a real end-to-end train+eval smoke test through the actual `weave_mix` dispatch path. `kvmem/hmn.py` was archived to `kvmem/hmn_v4_backup.py` (old tagged design, `V=274`) before promotion, and that backup (along with the earlier `hmn_v1-v3_backup.py` diffing snapshots) has since been deleted as a pure cleanup pass; `kvmem/hmn_notags.py` no longer exists as a separate file either — its content IS `kvmem/hmn.py` now. **Caveat not yet resolved**: every config with `_pretrained_ckpt` set (`hmn_single_recall_c128`, `hmn_weave_c64*`, `hmn_weave_mix*`, `hmn_stitch_src1024`, etc.) warm-starts from a checkpoint trained under the OLD tagged vocab (`V=274`, tags at IDs 256-261) — loading it into the new opcode vocab (`V=271`, opcodes at 256-258) via the existing shape-mismatch-tolerant loader would silently reinterpret old tag-token embeddings as new opcode-token embeddings, which is wrong, not just stale. Do not warm-start any of those configs from pre-promotion checkpoints without addressing this; configs training from scratch (e.g. `hmn_notags_w25`, `hmn_notags_locate` — no `_pretrained_ckpt` key) are unaffected. Full detail: `docs/HISTORY.md` §15.
- **`hmn_stitch_src1024_anchor` — content-addressing confirmed both behaviorally and mechanistically for the suffix-recall (single-query, `hops=-1`) design, reconfirmed at n_chunks=8; run deliberately stopped after stage1's first eval.** Restarted with a 10x'd schedule (100000/150000/150000 steps), `adaptive=True` (`early_stop_mean=80.0`), `repeat_batch` B4 (stage0)/B8 (stage1-2), warm-started from `hmn_notags_weave_anchor_rope`. **Stage0 (n_chunks in {2,4}) done**: best val MEAN=49.4% (eval1, step 20000), ended at 41.1% (step 100000) — 5 evals across the stage oscillated 38.9-49.4% without a clean monotonic trend (one dip to 38.9% at step 60000, mistaken mid-run for a possible collapse, fully recovered by step 80000 — a transient reweighting wobble, not real degradation). Near-end anchors (short response) consistently hit 83-100%; two entries (`Q(2,4,44,32)`, `Q(0,4,0,64)`) never broke above ~2% match across all 5 evals — the model's stubborn weak spot, unrelated to the addressing mechanism itself (see below). **Stage1 (n_chunks up to 8) eval #1** (step 30000, global step 130000): MEAN=36.1%. Per-entry: `Q(0,2,0,32)`=0.0%, `Q(0,2,88,32)`=58.3%, `Q(0,4,0,64)`=0.2%, `Q(0,4,184,64)`=83.3%, `Q(4,8,0,64)`=5.9%, `Q(4,8,92,64)`=9.0%, `Q(4,8,184,64)`=100.0%, `Q(0,8,0,96)`=1.0%, `Q(0,8,408,96)`=75.0%, `Q(0,8,0,128)`=2.1%, `Q(0,8,376,128)`=62.5% — same near-end-anchor-easy/near-start-anchor-hard pattern as stage0, now replicated at the harder n_chunks=8 shapes too, with no early-stop trigger (36.1% << 80.0%). **Run was then deliberately stopped (`kill 17291`, confirmed exited) per explicit user instruction, right after this eval and the probe re-run below** — not a crash or failure; `stage1_last.pt`/`stage1_best.pt` are the final checkpoints.
  - **Content-addressing swap test** (`kvmem/probe_stitch_content_addressing.py`) — originally against `stage0_best.pt`/`stage0_end.pt` at n_chunks=4 (4 anchor pairs: 44↔0, 64↔0, 20↔80, 80↔20), **position-match pinned at 0.0-0.8% in every trial** vs. content-match 33.7-79.7%. **Re-run against `stage1_last.pt` at the harder n_chunks=8, window_chunks=4/8 shapes** (matching stage1's own trained entries): low-baseline anchor pairs (0↔92, baseline 0.3-6.0%) were inconclusive (too weak a baseline to read a signal from either way), but the two high-baseline near-end-anchor pairs tested — `184↔92` (window_chunks=4, baseline 90.6%) and `408↔0` (window_chunks=8, baseline 90.6%) — both again showed the same clean pattern: **position-match 1.6% in both, content-match 37.5%/40.6%** — content-addressing reconfirmed at n_chunks=8, not just n_chunks=4.
  - **Mechanistic confirmation** (`kvmem/probe_stitch_mechanistic_addressing.py`) — against `stage0_end.pt` (n_chunks=4, anchor=44←chunk0 and anchor=88←chunk1): gradient L2 norm 5.9x/10.4x higher on the swap-source STATE, attention mass layers swap-dominant in both — **confirmed**. **Re-run against `stage1_last.pt`** (n_chunks=8, anchor=184←swap-chunk): swap-chunk=0 (far outside the window) was **mixed/inconclusive** (teacher-forced behavioral match 0.0% — this exact cross-distant-chunk construction is highly out-of-distribution for a checkpoint only 1 eval into n_chunks=8 training), but swap-chunk=3 (immediately adjacent to the window, still a genuinely different STATE) gave a clean **CONFIRMED** result: gradient L2 norm 7.7x higher on the swap-source STATE (1.38 vs 0.18), attention mass in the layer carrying the most overall weight (layer 1) swap-dominant. Net read: content-addressing holds mechanistically at n_chunks=8 too, though the far-chunk case shows the mechanism gets harder to probe cleanly (not necessarily weaker) the more novel/OOD the specific swap construction is relative to what's actually been trained so far.

---

## Structured-data track (queued, not yet used in training)

**Why**: genuine compression (zip/gzip-style, exploiting statistical redundancy) cannot emerge from training on the max-entropy random bytes used everywhere else in this project — Shannon's source coding theorem makes such data literally incompressible, so there's no redundancy for `STATE` to learn to exploit. Random-byte training only teaches raw lossless storage density and the addressing algorithm, not compression. Getting emergent compression requires structured/compressible training data.

**`kvmem/structured_data.py`** implements nine generator families (plus one documented placeholder, `gen_template_repeat`), each sampling **fresh random parameters per call** (required, not optional — a fixed rule across all examples lets the model bake it into static weights instead of encoding anything into `STATE`, the same FFN-as-static-knowledge failure mode this project's `dual_attn` design already avoids elsewhere), organized by which real compressor family they're built to exercise (see `docs/HISTORY.md` §8 for full detail and `LANGUAGE.md` for the generative-hierarchy framing that motivated the newer six):
- `gen_chaotic_logistic` / `gen_fractal_midpoint` — continuous dynamical systems (logistic map / midpoint-displacement fractal), byte-quantized. Weak, quantization-lossy structure — kept for ablation, not the default.
- `gen_ca` — 1D cellular automaton, random rule table + initial condition. **Default for general use** — discrete-native, exactly reproducible, huge tunable rule space.
- `gen_markov` — order-1 Markov chain over the full 256-byte alphabet, **exact closed-form entropy-rate calibration** (bisection against the true stationary-distribution entropy, no measure-and-search). **Measured finding**: `measure_bits_per_byte` (zlib) is *not* a valid sanity check for this generator — DEFLATE's Huffman stage codes against marginal/global frequency, not the previous byte, so it's structurally blind to order-1 conditional structure even when that structure is real and exactly calibrated (empirically confirmed: zlib stayed ~7-8 bits/byte across target_bits=1/2/4/6, while direct empirical order-1 conditional entropy tracked correctly at ~0.85/1.6/2.8/3.6). This is exactly the kind of structure a context-conditional (attention-based) model like this project's own architecture CAN exploit, even though zlib can't see it.
- `gen_iid_skewed` — i.i.d. bytes from a skewed (Zipf-like) marginal distribution. The deliberate "control case zlib CAN see" pairing against `gen_markov` — exact closed-form entropy, and zlib tracks it closely (verified).
- `gen_run_length` — RLE/LZ77-visible: fresh byte + geometric-length run, repeated. Second "zlib-visible" control case, different mechanism (literal repeats, not marginal skew) — approximate closed-form calibration, verified tracking zlib well.
- `gen_markov_order_k` — generalizes `gen_markov` to context length > 1, small alphabet (`K`/`order` tunable, meta-state space `K^order`) for tractability, same exact bisection-on-entropy approach.
- `gen_match_distance` — parametrized LZ77-style generator (match probability + match-DISTANCE range + match-length, not a fixed phrase vocabulary): **required a mid-implementation fix** — an initial single-byte-copy-per-event version was nearly invisible to zlib (DEFLATE needs a 3+ byte match to encode one at all; isolated single-byte copies almost never chain into that by chance), fixed by emitting genuine multi-byte match runs, after which zlib tracked target_bits closely. **Recovery-probe contamination warning**: exact byte repetition means a model could "recover" a matched byte via simple positional copying — do not use for the chain-memory recovery probe without accounting for this.
- `gen_mixed_order` — stochastically blends order-0/1/3 components per position (CTW/PPM-exploitable — defeats any single fixed context length). Least precisely calibrated of the nine (component-wise calibration is only an approximation of the true switched-process entropy) — flagged honestly, not overclaimed.
- `gen_template_repeat` — **placeholder, `NotImplementedError`**, fixed-vocabulary phrase-repetition design documented but not built (superseded in practice by `gen_match_distance`'s more general, already-implemented approach).

**`target_bits` parameter**: all generators accept `target_bits` — desired bits/byte of TRUE compressibility. Calibration precision varies by generator and is documented per-function: exact closed-form bisection for `gen_markov`/`gen_iid_skewed`/`gen_markov_order_k`; approximate closed-form bisection for `gen_run_length`/`gen_match_distance`/`gen_mixed_order`; `measure_bits_per_byte` (zlib) measure-and-search for `gen_chaotic_logistic`/`gen_fractal_midpoint`/`gen_ca`, with the known seed-dependent imprecision documented in each of those three's own docstrings (unchanged from before this session).

**Caution before using this for the chain-memory recovery probe specifically**: structured data risks contaminating that probe — a model could "recover" an earlier chain step's content by inferring the generating rule from its own visible span, without touching the relay at all. Keep the recovery probe on pure random data first; structured data is queued as a separate, later question (does bounded `STATE` capacity effectively increase when content is compressible), not a replacement for the current validation.

---

## TPU port (2026-07-30) — status: infra confirmed working, first real training run not yet completed

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

**Given this, `kvmem/hmn_jax.py` is now the recommended path for any `rope=True` work at Run A's
scale** — `hmn_tpu_recall1024_flat.py`'s original `rope=False` torch_xla config remains a valid,
still-untested-post-OOM-fix fallback (`max_shape_buckets=4`/`attn_sq_budget=31_000_000`, found
necessary via a real `RESOURCE_EXHAUSTED` at `Lb=1744/B=32`, never re-verified since), but is no
longer the only option. `hmn_tpu_recall1024_flat_rope.py`/`_noyarn.py`/`_noyarn_nockpt.py`
(the torch_xla ablation trio) stay in the repo as the investigation record.

---

## Key Principles

- `null_kv=True` always — 1.5–2× faster, better bpb
- Mask convention: `0.0`=attend, `-1e9`=blocked (additive bias for `F.scaled_dot_product_attention`)
- Round 0 (initial) before refine rounds — always required for the feedback mechanism
- The nochain blackout (each chain step's round-0 STATE blocked from ALL tokens in prior rec_blocks) is what makes chain steps independently trainable, and the relay exception (`hop`'s `hops`-controlled attention permission, `chunk_mask_fb_hop`/`chunk_mask_fb_traj`) is the sole sanctioned carve-out from it. Do not weaken either without understanding the consequences — the nochain blackout is what keeps chain steps from leaking raw content forward.
- **Always verify masking changes against the actual attention-mask matrix, not just "does it run"** — a smoke test that completes without crashing says nothing about whether the intended access pattern is actually being enforced (the `hop` encoding-pass leak ran and trained "successfully" for the whole time it was unintentionally leaky). Check specific (row, col) blocks directly, per chain step/op, before trusting a masking change.
- **Report precisely, never round up** — state exactly what was measured (e.g. a padded/truncated excerpt), not the whole file
- **Verify infra before trusting it mid-run**: always check process liveness (`ps -p <pid>`) explicitly on every wake, not just log content — a silently-exited process produces no new log lines
- **Never run two training jobs at once**
- **Editing `kvmem/hmn.py` while a training job is running is safe** — Python has already loaded the module into the live process's memory; on-disk edits don't affect it. Verified multiple times this session (deleting `h_inject`, the vocab reorder) without disrupting an in-progress run.

---

## Docs

| What | Where |
|------|-------|
| **`docs/HMN_RECIPE.md`** — quickstart + current-state-only reference (model architecture, the E/S/Q trajectory DSL, the relay, val/test mechanics) for a newcomer with zero context | [`docs/HMN_RECIPE.md`](docs/HMN_RECIPE.md) |
| **`docs/HISTORY.md`** — the full narrative: every design decision, terminology evolution, the deleted `relay`/`STATE_QUEUE` mechanism, the vocab reorder, structured-data track detail, compression diagnostics design, a classical/non-DNN alternatives discussion | [`docs/HISTORY.md`](docs/HISTORY.md) |
| **`docs/TRC_TPU.md`** — planning estimate (not yet implemented) for TPU Research Cloud access: TRC tier specs (v2/v3/v4-8 HBM), 1M/10M/50M-param architecture options sized to this project's `single_attn` param formula, per-tier batch-size ballparks, sequence-packing tradeoffs, bf16 caveats, and the eager-PyTorch→XLA/JAX porting prerequisite | [`docs/TRC_TPU.md`](docs/TRC_TPU.md) |
| The rewrite plan (original design/approval record — every naming decision, worked `STATE_QUEUE` example predating the `hop` mechanism, why each choice was made) | [`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`](/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md) |
| Current implementation | [`kvmem/hmn.py`](kvmem/hmn.py) (single file), active configs in [`kvmem/configs/`](kvmem/configs/) (completed/superseded configs archived under [`kvmem/configs/archive/`](kvmem/configs/archive/)), structured-data generators in [`kvmem/structured_data.py`](kvmem/structured_data.py), compression diagnostics in [`kvmem/eval_compression.py`](kvmem/eval_compression.py), trajectory-generalization diagnostics in [`kvmem/eval_weave.py`](kvmem/eval_weave.py), positional-shortcut diagnostics in [`kvmem/probe_positional_shortcut.py`](kvmem/probe_positional_shortcut.py) (behavioral swap test, `batch`/`interleave_delayed` shapes) and [`kvmem/probe_mechanistic_addressing.py`](kvmem/probe_mechanistic_addressing.py) (attention-mass + gradient-saliency counterpart, uses `MHAttention.capture_attn` in `hmn.py`); the suffix-recall/stitch design's own single-query shape has its own pair, [`kvmem/probe_stitch_content_addressing.py`](kvmem/probe_stitch_content_addressing.py) (behavioral swap test) and [`kvmem/probe_stitch_mechanistic_addressing.py`](kvmem/probe_stitch_mechanistic_addressing.py) (mechanistic counterpart, adapted for `hops=-1` routing) |
| `kvmem/hmn.py` dated snapshots (`hmn_v1_backup.py` through `hmn_v4_backup.py` — pre-cleanup draft, post-cleanup, pre-DSL/repeat_batch/stitch feature work, and the pre-promotion old tagged design respectively) were **deleted** (not archived) once the promoted `kvmem/hmn.py` stabilized — they were pure diffing artifacts, never imported by anything, and their content is superseded by the current file plus this doc's own narrative (Results section, `docs/HISTORY.md` §15). `archive_v1/` remains the actual archival record for pre-rewrite code/docs. | — |
| Everything from before the rewrite (dual-attn discovery, RMSNorm, stitching, `juz1.txt` design, MDL theory, all prior architecture history — code AND docs) | [`archive_v1/`](archive_v1/) — old `kvmem/`, old `experiments/`, old `docs/` (`SRS_RECIPE.md`, `EARLY_ARCHITECTURE_HISTORY.md`, `MDL_MODEL_SIZE.md`, etc. all moved here, `docs/` at the repo root is a fresh start for this rewrite going forward) |
| Previous version of this file | [`archive_v1/CLAUDE_v1.md`](archive_v1/CLAUDE_v1.md) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
| `juz1` scaling target (not yet used in training) | [`datasets/juz1.txt`](datasets/juz1.txt) |
