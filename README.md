# kvmem — Fast-Weight Language Model

## Vision

**A model that reads any document once and answers any question about it — without storing the document, without backprop, and without retraining.**

A compressed fast-weight register is built from whatever was ingested. At inference, base weights are frozen. Reading = forward passes that update the register. Querying = next-token prediction from a warmup prefix.

**The theoretical ceiling:**
> Ingest any corpus forward-pass-only, then answer arbitrary queries — at quality comparable to a full-context LLM — but in O(register width) memory instead of O(corpus length) KV cache.

This vision hasn't changed since the project started; the architecture implementing it has gone through several redesigns as each one hit a real limitation. See below for where things stand now.

---

## Current status — read `CLAUDE.md` first

**[`CLAUDE.md`](CLAUDE.md)** is the living instructions/status doc for this repo (agent-facing, but also the best human-facing summary of current architecture, terminology, and open questions) — read it before this file for anything beyond the vision above.

Short version: the current implementation is **[`kvmem/hmn.py`](kvmem/hmn.py)**, a single consolidated file (chat-tag-style vocabulary, three selectable transformer block types, a training loop with bounded chain memory) — a from-scratch rewrite that replaced an earlier multi-file `kvmem/`+`experiments/` stack after a design review found the old chat-tag vocabulary encoded window identity into the token vocabulary itself (fixable, but required retraining, so it was also the point to add bounded persistent memory across chunks from the start rather than bolt it on later).

**Everything from before that rewrite — including the RNN-style `<h>`-state design this README originally described, the dual-attention-block discovery, the chunk-memorization/SRS scaling work, and every prior experiment's code AND docs — is preserved, not deleted**, under [`archive_v1/`](archive_v1/) (old `kvmem/`, old `experiments/`, old `docs/`; still runs standalone via `PYTHONPATH=archive_v1`). The historical design docs (`archive_v1/docs/SRS_RECIPE.md`, `EARLY_ARCHITECTURE_HISTORY.md`, `MDL_MODEL_SIZE.md`) are still the most detailed record of *why* many of the current design's proven mechanisms (nochain masking, warmup-seeded stitching, IQ-before-IR staging, RoPE necessity) work the way they do — `kvmem/hmn.py` reuses that proven logic, not just the vision. The `docs/` folder at the repo root is a fresh start for the current rewrite — see [`docs/HMN_RECIPE.md`](docs/HMN_RECIPE.md) for the primary detailed writeup.

---

## Code layout

```
kvmem/
  hmn.py            — current implementation: vocab, position/mask builders,
                       model (3 block types), training loop, chain memory
  configs/           — current training configs (Stage 0, Stage 1, ...)
  structured_data.py — compressible synthetic data generators (chaotic
                       maps, fractals, cellular automata), queued track
  eval_compression.py — test-time compression-quality diagnostics
archive_v1/
  kvmem/             — everything before the rewrite (train_hmn_chunk.py,
                       model.py, data.py, and older still — kvmem/old/v1/)
  experiments/       — attn_dual, chat_tags, srs_tagged, densenet_kv
  docs/              — every doc from before the rewrite (SRS_RECIPE.md,
                       EARLY_ARCHITECTURE_HISTORY.md, MDL_MODEL_SIZE.md, ...)
  CLAUDE_v1.md        — previous version of CLAUDE.md
docs/
  HMN_RECIPE.md       — primary detailed writeup for the current architecture
datasets/
  suratalfatihah.txt — test set
  juz1.txt            — scaling target (not yet used in training)
```

Run a config:
```bash
python3 -m kvmem.hmn --config kvmem/configs/hmn_stage0_round0_single.py --device mps
```

---

## Training objective

**NTP on output positions only.** Source NTP would train reconstruction, not retrieval — the wrong objective for random-data recall, which is the proven task this project trains and evaluates on (see `CLAUDE.md`/`docs/HMN_RECIPE.md` for why the dataset is deliberately infinite random bytes for validation, with real text and now a structured-data track reserved for the generalization tests).
