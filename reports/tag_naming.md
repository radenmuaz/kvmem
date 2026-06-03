# Tag Naming

## Chosen: Set D — DB-style minimal

```
<d>   data      (source to ingest)
<e>   extract   (ponder — pre-process before compression)
<k>   key       (memory slots — compressed KV index)
<q>   query     (anchor/warmup — lookup predicate)
<v>   value     (output — returned result)
```

Reads as: **data → extract → key → query → value**.
Maps to: *ingest document → build index → store as keys → issue query → return value*.

`<k>` for memory slots is precise — slots literally become the K matrix during attention recall.
`<v>` for output is precise — the model retrieves a value from the KV index given a query.

---

## Alternatives for the extract position

`<x>` was the original choice but conflicts with standard notation (`x_t` = input in RNN/ML).

| Tag | Meaning | Notes |
|-----|---------|-------|
| `<e>` | extract | **chosen** — clean, unambiguous |
| `<h>` | hash | good DB connotation (hash index build) |
| `<t>` | transform | readable, but `t` = timestep in some notations |
| `<z>` | latent/intermediate | VAE/diffusion connotation (z = latent variable) |
| `<b>` | build | "build index" — clear but less common single-char |
| `<n>` | encode | `n` for eNcode — avoids all common conflicts |
| `<a>` | analyse/aggregate | DB aggregate step; but `a` sometimes = attention |

`<e>` wins: "extract features before compression" is the correct description, no standard conflicts.

---

## Other naming sets considered

**RNN variables** — `<x><z><h><q><y>`:
maps to x_t (input), z (intermediate), h_t (hidden state), q (query), y_t (output).
Clean for researchers familiar with sequence models.

**CPU pipeline** — `<fe><dc><ex><wb><io>`:
fetch/decode/execute/writeback/output. Accurate for the pipeline metaphor
but verbose for a token vocabulary.

**I/O stream** — `<src><flt><buf><seek><drn>`:
source/filter/buffer/seek/drain. Evocative but too long for single-token tags.

**Verb-based** — `<r><t><w><a><g>`:
read/think/write/anchor/generate. Readable as verbs but loses DB precision.

---

## Full sequence with chosen tags

```
<d> src_bytes </d>  <e> enc_0..enc_{P-1} </e>  <k> key_0..key_{N-1} </k>
<q> warmup </q>  <v> output </v>
```

Single-block, fully causal:
- `<e>` sees src (causal)
- `<k>` sees src + enc (causal)  
- `<q>/<v>` blocked from src and enc — bottleneck through `<k>` only
- `<v>` is write-only
