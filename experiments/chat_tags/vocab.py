"""
Chat-tags experiment — explicit boundary tokens for src/mem/query/response regions.

New IDs, disjoint from kvmem.data's HMN_* namespace (which stops at HMN_VOCAB_SIZE=268).
Kept in this experiment folder rather than appended to kvmem/data.py so the base
codebase is untouched — see /Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md.
"""
from kvmem.data import HMN_VOCAB_SIZE  # 268, read-only import for the base offset

HMN_SRC_OPEN       = HMN_VOCAB_SIZE + 0   # 268  <src>
HMN_SRC_CLOSE      = HMN_VOCAB_SIZE + 1   # 269  </src>
HMN_MEM_OPEN       = HMN_VOCAB_SIZE + 2   # 270  <mem>
HMN_MEM_CLOSE      = HMN_VOCAB_SIZE + 3   # 271  </mem>
HMN_QUERY_OPEN     = HMN_VOCAB_SIZE + 4   # 272  <query>       generic, used for uniform-X training
HMN_QUERY_CLOSE    = HMN_VOCAB_SIZE + 5   # 273  </query>
HMN_RESPONSE_OPEN  = HMN_VOCAB_SIZE + 6   # 274  <response>
HMN_RESPONSE_CLOSE = HMN_VOCAB_SIZE + 7   # 275  </response>

# Window-specific query tags (Phase B4) — replace the generic <query> pair only for
# the three canonical windows (warmup_x_fixed in {0,16,32}), giving the model an
# explicit "this recall is window A/B/C" identity signal instead of one shared
# anchor. Uniform-X training keeps the generic HMN_QUERY_OPEN/CLOSE above since
# arbitrary X doesn't map to one named window.
HMN_QUERY_A_OPEN   = HMN_VOCAB_SIZE + 8    # 276  <query_a>  window (0,2), X=0
HMN_QUERY_A_CLOSE  = HMN_VOCAB_SIZE + 9    # 277  </query_a>
HMN_QUERY_B_OPEN   = HMN_VOCAB_SIZE + 10   # 278  <query_b>  window (1,3), X=16
HMN_QUERY_B_CLOSE  = HMN_VOCAB_SIZE + 11   # 279  </query_b>
HMN_QUERY_C_OPEN   = HMN_VOCAB_SIZE + 12   # 280  <query_c>  window (2,4), X=32
HMN_QUERY_C_CLOSE  = HMN_VOCAB_SIZE + 13   # 281  </query_c>

HMN_TAG_VOCAB_SIZE = HMN_VOCAB_SIZE + 8    # 276 — unchanged, window-query IDs added on top
HMN_TAG_VOCAB_SIZE_V2 = HMN_VOCAB_SIZE + 14  # 282 — Phase B4 vocab size (includes window tags)

# Map a canonical warmup_x_fixed value to its (open, close) window-query tag pair.
WINDOW_QUERY_TAGS = {
    0:  (HMN_QUERY_A_OPEN, HMN_QUERY_A_CLOSE),
    16: (HMN_QUERY_B_OPEN, HMN_QUERY_B_CLOSE),
    32: (HMN_QUERY_C_OPEN, HMN_QUERY_C_CLOSE),
}
