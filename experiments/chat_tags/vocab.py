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
HMN_QUERY_OPEN     = HMN_VOCAB_SIZE + 4   # 272  <query>
HMN_QUERY_CLOSE    = HMN_VOCAB_SIZE + 5   # 273  </query>
HMN_RESPONSE_OPEN  = HMN_VOCAB_SIZE + 6   # 274  <response>
HMN_RESPONSE_CLOSE = HMN_VOCAB_SIZE + 7   # 275  </response>

HMN_TAG_VOCAB_SIZE = HMN_VOCAB_SIZE + 8   # 276
