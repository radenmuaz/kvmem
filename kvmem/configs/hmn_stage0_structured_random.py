"""Structured-data ablation, `data_kind='random'` — the high-entropy baseline
control (no exploitable structure at all, `data_target_bits` inapplicable/
ignored for this kind). Same stage0 shape/fixed hparams as
`hmn_stage0_structured_markov.py`/`_ca.py` — see
`hmn_stage0_structured_ablation.py` for full rationale. This is the existing
random-byte task every other config in this project already trains against;
included here explicitly as the same-shape baseline so markov/ca results are
read as a delta against it, not in isolation."""
from kvmem.configs.hmn_stage0_structured_ablation import base_hp

hp = base_hp('random')
hp['data_target_bits'] = None
