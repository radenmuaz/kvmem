"""
Continuation fix for srs_stitch_nc8_slot8's window-G failure.

Diagnosis (qualitative decode inspection of stage0_end.pt): windows A-F reached
a clean IQ(wrong)->IR1(100%)->IR2(100%) pattern. Window G (last, hardest)
reached IQ(0%)->IR1(100%)->IR2(4.2%) — IR1 already recovers full correctness,
but IR2 DESTROYS it. This is the same "IR2 destroys IR1's gain" pathology
documented elsewhere in this project (down_counter, 64B scale) — wrong_token_
weight only upweights loss where the fed-back argmax is WRONG; it does nothing
to protect already-correct positions from being overwritten by IR2's transform.

Root cause: window G's IR1 only becomes reliably correct very late in the
60k-step run (after A-F have mostly converged and absorbed the LR budget), by
which point cosine_T0=60000's single-cycle LR has decayed to ~1e-6 — leaving
no gradient room to teach IR2 "leave this alone, it's already right" in that
regime. windows A-F didn't hit this because they became reliably correct
earlier, while LR was still high enough to adapt IR2's behavior for them.

Fix: warm-start from stage0_end.pt (60000 steps, best=80.8%) and run a FRESH
short cosine cycle at a lower peak LR (5e-5, vs the original 1.5e-4) — gives
the model renewed gradient signal specifically in the "IR1 already correct"
regime that window G only reached at the very end of the original run.

If this does not resolve window G within 20k steps, the next fix to try is
oversampling window G specifically (2x weight in the schedule) rather than
more uniform continuation — not attempted first since the current design has
no traj_mix/weighted-sampling machinery (every window is in the schedule
every step already), so oversampling would require restructuring the
single-fixed-schedule sequence into multiple schedule variants, a bigger
change than this LR-based first attempt.

Run:
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_stitch_nc8_slot8_continue.py \\
        --pretrained experiments/srs_tagged/logs/srs_stitch_nc8_slot8/checkpoints/stage0_end.pt \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_stitch_nc8_slot8_continue/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=290,
    lr_max=5e-5, lr_min=1e-7, wd=0.001,
    warmup_steps=200, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='srs_stitch_nc8_slot8_continue', seed=49,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=8, chunk_len=16, n_refine=2, B=6, n_steps=20000, eval_every=2000,
             windows=[(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)],
             eval_mode='stitch'),
    ],
)
