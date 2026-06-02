For a PID stored in $PID:

Show where stdout/stderr are going (step 3):

ls -l /proc/$PID/fd/{1,2}

Or with labels:

echo "stdout -> $(readlink -f /proc/$PID/fd/1)"; echo "stderr -> $(readlink -f /proc/$PID/fd/2)"

Follow stdout and stderr directly (step 4):

tail -f /proc/$PID/fd/{1,2}

If you don't know the PID and want to find it by process name:

PID=$(pgrep -f myscript | head -1) && tail -f /proc/$PID/fd/{1,2}

Or inspect the destinations first:

PID=$(pgrep -f myscript | head -1) && ls -l /proc/$PID/fd/{1,2}

Mac:

lsof -p $PID -a -d 1,2

If you know the PID:

lsof -p $PID -a -d 1,2

To print just the stdout/stderr targets:

lsof -p $PID -a -d 1,2 -Fn | grep '^n'

If you want to immediately tail the file that stdout is writing to:

tail -f "$(lsof -p $PID -a -d 1 -Fn | sed -n 's/^n//p')"

Find by process name and tail stdout:

tail -f "$(lsof -p $(pgrep -f myprocess | head -1) -a -d 1 -Fn | sed -n 's/^n//p')"

For a typical nohup process, the fastest check is often:

tail -f nohup.out

since that's where stdout/stderr usually end up unless explicitly redirected elsewhere.

| NLL (nats/byte) | BPB (bits/byte) |   PPL |
| --------------: | --------------: | ----: |
|           5.545 |             8.0 | 256.0 |
|           4.852 |             7.0 | 128.0 |
|           4.159 |             6.0 |  64.0 |
|           3.466 |             5.0 |  32.0 |
|           2.773 |             4.0 |  16.0 |
|           2.079 |             3.0 |   8.0 |
|           1.386 |             2.0 |   4.0 |
|           0.693 |             1.0 |   2.0 |
|           0.000 |             0.0 |   1.0 |

mkdir -p /tmp/kvmem_runs
python -m kvmem.train_role --curriculum none --seg-len 16 --slot-len 16 --warmup-len 4 --out-len 4 --steps 40000 --eval-every 5000 --log-every 2000 --d 64 --n-layers 4 --B 16 --lr 3e-4 --no-grok --cycle-steps 0 --device mps --log-dir logs --seed 42 --name adamw_flat > /tmp/kvmem_runs/adamw_flat.log 2>&1 && \
python -m kvmem.train_role --curriculum none --seg-len 16 --slot-len 16 --warmup-len 4 --out-len 4 --steps 40000 --eval-every 5000 --log-every 2000 --d 64 --n-layers 4 --B 16 --lr 3e-4 --no-grok --cycle-steps 40000 --device mps --log-dir logs --seed 42 --name adamw_cosine > /tmp/kvmem_runs/adamw_cosine.log 2>&1 && \
python -m kvmem.train_role --curriculum none --seg-len 16 --slot-len 16 --warmup-len 4 --out-len 4 --steps 40000 --eval-every 5000 --log-every 2000 --d 64 --n-layers 4 --B 16 --lr 3e-4 --cycle-steps 0 --device mps --log-dir logs --seed 42 --name grok_flat > /tmp/kvmem_runs/grok_flat.log 2>&1 && \
python -m kvmem.train_role --curriculum none --seg-len 16 --slot-len 16 --warmup-len 4 --out-len 4 --steps 40000 --eval-every 5000 --log-every 2000 --d 64 --n-layers 4 --B 16 --lr 3e-4 --cycle-steps 40000 --device mps --log-dir logs --seed 42 --name grok_cosine > /tmp/kvmem_runs/grok_cosine.log 2>&1
echo "ALL DONE"