#!/usr/bin/env bash
# Installs JAX + libtpu + Flax/optax on a fresh TPU VM, for kvmem/hmn_jax.py
# (the JAX/Flax NNX port — see that file's own module docstring and
# CLAUDE.md's "TPU port" section for why it exists: an independent data
# point on bug 5, the unresolved data-dependent loss=NaN on torch_xla).
#
# Verified on tpu2 (v6e-1, europe-west4-a) starting from a completely fresh
# VM (no ML packages installed at all) — `jax.devices()` correctly returned
# a TpuDevice afterward.
#
# Run ON the TPU VM itself (not locally):
#   gcloud compute tpus tpu-vm ssh <name> --zone=<zone> --command="bash -s" < kvmem/setup_tpu_jax.sh
# or copy it over first and run directly:
#   gcloud compute tpus tpu-vm scp kvmem/setup_tpu_jax.sh <name>:~/ --zone=<zone>
#   gcloud compute tpus tpu-vm ssh <name> --zone=<zone> --command="bash setup_tpu_jax.sh"
set -euo pipefail

# No flax version pin: newer flax (>=0.11) requires Python 3.11+, and TPU VM
# images have shipped Python 3.10 in practice (verified on tpu2) — pinning a
# floor here just breaks the install outright instead of degrading gracefully.
# kvmem/hmn_jax.py itself detects the installed flax's nnx API surface
# (nnx.List availability) at runtime and adapts, so whatever pip resolves
# here works.
pip install -q -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html flax optax tpu-info tqdm

python3 -c "
import jax
print('JAX backend:', jax.default_backend())
print('JAX devices:', jax.devices())
assert jax.default_backend() == 'tpu', 'JAX did not pick up the TPU backend — check libtpu install above'
print('OK — JAX is using the TPU.')
"
echo "tpu-info: $(~/.local/bin/tpu-info --help >/dev/null 2>&1 && echo OK || echo 'installed, check PATH (~/.local/bin)')"
