import time
import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np

N = 4096
STEPS = 50
WARMUP = 5


def bench_jax_cpu():
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        x = jax.random.normal(jax.random.key(0), (N, N))
        w = jax.random.normal(jax.random.key(1), (N, N))

        @jax.jit
        def step(x, w):
            return jnp.tanh(x @ w)

        for _ in range(WARMUP):
            y = step(x, w)
        y.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(STEPS):
            y = step(x, w)
        y.block_until_ready()
    return (time.perf_counter() - t0) / STEPS * 1000


def bench_jax_mps():
    cpu = jax.devices("cpu")[0]
    mps = jax.devices("mps")[0]
    with jax.default_device(cpu):
        x_cpu = jax.random.normal(jax.random.key(0), (N, N))
        w_cpu = jax.random.normal(jax.random.key(1), (N, N))
    x = jax.device_put(x_cpu, mps)
    w = jax.device_put(w_cpu, mps)

    with jax.default_device(mps):
        @jax.jit
        def step(x, w):
            return jnp.tanh(x @ w)

        for _ in range(WARMUP):
            y = step(x, w)
        y.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(STEPS):
            y = step(x, w)
        y.block_until_ready()
    return (time.perf_counter() - t0) / STEPS * 1000


def bench_mlx(device):
    mx.set_default_device(device)
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((N, N)).astype(np.float32))
    w = mx.array(rng.standard_normal((N, N)).astype(np.float32))
    mx.eval(x, w)

    def step(x, w):
        return mx.tanh(x @ w)

    for _ in range(WARMUP):
        y = step(x, w)
        mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(STEPS):
        y = step(x, w)
        mx.eval(y)
    return (time.perf_counter() - t0) / STEPS * 1000


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print(f"JAX {jax.__version__}  |  MLX {mx.__version__}")
    print(f"Matrix size: {N}x{N}  |  Steps: {STEPS}\n")

    results = {}

    print("JAX CPU (JIT)...")
    results["JAX CPU"] = bench_jax_cpu()
    print(f"  {results['JAX CPU']:.2f} ms/step")

    print("JAX MPS (JIT)...")
    results["JAX MPS"] = bench_jax_mps()
    print(f"  {results['JAX MPS']:.2f} ms/step")

    print("MLX CPU...")
    results["MLX CPU"] = bench_mlx(mx.Device(mx.cpu))
    print(f"  {results['MLX CPU']:.2f} ms/step")

    print("MLX GPU (Metal)...")
    results["MLX GPU"] = bench_mlx(mx.Device(mx.gpu))
    print(f"  {results['MLX GPU']:.2f} ms/step")

    baseline = results["JAX CPU"]
    print("\n--- Results ---")
    for name, ms in results.items():
        bar = "#" * int(50 / ms * baseline / baseline * (baseline / ms))
        print(f"  {name:<12} {ms:7.2f} ms   {baseline/ms:5.2f}x  {bar}")
