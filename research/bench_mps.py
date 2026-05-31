import time
import jax
import jax.numpy as jnp

def bench(device, n=4096, steps=50, warmup=5, use_jit=True):
    # create on CPU then transfer, since random may not work on all backends
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        x_cpu = jax.random.normal(jax.random.key(0), (n, n))
        w_cpu = jax.random.normal(jax.random.key(1), (n, n))
    x = jax.device_put(x_cpu, device)
    w = jax.device_put(w_cpu, device)
    with jax.default_device(device):

        def step(x, w):
            return jnp.tanh(x @ w)

        fn = jax.jit(step) if use_jit else step

        for _ in range(warmup):
            y = fn(x, w)
        y.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(steps):
            y = fn(x, w)
        y.block_until_ready()
        elapsed = time.perf_counter() - t0

    return elapsed / steps * 1000

if __name__ == "__main__":
    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())
    print()

    cpu = jax.devices("cpu")[0]
    print(f"Benchmarking CPU ({cpu})...")
    cpu_ms = bench(cpu)
    print(f"  {cpu_ms:.2f} ms/step")

    try:
        mps = jax.devices("mps")[0]
        print(f"\nBenchmarking MPS ({mps}) with JIT...")
        try:
            mps_ms = bench(mps, use_jit=True)
            print(f"  {mps_ms:.2f} ms/step  (JIT)")
            print(f"  Speedup vs CPU: {cpu_ms/mps_ms:.2f}x")
        except Exception as e:
            print(f"  JIT failed: {e}")
            print(f"  Falling back to eager mode...")
            mps_ms = bench(mps, use_jit=False)
            print(f"  {mps_ms:.2f} ms/step  (eager)")
            print(f"  Speedup vs CPU (JIT) / MPS (eager): {cpu_ms/mps_ms:.2f}x")
    except Exception as e:
        print(f"\nMPS device not available: {e}")
