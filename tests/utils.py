import time
import jax


def timeit(f, warmup=10, runs=100):
    """
    Measures average runtime (in seconds) of calling f().
    - warmup: number of warm-up calls (not measured).
    - runs:   number of calls for measurement.
    Returns: average runtime over the measured runs.
    """
    # Warm-up
    for _ in range(warmup):
        out = f()
    jax.block_until_ready(out)

    # Actual timing
    t0 = time.perf_counter()
    for _ in range(runs):
        out = f()
    jax.block_until_ready(out)
    t1 = time.perf_counter()

    return (t1 - t0) / runs