"""PSO with the manuscript energy-aware objective, also supporting vector targets."""
import numpy as np
def evaluate_population(daylight_array, lamp_arrays, pos, target_lux):
    """
    daylight_array: [n_points]
    lamp_arrays   : [n_lamps, n_points]
    pos           : [n_particles, n_lamps]
    """
    artificial = pos @ lamp_arrays
    total = artificial + daylight_array[None, :]
    error = total - np.asarray(target_lux, dtype=np.float32)
    mse = np.mean(error ** 2, axis=1)
    mean_dimming_ratio = np.mean(pos, axis=1)
    # Manuscript objective: MSE * (1 + mean dimming ratio).
    obj = mse * (mean_dimming_ratio + 1)

    return (
        obj.astype(np.float32),
        artificial.astype(np.float32),
        total.astype(np.float32),
        error.astype(np.float32),
    )


def pso_optimize_lighting_fast(
    daylight_array,
    lamp_arrays,
    target_lux=500.0,
    lower_bound=0.0,
    upper_bound=1.0,
    n_particles=100,
    n_iters=50,
    w=0.72,
    c1=1.49,
    c2=1.49,
    seed=42,
):
    daylight_array = np.asarray(daylight_array, dtype=np.float32).ravel()
    lamp_arrays = np.asarray(lamp_arrays, dtype=np.float32)

    n_lamps, _ = lamp_arrays.shape
    rng = np.random.RandomState(seed)

    pos = lower_bound + (upper_bound - lower_bound) * rng.rand(
        n_particles,
        n_lamps,
    ).astype(np.float32)
    vel = np.zeros((n_particles, n_lamps), dtype=np.float32)

    pbest_pos = pos.copy()
    pbest_val, _, _, _ = evaluate_population(
        daylight_array,
        lamp_arrays,
        pos,
        target_lux,
    )

    gbest_idx = int(np.argmin(pbest_val))
    gbest_val = float(pbest_val[gbest_idx])
    gbest_pos = pbest_pos[gbest_idx].copy()

    history = [gbest_val]
    generation_best_records = [
        np.concatenate([[0.0, gbest_val], gbest_pos])
    ]

    for gen in range(1, n_iters + 1):
        r1 = rng.rand(n_particles, n_lamps).astype(np.float32)
        r2 = rng.rand(n_particles, n_lamps).astype(np.float32)

        vel = (
            w * vel
            + c1 * r1 * (pbest_pos - pos)
            + c2 * r2 * (gbest_pos[None, :] - pos)
        )
        pos = np.clip(pos + vel, lower_bound, upper_bound)

        vals, _, _, _ = evaluate_population(
            daylight_array,
            lamp_arrays,
            pos,
            target_lux,
        )

        improved = vals < pbest_val
        pbest_val[improved] = vals[improved]
        pbest_pos[improved] = pos[improved]

        gbest_idx = int(np.argmin(pbest_val))
        gbest_val = float(pbest_val[gbest_idx])
        gbest_pos = pbest_pos[gbest_idx].copy()

        history.append(gbest_val)
        generation_best_records.append(
            np.concatenate([[float(gen), gbest_val], gbest_pos])
        )

    best_val, best_artificial_all, best_total_all, best_error_all = evaluate_population(
        daylight_array,
        lamp_arrays,
        gbest_pos[None, :],
        target_lux,
    )

    return {
        "best_ratios": gbest_pos.astype(np.float32),
        "best_objective": float(best_val[0]),
        "best_artificial": best_artificial_all[0],
        "best_total": best_total_all[0],
        "best_error": best_error_all[0],
        "history": np.asarray(history, dtype=np.float32),
        "generation_best_records": np.asarray(generation_best_records, dtype=np.float32),
    }
