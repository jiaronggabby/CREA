from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from .graphs import all_to_all_graph, from_edges, identity_graph, save_graph
from .utils import PROCESSED_ROOT, atomic_json, ensure_dirs, save_series

V3_GENERATOR_VERSION = "selective_prior_synthetic_v3_dgp_1"
V3_LOCAL_COEFFICIENT = 0.20
V3_RELATION_COEFFICIENT = 0.65
V3_NOISE_SCALE = 0.12
V3_BURN_IN = 512
V3_REGIME_BLOCK = 256
V3_SHAM_BURN_IN = 300
V3_CORRUPTION_LEVELS = (0, 10, 25, 50, 75, 100)


def _stable_random_graph(n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a graph stable for the second-order simulation recurrences.

    The simulators use both ``x[t - 1]`` and ``x[t - 2] @ A``.  A graph
    spectral radius of 0.65 (the previous target) can therefore still make
    the characteristic root exceed one when combined with the AR term.  Keep
    the externally supplied graph and the data-generating operator identical,
    but cap the radius at 0.25 so long synthetic runs remain finite.
    """
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    weights = rng.uniform(0.05, 0.25, size=(n, n)) * mask
    # Ensure a directed ring so every node participates.
    for i in range(n):
        weights[i, (i + 1) % n] = rng.uniform(0.08, 0.18)
    spectral = max(abs(np.linalg.eigvals(weights)))
    if spectral > 0:
        weights *= min(0.25 / float(spectral), 1.0)
    return weights


def _simulate_propagation(T: int, A: np.ndarray, rng: np.random.Generator, burn: int = 300) -> np.ndarray:
    n = A.shape[0]
    x = np.zeros((T + burn, n), dtype=np.float64)
    for t in range(2, T + burn):
        seasonal = 0.2 * np.sin(2 * np.pi * t / 48 + np.arange(n) * 0.1)
        x[t] = 0.45 * x[t - 1] + x[t - 2] @ A + seasonal + rng.normal(0, 0.25, n)
    return x[burn:]


def _simulate_common_driver(T: int, n: int, rng: np.random.Generator, burn: int = 300) -> np.ndarray:
    total = T + burn
    z = np.zeros(total)
    x = np.zeros((total, n))
    phases = rng.uniform(0, 2 * np.pi, n)
    loadings = rng.uniform(0.6, 1.2, n)
    for t in range(2, total):
        z[t] = 0.85 * z[t - 1] - 0.10 * z[t - 2] + rng.normal(0, 0.4)
        shared = z[t] + 0.4 * np.sin(2 * np.pi * t / 48)
        x[t] = 0.50 * x[t - 1] + loadings * shared + 0.2 * np.sin(2 * np.pi * t / 24 + phases) + rng.normal(0, 0.3, n)
    return x[burn:]


def generate_synthetic(
    seed: int = 2026,
    T: int = 12000,
    n: int = 20,
    density: float = 0.10,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> None:
    ensure_dirs()
    rng = np.random.default_rng(seed)
    A = _stable_random_graph(n, density, rng)
    prop = _simulate_propagation(T, A, rng)
    neg = _simulate_common_driver(T, n, rng)
    decoy = _stable_random_graph(n, density, rng)
    for name, values, graph_matrix, truth in [
        ("synthetic_propagation", prop, A, "cross_node_propagation_present"),
        ("synthetic_common_driver", neg, decoy, "no_cross_node_propagation"),
    ]:
        out = PROCESSED_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        observed = np.ones_like(values, dtype=np.uint8)
        train_end, val_end = round(T * train_ratio), round(T * (train_ratio + val_ratio))
        save_series(out, values, observed, np.arange(T, dtype=np.int64), train_end, val_end)
        nodes = pd.DataFrame({"node_index": np.arange(n), "node_id": [f"v{i:02d}" for i in range(n)]})
        nodes.to_csv(out / "nodes.csv", index=False)
        edges = [(i, j, float(graph_matrix[i, j])) for i in range(n) for j in range(n) if graph_matrix[i, j] != 0]
        graph = from_edges(edges, n=n, add_self=True, directed=True)
        save_graph(out / "graph_physical.npz", graph)
        save_graph(out / "graph_identity.npz", identity_graph(n))
        save_graph(out / "graph_all.npz", all_to_all_graph(n))
        atomic_json(out / "metadata.json", {
            "dataset": name,
            "role": "positive_control" if name.endswith("propagation") else "negative_control",
            "ground_truth": truth,
            "seed": seed,
            "time_steps": T,
            "num_variables": n,
            "split": {"train_end": train_end, "val_end": val_end, "train_ratio": train_ratio, "val_ratio": val_ratio},
            "generation": {"density": density},
        })
        print(f"[prepared] {name}: {values.shape}")


def _simulate_switching_sources(
    T: int,
    prior: np.ndarray,
    alternative: np.ndarray,
    rng: np.random.Generator,
    burn: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate repeated local/common/prior/alternative relational regimes."""
    n = prior.shape[0]
    total = T + burn
    x = np.zeros((total, n), dtype=np.float64)
    z = np.zeros(total, dtype=np.float64)
    loadings = rng.uniform(0.5, 1.1, n)
    regimes = np.empty(total, dtype=np.int64)
    block = 192
    order = np.asarray([0, 2, 1, 3], dtype=np.int64)  # local, prior, common, alternative
    for t in range(total):
        regimes[t] = int(order[(t // block) % len(order)])
    for t in range(2, total):
        z[t] = 0.82 * z[t - 1] - 0.08 * z[t - 2] + rng.normal(0.0, 0.35)
        value = 0.48 * x[t - 1] - 0.08 * x[t - 2]
        if regimes[t] == 1:
            value += x[t - 2] @ prior
        elif regimes[t] == 2:
            value += loadings * (z[t] + 0.25 * np.sin(2 * np.pi * t / 48))
        elif regimes[t] == 3:
            value += x[t - 2] @ alternative
        value += 0.12 * np.sin(2 * np.pi * t / 24 + np.arange(n) * 0.07)
        x[t] = value + rng.normal(0.0, 0.25, n)
    return x[burn:], regimes[burn:]


def _nested_corruption_plan(
    graph_matrix: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Freeze one removal/addition order shared by the complete E3 curve."""
    matrix = np.asarray(graph_matrix, dtype=np.float64)
    n = matrix.shape[0]
    original = np.asarray(
        [(i, j) for i in range(n) for j in range(n) if i != j and matrix[i, j] != 0],
        dtype=np.int64,
    )
    original_set = {(int(i), int(j)) for i, j in original}
    complement = np.asarray(
        [
            (i, j)
            for i in range(n)
            for j in range(n)
            if i != j and (i, j) not in original_set
        ],
        dtype=np.int64,
    )
    if len(complement) < len(original):
        raise ValueError("E3 graph is too dense to replace every original endpoint without reuse")
    removal_order = original[rng.permutation(len(original))]
    replacement_order = complement[rng.permutation(len(complement))[: len(original)]]
    payload = removal_order.tobytes() + replacement_order.tobytes()
    return {
        "removal_order": removal_order,
        "replacement_order": replacement_order,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _corrupt_connectivity(
    graph_matrix: np.ndarray,
    fraction: float,
    plan: dict[str, object],
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Apply a prefix of a fixed plan, yielding a genuinely nested corruption curve."""
    matrix = np.asarray(graph_matrix, dtype=np.float64).copy()
    removal_order = np.asarray(plan["removal_order"], dtype=np.int64)
    replacement_order = np.asarray(plan["replacement_order"], dtype=np.int64)
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("Corruption fraction must lie in [0, 1]")
    change = min(round(float(fraction) * len(removal_order)), len(removal_order))
    for index in range(change):
        src, dst = (int(value) for value in removal_order[index])
        new_src, new_dst = (int(value) for value in replacement_order[index])
        weight = float(matrix[src, dst])
        if weight == 0.0:
            raise RuntimeError("Nested E3 plan attempted to remove an endpoint twice")
        matrix[src, dst] = 0.0
        if matrix[new_src, new_dst] != 0.0:
            raise RuntimeError("Nested E3 plan attempted to reuse an occupied endpoint")
        matrix[new_src, new_dst] = weight
    original_endpoints = {
        (i, j)
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
        if i != j and graph_matrix[i, j] != 0
    }
    corrupted_endpoints = {
        (i, j)
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
        if i != j and matrix[i, j] != 0
    }
    actual = (
        len(original_endpoints - corrupted_endpoints) / len(original_endpoints)
        if original_endpoints
        else 0.0
    )
    expected_actual = change / len(original_endpoints) if original_endpoints else 0.0
    if abs(actual - expected_actual) > 1e-12:
        raise RuntimeError(
            f"E3 realized endpoint change {actual} does not match planned prefix {expected_actual}"
        )
    rounding_tolerance = 0.5 / len(original_endpoints) if original_endpoints else 0.0
    if abs(actual - float(fraction)) > rounding_tolerance + 1e-12:
        raise RuntimeError(
            f"E3 endpoint change {actual} is incompatible with requested fraction {fraction}"
        )
    return matrix, {
        "requested_corruption_fraction": float(fraction),
        "actual_endpoint_change_fraction": float(actual),
        "replaced_endpoint_count": int(change),
        "original_endpoint_count": len(original_endpoints),
        "corruption_plan_sha256": str(plan["sha256"]),
    }


def _save_selective_dataset(
    name: str,
    values: np.ndarray,
    graph_matrix: np.ndarray,
    metadata: dict,
    prior_valid: np.ndarray | None = None,
    processed_root: Path | None = None,
    *,
    artifact_schema: str = "selective_prior_synthetic_v1",
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    T, n = values.shape
    out = (processed_root or PROCESSED_ROOT) / name
    out.mkdir(parents=True, exist_ok=True)
    observed = np.ones_like(values, dtype=np.uint8)
    train_end = round(0.60 * T)
    val_end = round(0.70 * T)
    save_series(out, values, observed, np.arange(T, dtype=np.int64), train_end, val_end)
    pd.DataFrame(
        {"node_index": np.arange(n), "node_id": [f"v{i:03d}" for i in range(n)]}
    ).to_csv(out / "nodes.csv", index=False)
    edges = [
        (src, dst, float(graph_matrix[src, dst]))
        for src in range(n)
        for dst in range(n)
        if graph_matrix[src, dst] != 0
    ]
    save_graph(out / "graph_physical.npz", from_edges(edges, n=n, add_self=True, directed=True))
    save_graph(out / "graph_identity.npz", identity_graph(n))
    save_graph(out / "graph_all.npz", all_to_all_graph(n))
    if prior_valid is not None:
        np.save(out / "ground_truth_prior_valid.npy", np.asarray(prior_valid, dtype=np.uint8), allow_pickle=False)
    for filename, array in (extra_arrays or {}).items():
        if not filename.endswith(".npy") or Path(filename).name != filename:
            raise ValueError(f"Synthetic extra array must be a plain .npy filename: {filename}")
        np.save(out / filename, np.asarray(array), allow_pickle=False)
    atomic_json(
        out / "metadata.json",
        {
            "dataset": name,
            "schema": artifact_schema,
            "time_steps": int(T),
            "num_variables": int(n),
            "split_contract": "60/10/10/10/10; standardization is refit by the experiment loader",
            **metadata,
        },
    )


def generate_selective_prior_synthetic(
    seed: int = 20260810,
    T: int = 12000,
    n: int = 50,
    density: float = 0.08,
    processed_root: Path | None = None,
    only: set[str] | None = None,
) -> list[str]:
    """Create E1--E3 synthetic artifacts without consulting forecast results."""
    ensure_dirs()
    rng = np.random.default_rng(int(seed))
    prior = _stable_random_graph(n, density, rng)
    alternative = _stable_random_graph(n, density, rng)
    switching, regimes = _simulate_switching_sources(T, prior, alternative, rng)
    if only is None or "spi_synthetic_switching" in only:
        _save_selective_dataset(
            "spi_synthetic_switching",
            switching,
            prior,
            {
                "experiment": "E1",
                "generation_seed": int(seed),
                "regime_labels": {"0": "local", "1": "prior_valid", "2": "common_driver", "3": "alternative_relation"},
            },
            prior_valid=(regimes == 1),
            processed_root=processed_root,
        )
    common = _simulate_common_driver(T, n, rng)
    sham = _stable_random_graph(n, density, rng)
    if only is None or "spi_synthetic_sham" in only:
        _save_selective_dataset(
            "spi_synthetic_sham",
            common,
            sham,
            {
                "experiment": "E2",
                "generation_seed": int(seed),
                "reference_prior": "sham graph with identical training exposure",
            },
            prior_valid=np.zeros(T, dtype=np.uint8),
            processed_root=processed_root,
        )
    propagation = _simulate_propagation(T, prior, rng)
    corruption_plan = _nested_corruption_plan(prior, rng)
    names = ["spi_synthetic_switching", "spi_synthetic_sham"]
    for percentage in (0, 10, 25, 50, 75, 100):
        name = f"spi_synthetic_corruption_{percentage:03d}"
        if only is not None and name not in only:
            names.append(name)
            continue
        corrupted, corruption_metadata = _corrupt_connectivity(
            prior,
            percentage / 100.0,
            corruption_plan,
        )
        _save_selective_dataset(
            name,
            propagation,
            corrupted,
            {
                "experiment": "E3",
                "generation_seed": int(seed),
                "corruption_curve": "nested_endpoint_replacement_without_original_endpoint_reentry",
                **corruption_metadata,
            },
            prior_valid=np.full(T, percentage == 0, dtype=np.uint8),
            processed_root=processed_root,
        )
        names.append(name)
    return names


def _contractive_relation_graph(
    n: int,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a sparse relation operator with unit induced 2-norm.

    Protocol v2 bounded only each matrix's spectral radius.  That does not
    bound transient growth or the joint spectral radius of a switched system.
    Normalizing the induced 2-norm gives the v3 recurrence an explicit
    stability bound under every regime ordering.
    """
    mask = rng.random((n, n)) < float(density)
    np.fill_diagonal(mask, False)
    weights = rng.uniform(0.20, 1.00, size=(n, n)) * mask
    for source in range(n):
        weights[source, (source + 1) % n] = rng.uniform(0.40, 1.00)
    operator_norm = float(np.linalg.norm(weights, ord=2))
    if not np.isfinite(operator_norm) or operator_norm <= 0.0:
        raise RuntimeError("Could not construct a nonzero contractive relation graph")
    return weights / operator_norm


def _v3_regimes(length: int, block: int) -> np.ndarray:
    order = np.asarray([0, 1, 2, 3], dtype=np.int64)
    return order[(np.arange(length, dtype=np.int64) // int(block)) % len(order)]


def _simulate_switching_sources_v3(
    T: int,
    prior: np.ndarray,
    alternative: np.ndarray,
    rng: np.random.Generator,
    *,
    burn: int = V3_BURN_IN,
    block: int = V3_REGIME_BLOCK,
    local_coefficient: float = V3_LOCAL_COEFFICIENT,
    relation_coefficient: float = V3_RELATION_COEFFICIENT,
    noise_scale: float = V3_NOISE_SCALE,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Stable switching DGP with an auditable relational signal component."""
    n = prior.shape[0]
    total = int(T) + int(burn)
    regimes = _v3_regimes(total, block)
    x = np.zeros((total, n), dtype=np.float64)
    z = np.zeros(total, dtype=np.float64)
    loadings = rng.uniform(0.60, 1.00, n)
    prior_component = np.zeros_like(x)
    alternative_component = np.zeros_like(x)
    common_component = np.zeros_like(x)
    noise_component = np.zeros_like(x)
    for t in range(1, total):
        z[t] = 0.70 * z[t - 1] + rng.normal(0.0, 0.20)
        value = float(local_coefficient) * x[t - 1]
        if regimes[t] == 1:
            prior_component[t] = float(relation_coefficient) * (x[t - 1] @ prior)
            value += prior_component[t]
        elif regimes[t] == 2:
            common_component[t] = 0.32 * loadings * z[t]
            value += common_component[t]
        elif regimes[t] == 3:
            alternative_component[t] = float(relation_coefficient) * (
                x[t - 1] @ alternative
            )
            value += alternative_component[t]
        value += 0.05 * np.sin(2 * np.pi * t / 24 + np.arange(n) * 0.07)
        noise_component[t] = rng.normal(0.0, float(noise_scale), n)
        x[t] = value + noise_component[t]
    sl = slice(int(burn), None)
    return x[sl], regimes[sl], {
        "prior": prior_component[sl],
        "alternative": alternative_component[sl],
        "common": common_component[sl],
        "noise": noise_component[sl],
    }


def _simulate_propagation_v3(
    T: int,
    prior: np.ndarray,
    rng: np.random.Generator,
    *,
    burn: int = V3_BURN_IN,
    local_coefficient: float = V3_LOCAL_COEFFICIENT,
    relation_coefficient: float = V3_RELATION_COEFFICIENT,
    noise_scale: float = V3_NOISE_SCALE,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = prior.shape[0]
    total = int(T) + int(burn)
    x = np.zeros((total, n), dtype=np.float64)
    relation = np.zeros_like(x)
    noise = np.zeros_like(x)
    for t in range(1, total):
        relation[t] = float(relation_coefficient) * (x[t - 1] @ prior)
        seasonal = 0.05 * np.sin(2 * np.pi * t / 24 + np.arange(n) * 0.07)
        noise[t] = rng.normal(0.0, float(noise_scale), n)
        x[t] = float(local_coefficient) * x[t - 1] + relation[t] + seasonal + noise[t]
    sl = slice(int(burn), None)
    return x[sl], {"prior": relation[sl], "noise": noise[sl]}


def _stable_regime_evaluable(
    regimes: np.ndarray,
    *,
    history: int,
    max_horizon: int,
) -> np.ndarray:
    """Mark targets whose complete observable context is in one regime."""
    labels = np.asarray(regimes, dtype=np.int64)
    width = int(history) + int(max_horizon)
    evaluable = np.zeros(len(labels), dtype=np.uint8)
    for target in range(width - 1, len(labels)):
        window = labels[target - width + 1 : target + 1]
        evaluable[target] = int(np.all(window == window[-1]))
    return evaluable


def _v3_generation_contract(
    values: np.ndarray,
    relation_component: np.ndarray,
    noise_component: np.ndarray,
    *,
    regimes: np.ndarray | None,
    local_coefficient: float,
    relation_coefficient: float,
) -> dict[str, float | bool]:
    values = np.asarray(values, dtype=np.float64)
    relation = np.asarray(relation_component, dtype=np.float64)
    noise = np.asarray(noise_component, dtype=np.float64)
    selected = np.ones(len(values), dtype=bool)
    if regimes is not None:
        selected = np.asarray(regimes, dtype=np.int64) == 1
    relation_rms = float(np.sqrt(np.mean(np.square(relation[selected]))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise[selected]))))
    value_rms = float(np.sqrt(np.mean(np.square(values[selected]))))
    ratio = relation_rms / max(noise_rms, 1e-12)
    fraction = float(np.mean(np.square(relation[selected]))) / max(value_rms**2, 1e-12)
    regime_scale_ratio = 1.0
    if regimes is not None:
        scales = [
            float(np.std(values[np.asarray(regimes) == label]))
            for label in sorted(set(np.asarray(regimes, dtype=int).tolist()))
        ]
        regime_scale_ratio = max(scales) / max(min(scales), 1e-12)
    maximum_absolute_value = float(np.max(np.abs(values)))
    joint_norm_upper_bound = float(local_coefficient + relation_coefficient)
    checks = {
        "finite": bool(np.isfinite(values).all()),
        "maximum_absolute_value_below_25": maximum_absolute_value < 25.0,
        "joint_update_norm_below_0_90": joint_norm_upper_bound < 0.90,
        "relation_to_noise_rms_at_least_0_35": ratio >= 0.35,
        "relational_energy_fraction_at_least_0_05": fraction >= 0.05,
        "regime_scale_ratio_below_4": regime_scale_ratio < 4.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Protocol v3 synthetic generation contract failed: {checks}")
    return {
        **checks,
        "maximum_absolute_value": maximum_absolute_value,
        "joint_update_norm_upper_bound": joint_norm_upper_bound,
        "relation_rms": relation_rms,
        "noise_rms": noise_rms,
        "relation_to_noise_rms": ratio,
        "relational_energy_fraction": fraction,
        "regime_scale_ratio": regime_scale_ratio,
    }


def selective_prior_v3_generation_spec(
    *,
    seed: int,
    time_steps: int,
    variables: int,
    density: float,
    history: int,
    max_horizon: int,
) -> dict[str, object]:
    """Return the complete, model-independent v3 data-generation contract."""
    return {
        "schema": "selective_prior_generation_spec_v3",
        "generator_version": V3_GENERATOR_VERSION,
        "seed": int(seed),
        "time_steps": int(time_steps),
        "variables": int(variables),
        "density": float(density),
        "history_contract": int(history),
        "maximum_horizon_contract": int(max_horizon),
        "split_contract": [0.60, 0.10, 0.10, 0.10, 0.10],
        "relation_operator_normalization": "induced_2_norm_to_one",
        "local_coefficient": V3_LOCAL_COEFFICIENT,
        "relation_coefficient": V3_RELATION_COEFFICIENT,
        "innovation_standard_deviation": V3_NOISE_SCALE,
        "switching_and_propagation_burn_in": V3_BURN_IN,
        "switching_regime_block": V3_REGIME_BLOCK,
        "sham_common_driver_burn_in": V3_SHAM_BURN_IN,
        "corruption_levels_percent": list(V3_CORRUPTION_LEVELS),
    }


def selective_prior_v3_generator_fingerprint() -> str:
    """Hash only source that can change a v3 synthetic scientific artifact."""
    functions = (
        _simulate_common_driver,
        _nested_corruption_plan,
        _corrupt_connectivity,
        _save_selective_dataset,
        _contractive_relation_graph,
        _v3_regimes,
        _simulate_switching_sources_v3,
        _simulate_propagation_v3,
        _stable_regime_evaluable,
        _v3_generation_contract,
        selective_prior_v3_generation_spec,
        generate_selective_prior_synthetic_v3,
    )
    digest = hashlib.sha256()
    digest.update(V3_GENERATOR_VERSION.encode("utf-8"))
    for function in functions:
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot fingerprint v3 generator function {function.__name__}"
            ) from exc
        digest.update(function.__name__.encode("utf-8"))
        digest.update(source.encode("utf-8"))
    return digest.hexdigest()


def selective_prior_v3_generation_identity(
    *,
    seed: int,
    time_steps: int,
    variables: int,
    density: float,
    history: int,
    max_horizon: int,
) -> dict[str, object]:
    """Bind reusable artifacts to both parameters and active generator code."""
    return {
        "generation_spec": selective_prior_v3_generation_spec(
            seed=seed,
            time_steps=time_steps,
            variables=variables,
            density=density,
            history=history,
            max_horizon=max_horizon,
        ),
        "generator_fingerprint": selective_prior_v3_generator_fingerprint(),
    }


def generate_selective_prior_synthetic_v3(
    seed: int = 20260820,
    T: int = 12000,
    n: int = 50,
    density: float = 0.08,
    processed_root: Path | None = None,
    only: set[str] | None = None,
    *,
    history: int = 48,
    max_horizon: int = 12,
) -> list[str]:
    """Create v3 E1--E3 artifacts under new names and a fixed signal contract."""
    ensure_dirs()
    rng = np.random.default_rng(int(seed))
    local_coefficient = V3_LOCAL_COEFFICIENT
    relation_coefficient = V3_RELATION_COEFFICIENT
    noise_scale = V3_NOISE_SCALE
    generation_identity = selective_prior_v3_generation_identity(
        seed=seed,
        time_steps=T,
        variables=n,
        density=density,
        history=history,
        max_horizon=max_horizon,
    )
    prior = _contractive_relation_graph(n, density, rng)
    alternative = _contractive_relation_graph(n, density, rng)
    switching, regimes, switching_components = _simulate_switching_sources_v3(
        T,
        prior,
        alternative,
        rng,
        local_coefficient=local_coefficient,
        relation_coefficient=relation_coefficient,
        noise_scale=noise_scale,
    )
    switching_contract = _v3_generation_contract(
        switching,
        switching_components["prior"],
        switching_components["noise"],
        regimes=regimes,
        local_coefficient=local_coefficient,
        relation_coefficient=relation_coefficient,
    )
    evaluable = _stable_regime_evaluable(
        regimes,
        history=history,
        max_horizon=max_horizon,
    )
    names = ["spi_v3_synthetic_switching", "spi_v3_synthetic_sham"]
    if only is None or "spi_v3_synthetic_switching" in only:
        _save_selective_dataset(
            "spi_v3_synthetic_switching",
            switching,
            prior,
            {
                "experiment": "E1",
                "generation_seed": int(seed),
                "regime_labels": {
                    "0": "local",
                    "1": "prior_valid",
                    "2": "common_driver",
                    "3": "alternative_relation",
                },
                "generation_contract": switching_contract,
                **generation_identity,
            },
            prior_valid=(regimes == 1),
            processed_root=processed_root,
            artifact_schema="selective_prior_synthetic_v3",
            extra_arrays={
                "ground_truth_regime.npy": regimes.astype(np.int64),
                "ground_truth_evaluable.npy": evaluable,
            },
        )
    common = _simulate_common_driver(T, n, rng)
    common_maximum = float(np.max(np.abs(common)))
    common_contract = {
        "finite": bool(np.isfinite(common).all()),
        "maximum_absolute_value_below_25": common_maximum < 25.0,
        "maximum_absolute_value": common_maximum,
    }
    if not all(value for value in common_contract.values() if isinstance(value, bool)):
        raise RuntimeError(
            f"Protocol v3 sham generation contract failed: {common_contract}"
        )
    sham = _contractive_relation_graph(n, density, rng)
    if only is None or "spi_v3_synthetic_sham" in only:
        _save_selective_dataset(
            "spi_v3_synthetic_sham",
            common,
            sham,
            {
                "experiment": "E2",
                "generation_seed": int(seed),
                "reference_prior": "sham graph with identical training exposure",
                "generation_contract": common_contract,
                **generation_identity,
            },
            prior_valid=np.zeros(T, dtype=np.uint8),
            processed_root=processed_root,
            artifact_schema="selective_prior_synthetic_v3",
            extra_arrays={"ground_truth_evaluable.npy": np.ones(T, dtype=np.uint8)},
        )
    propagation, propagation_components = _simulate_propagation_v3(
        T,
        prior,
        rng,
        local_coefficient=local_coefficient,
        relation_coefficient=relation_coefficient,
        noise_scale=noise_scale,
    )
    propagation_contract = _v3_generation_contract(
        propagation,
        propagation_components["prior"],
        propagation_components["noise"],
        regimes=None,
        local_coefficient=local_coefficient,
        relation_coefficient=relation_coefficient,
    )
    corruption_plan = _nested_corruption_plan(prior, rng)
    for percentage in V3_CORRUPTION_LEVELS:
        name = f"spi_v3_synthetic_corruption_{percentage:03d}"
        names.append(name)
        if only is not None and name not in only:
            continue
        corrupted, corruption_metadata = _corrupt_connectivity(
            prior,
            percentage / 100.0,
            corruption_plan,
        )
        _save_selective_dataset(
            name,
            propagation,
            corrupted,
            {
                "experiment": "E3",
                "generation_seed": int(seed),
                "corruption_curve": "nested_endpoint_replacement_without_original_endpoint_reentry",
                "generation_contract": propagation_contract,
                **generation_identity,
                **corruption_metadata,
            },
            prior_valid=np.full(T, percentage == 0, dtype=np.uint8),
            processed_root=processed_root,
            artifact_schema="selective_prior_synthetic_v3",
            extra_arrays={"ground_truth_evaluable.npy": np.ones(T, dtype=np.uint8)},
        )
    return names


def validate_selective_prior_v3_metadata(
    metadata: dict,
    dataset: str,
    *,
    expected_generation_spec: dict[str, object],
    expected_generator_fingerprint: str,
) -> None:
    """Fail closed unless a processed v3 synthetic satisfies its full contract."""
    if metadata.get("schema") != "selective_prior_synthetic_v3":
        raise RuntimeError(f"Stale v3 synthetic schema for {dataset}")
    if metadata.get("dataset") != dataset:
        raise RuntimeError(f"V3 synthetic dataset identity mismatch for {dataset}")
    actual_spec = metadata.get("generation_spec")
    if actual_spec != expected_generation_spec:
        expected_keys = set(expected_generation_spec)
        actual_keys = set(actual_spec) if isinstance(actual_spec, dict) else set()
        mismatched = sorted(
            key
            for key in expected_keys & actual_keys
            if actual_spec[key] != expected_generation_spec[key]
        )
        raise RuntimeError(
            f"Stale v3 synthetic generation spec for {dataset}; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}, mismatched={mismatched}"
        )
    if metadata.get("generator_fingerprint") != expected_generator_fingerprint:
        raise RuntimeError(
            f"Stale v3 synthetic generator fingerprint for {dataset}; "
            "archive the artifact and rerun prepare-data"
        )
    contract = metadata.get("generation_contract")
    if not isinstance(contract, dict):
        raise TypeError(f"V3 synthetic generation contract is missing for {dataset}")
    required = {"finite", "maximum_absolute_value_below_25"}
    if "switching" in dataset or "corruption" in dataset:
        required.update(
            {
                "joint_update_norm_below_0_90",
                "relation_to_noise_rms_at_least_0_35",
                "relational_energy_fraction_at_least_0_05",
                "regime_scale_ratio_below_4",
            }
        )
    missing = sorted(required.difference(contract))
    failed = sorted(key for key in required if contract.get(key) is not True)
    if missing or failed:
        raise RuntimeError(
            f"V3 synthetic generation contract failed for {dataset}; "
            f"missing={missing}, failed={failed}"
        )
