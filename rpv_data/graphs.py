from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GraphArrays:
    edge_index: np.ndarray
    edge_weight: np.ndarray
    num_nodes: int
    directed: bool = True

    def validate(self) -> None:
        ei = np.asarray(self.edge_index)
        ew = np.asarray(self.edge_weight)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError(f"edge_index must be [2,E], got {ei.shape}")
        if ew.shape != (ei.shape[1],):
            raise ValueError("edge_weight length mismatch")
        if ei.size and (ei.min() < 0 or ei.max() >= self.num_nodes):
            raise ValueError("Graph node index out of range")
        pairs = list(zip(ei[0].tolist(), ei[1].tolist()))
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate graph edges")
        if not np.isfinite(ew).all():
            raise ValueError("Non-finite graph weights")


def save_graph(path: Path, graph: GraphArrays) -> None:
    graph.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        edge_index=np.asarray(graph.edge_index, dtype=np.int64),
        edge_weight=np.asarray(graph.edge_weight, dtype=np.float32),
        num_nodes=np.int64(graph.num_nodes),
        directed=np.uint8(graph.directed),
    )


def load_graph(path: Path) -> GraphArrays:
    z = np.load(path, allow_pickle=False)
    g = GraphArrays(
        edge_index=np.asarray(z["edge_index"], dtype=np.int64),
        edge_weight=np.asarray(z["edge_weight"], dtype=np.float32),
        num_nodes=int(z["num_nodes"]),
        directed=bool(int(z["directed"])),
    )
    g.validate()
    return g


def identity_graph(n: int) -> GraphArrays:
    idx = np.arange(n, dtype=np.int64)
    return GraphArrays(np.vstack([idx, idx]), np.ones(n, dtype=np.float32), n, True)


def all_to_all_graph(n: int, include_self: bool = True) -> GraphArrays:
    src, dst = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    src, dst = src.ravel(), dst.ravel()
    if not include_self:
        keep = src != dst
        src, dst = src[keep], dst[keep]
    return GraphArrays(np.vstack([src, dst]), np.ones(len(src), dtype=np.float32), n, True)


def from_edges(edges: Iterable[tuple[int, int, float]], n: int, add_self: bool = True, directed: bool = True) -> GraphArrays:
    mapping: dict[tuple[int, int], float] = {}
    for u, v, w in edges:
        u, v = int(u), int(v)
        if u < 0 or v < 0 or u >= n or v >= n:
            raise ValueError(f"Edge {(u, v)} outside graph with {n} nodes")
        mapping[(u, v)] = float(w)
        if not directed:
            mapping[(v, u)] = float(w)
    if add_self:
        for i in range(n):
            mapping.setdefault((i, i), 1.0)
    pairs = sorted(mapping)
    edge_index = np.array(pairs, dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)
    weights = np.array([mapping[p] for p in pairs], dtype=np.float32)
    return GraphArrays(edge_index, weights, n, True)


def correlation_graph(values_train: np.ndarray, top_k: int = 5, min_abs_corr: float = 0.0) -> GraphArrays:
    x = np.asarray(values_train, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected [time, variables]")
    n = x.shape[1]
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    neighbors: set[tuple[int, int]] = set()
    for i in range(n):
        order = np.argsort(-np.abs(corr[i]))
        selected = [j for j in order if j != i and abs(corr[i, j]) >= min_abs_corr][: min(top_k, n - 1)]
        for j in selected:
            neighbors.add((i, int(j)))
            neighbors.add((int(j), i))
    edges = [(u, v, float(abs(corr[u, v]))) for u, v in sorted(neighbors)]
    return from_edges(edges, n=n, add_self=True, directed=True)


def degree_sequences(graph: GraphArrays, include_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    src, dst = graph.edge_index
    keep = np.ones(len(src), dtype=bool) if include_self else (src != dst)
    return (
        np.bincount(src[keep], minlength=graph.num_nodes),
        np.bincount(dst[keep], minlength=graph.num_nodes),
    )


def endpoint_change_fraction(base: GraphArrays, other: GraphArrays) -> float:
    base_pairs = {(int(u), int(v)) for u, v in base.edge_index.T if u != v}
    new_pairs = {(int(u), int(v)) for u, v in other.edge_index.T if u != v}
    if not base_pairs:
        return 0.0
    return 1.0 - len(base_pairs & new_pairs) / len(base_pairs)


def _weight_bins(weights: np.ndarray, n_bins: int) -> np.ndarray:
    if len(weights) == 0 or n_bins <= 1:
        return np.zeros(len(weights), dtype=np.int64)
    unique = np.unique(weights)
    if len(unique) < 2:
        return np.zeros(len(weights), dtype=np.int64)
    quantiles = np.unique(np.quantile(weights, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) <= 2:
        return np.zeros(len(weights), dtype=np.int64)
    return np.digitize(weights, quantiles[1:-1], right=True).astype(np.int64)



def _is_symmetric(graph: GraphArrays) -> bool:
    weights = {(int(u), int(v)): float(w) for (u, v), w in zip(graph.edge_index.T, graph.edge_weight)}
    for (u, v), w in weights.items():
        if u == v:
            continue
        if (v, u) not in weights or not np.isclose(w, weights[(v, u)], rtol=1e-5, atol=1e-7):
            return False
    return True


def _rewire_symmetric(
    graph: GraphArrays,
    seed: int,
    matched_weight_bins: bool,
    min_endpoint_change: float,
    max_attempt_factor: int,
) -> GraphArrays:
    rng = np.random.default_rng(seed)
    weight_map = {(int(u), int(v)): float(w) for (u, v), w in zip(graph.edge_index.T, graph.edge_weight)}
    undirected = []
    self_edges = []
    for (u, v), w in weight_map.items():
        if u == v:
            self_edges.append((u, v, w))
        elif u < v:
            undirected.append((u, v, w))
    if len(undirected) < 4:
        raise ValueError("Symmetric graph has too few non-self edges to rewire")
    edge_set = {(u, v) for u, v, _ in undirected}
    bins = _weight_bins(
        np.array([w for _, _, w in undirected]),
        (3 if len(undirected) < 100 else 5) if matched_weight_bins else 1,
    )
    by_bin: dict[int, list[int]] = {}
    for idx, b in enumerate(bins):
        by_bin.setdefault(int(b), []).append(idx)
    eligible = [b for b, ids in by_bin.items() if len(ids) >= 2]
    if not eligible:
        raise RuntimeError("No weight bin has enough edges for matched symmetric rewiring")
    max_attempts = max_attempt_factor * len(undirected)
    # Constrained shifts can stop once the requested endpoint fraction is met;
    # keep the high-acceptance target only for the maximum-feasible path used
    # by the nominal 100% policy.
    target_accepts = max(10 * len(undirected), 1)
    accepted = attempts = 0

    def assemble() -> GraphArrays:
        directed_edges = list(self_edges)
        for u, v, w in undirected:
            directed_edges.append((u, v, w))
            directed_edges.append((v, u, w))
        directed_edges.sort(key=lambda x: (x[0], x[1]))
        return GraphArrays(
            np.array([(u, v) for u, v, _ in directed_edges], dtype=np.int64).T,
            np.array([w for _, _, w in directed_edges], dtype=np.float32),
            graph.num_nodes,
            True,
        )

    check_every = max(len(undirected) // 10, 50)
    if min_endpoint_change > 0.0:
        target_accepts = check_every
    while attempts < max_attempts:
        attempts += 1
        bin_id = int(rng.choice(eligible))
        i, j = rng.choice(by_bin[bin_id], size=2, replace=False)
        a, b, wa = undirected[int(i)]
        c, d, wc = undirected[int(j)]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            q1, q2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        else:
            q1, q2 = tuple(sorted((a, c))), tuple(sorted((b, d)))
        if q1[0] == q1[1] or q2[0] == q2[1] or q1 == q2:
            continue
        old1, old2 = (a, b), (c, d)
        if q1 in edge_set or q2 in edge_set:
            continue
        edge_set.remove(old1)
        edge_set.remove(old2)
        edge_set.add(q1)
        edge_set.add(q2)
        undirected[int(i)] = (q1[0], q1[1], wa)
        undirected[int(j)] = (q2[0], q2[1], wc)
        accepted += 1
        if (
            accepted >= target_accepts
            and accepted % check_every == 0
            and endpoint_change_fraction(graph, assemble()) >= min_endpoint_change
        ):
            break
    out = assemble()
    frac = endpoint_change_fraction(graph, out)
    if frac < min_endpoint_change:
        raise RuntimeError(
            f"Could only change {frac:.3f} of endpoints; requested {min_endpoint_change:.3f}. "
            "Do not silently weaken the null constraint."
        )
    if not np.array_equal(degree_sequences(graph)[0], degree_sequences(out)[0]):
        raise AssertionError("Symmetric degree sequence changed during rewiring")
    return out

def rewire_directed(
    graph: GraphArrays,
    seed: int,
    matched_weight_bins: bool,
    min_endpoint_change: float,
    max_attempt_factor: int = 500,
) -> GraphArrays:
    """Degree-preserving edge swaps with exact edge-weight multiset preservation.

    Symmetric masks are rewired as undirected graphs and expanded back to two
    directions. Other masks use directed double-edge swaps. For matched nulls,
    swaps are restricted to the same weight quantile bin. Carried edge weights
    are null-edge attributes, not claims about the physical distance between
    newly connected endpoints.
    """
    graph.validate()
    if _is_symmetric(graph):
        return _rewire_symmetric(
            graph, seed=seed, matched_weight_bins=matched_weight_bins,
            min_endpoint_change=min_endpoint_change, max_attempt_factor=max_attempt_factor
        )
    rng = np.random.default_rng(seed)
    src, dst = graph.edge_index
    nonself_idx = np.flatnonzero(src != dst)
    self_idx = np.flatnonzero(src == dst)
    if len(nonself_idx) < 4:
        raise ValueError("Graph has too few non-self edges to rewire")
    edges = [(int(src[i]), int(dst[i]), float(graph.edge_weight[i])) for i in nonself_idx]
    edge_set = {(u, v) for u, v, _ in edges}
    bins = _weight_bins(
        np.array([w for _, _, w in edges]),
        (3 if len(edges) < 100 else 5) if matched_weight_bins else 1,
    )
    by_bin: dict[int, list[int]] = {}
    for idx, b in enumerate(bins):
        by_bin.setdefault(int(b), []).append(idx)
    eligible_bins = [b for b, ids in by_bin.items() if len(ids) >= 2]
    attempts = 0
    accepted = 0
    max_attempts = max_attempt_factor * len(edges)
    # Match the symmetric implementation: constrained shifts stop at the
    # declared endpoint fraction, while unconstrained maximum rewiring keeps
    # the original high-acceptance target.
    target_accepts = max(10 * len(edges), 1)
    check_every = max(len(edges) // 10, 100)
    if min_endpoint_change > 0.0:
        target_accepts = check_every
    while attempts < max_attempts:
        attempts += 1
        b = int(rng.choice(eligible_bins))
        i, j = rng.choice(by_bin[b], size=2, replace=False)
        a, b1, wa = edges[int(i)]
        c, d, wc = edges[int(j)]
        if len({a, b1, c, d}) < 4:
            continue
        p1, p2 = (a, d), (c, b1)
        if p1[0] == p1[1] or p2[0] == p2[1] or p1 == p2:
            continue
        old1, old2 = (a, b1), (c, d)
        if p1 in edge_set or p2 in edge_set:
            continue
        edge_set.remove(old1)
        edge_set.remove(old2)
        edge_set.add(p1)
        edge_set.add(p2)
        edges[int(i)] = (p1[0], p1[1], wa)
        edges[int(j)] = (p2[0], p2[1], wc)
        accepted += 1
        if accepted >= target_accepts and accepted % check_every == 0:
            probe = _assemble(graph, edges, self_idx)
            if endpoint_change_fraction(graph, probe) >= min_endpoint_change:
                break
    out = _assemble(graph, edges, self_idx)
    frac = endpoint_change_fraction(graph, out)
    if frac < min_endpoint_change:
        raise RuntimeError(
            f"Could only change {frac:.3f} of endpoints; requested {min_endpoint_change:.3f}. "
            "Do not silently weaken the null constraint."
        )
    if not np.array_equal(degree_sequences(graph)[0], degree_sequences(out)[0]) or not np.array_equal(degree_sequences(graph)[1], degree_sequences(out)[1]):
        raise AssertionError("Degree sequence changed during rewiring")
    return out


def _assemble(base: GraphArrays, nonself_edges: list[tuple[int, int, float]], self_idx: np.ndarray) -> GraphArrays:
    edges = list(nonself_edges)
    for i in self_idx:
        edges.append((int(base.edge_index[0, i]), int(base.edge_index[1, i]), float(base.edge_weight[i])))
    edges.sort(key=lambda x: (x[0], x[1]))
    return GraphArrays(
        np.array([(u, v) for u, v, _ in edges], dtype=np.int64).T,
        np.array([w for _, _, w in edges], dtype=np.float32),
        base.num_nodes,
        base.directed,
    )
