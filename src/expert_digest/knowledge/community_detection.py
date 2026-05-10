"""Deterministic community detection for topic graphs."""

from __future__ import annotations


def detect_communities(
    *,
    graph: dict[str, set[str]],
    max_iter: int = 30,
) -> list[list[str]]:
    """Run deterministic label propagation and return grouped communities."""
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if not graph:
        return []

    labels = {node: node for node in graph}
    nodes = sorted(graph.keys())

    for _ in range(max_iter):
        changed = False
        for node in nodes:
            neighbors = sorted(graph.get(node, set()))
            if not neighbors:
                continue
            counts: dict[str, int] = {}
            for neighbor in neighbors:
                label = labels.get(neighbor, neighbor)
                counts[label] = counts.get(label, 0) + 1
            best_label = sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break

    grouped: dict[str, list[str]] = {}
    for node in nodes:
        grouped.setdefault(labels[node], []).append(node)

    communities = [sorted(items) for items in grouped.values()]
    communities.sort(key=lambda items: (-len(items), items[0]))
    return communities
