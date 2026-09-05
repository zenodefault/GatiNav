import numpy as np
import networkx as nx

from python.matching.matcher import OSMMatcher


def test_perfect_synthetic_road_matches_same_directed_edge(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node(0, x=0.0, y=0.0)
    graph.add_node(1, x=100.0, y=0.0)
    graph.add_edge(0, 1, key=0)

    matcher = object.__new__(OSMMatcher)
    matcher.graph = graph
    matcher._edge_attrs = {(0, 1, 0): {}}
    matcher.map = matcher._make_map()
    from leuvenmapmatching.matcher.distance import DistanceMatcher
    matcher.matcher = DistanceMatcher(
        matcher.map, obs_noise=1.0, dist_noise=1.0, max_dist=20.0
    )

    points = np.column_stack((np.linspace(1.0, 99.0, 20), np.zeros(20)))
    result = matcher.match(points)
    assert len(result) >= 19
    assert sum(sample.edge_id == (0, 1, 0) for sample in result) / len(result) >= 0.95
    assert all(abs(sample.heading) < 1e-12 for sample in result)
