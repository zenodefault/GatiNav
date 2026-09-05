"""Cached OSM graph acquisition and Leuven HMM map matching."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import networkx as nx
import numpy as np
import osmnx as ox
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher


@dataclass(frozen=True)
class MatchedSample:
    edge_id: tuple[Any, Any, Any]
    along_edge: float
    heading: float


class OSMMatcher:
    """Projected, directed OSM matcher using the verified Leuven API."""

    def __init__(
        self,
        place: str = "Cambridge, United Kingdom",
        cache_path: str | Path = "python/matching/osm_graph.graphml",
        obs_noise: float = 5.0,
        dist_noise: float = 5.0,
    ) -> None:
        self.place = place
        self.cache_path = Path(cache_path)
        self.obs_noise = obs_noise
        self.dist_noise = dist_noise
        self.graph = self._load_graph()
        self._edge_attrs = {
            (u, v, k): dict(data)
            for u, v, k, data in self.graph.edges(keys=True, data=True)
        }
        self.map = self._make_map()
        self.matcher = DistanceMatcher(
            self.map,
            obs_noise=obs_noise,
            dist_noise=dist_noise,
            max_dist=100.0,
            only_edges=True,
        )

    def _load_graph(self) -> nx.MultiDiGraph:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            return ox.load_graphml(self.cache_path)
        graph = ox.graph_from_place(
            self.place, network_type="drive", simplify=True, retain_all=False
        )
        graph = ox.projection.project_graph(graph)
        ox.save_graphml(graph, self.cache_path)
        return graph

    def _make_map(self) -> InMemMap:
        graph: dict[Any, tuple[tuple[float, float], list[Any]]] = {}
        for node, data in self.graph.nodes(data=True):
            graph[node] = ((float(data["y"]), float(data["x"])), [])
        for u, v in self.graph.edges():
            graph[u][1].append(v)
        return InMemMap(
            "osm",
            use_latlon=False,
            graph=graph,
            crs_xy=self.graph.graph.get("crs", "EPSG:3857"),
        )

    def match(self, trajectory_xy: np.ndarray) -> list[MatchedSample]:
        """Match projected ``(x, y)`` observations to directed OSM edges."""
        points = [(float(y), float(x)) for x, y in np.asarray(trajectory_xy)]
        _, last = self.matcher.match(points, unique=False)
        if last < 0 or not self.matcher.lattice_best:
            return []
        return [
            self._sample_from_matching(m)
            for m in self.matcher.lattice_best
            if m.is_emitting()
        ]

    def _sample_from_matching(self, matching: Any) -> MatchedSample:
        segment = matching.edge_m
        u, v = segment.l1, segment.l2
        key = self._edge_key(u, v)
        fraction = float(np.clip(segment.ti, 0.0, 1.0))
        attrs = self._edge_attrs[key]
        geometry = attrs.get("geometry")
        if geometry is not None and hasattr(geometry, "coords"):
            coords = list(geometry.coords)
            x0, y0 = coords[0]
            x1, y1 = coords[-1]
        else:
            x0, y0 = float(self.graph.nodes[u]["x"]), float(self.graph.nodes[u]["y"])
            x1, y1 = float(self.graph.nodes[v]["x"]), float(self.graph.nodes[v]["y"])
        return MatchedSample(key, fraction, math.atan2(y1 - y0, x1 - x0))

    def _edge_key(self, u: Any, v: Any) -> tuple[Any, Any, Any]:
        keys = list(self.graph[u][v])
        if len(keys) != 1:
            for key in keys:
                if (u, v, key) in self._edge_attrs:
                    return (u, v, key)
        return (u, v, keys[0])
