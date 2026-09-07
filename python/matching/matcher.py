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

_EARTH_RADIUS_M = 6371008.8


@dataclass(frozen=True)
class MatchedSample:
    edge_id: tuple[Any, Any, Any]
    along_edge: float
    heading: float
    residual: float  # distance from the observation to the matched road


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
            # candidate search radius must exceed the graph's node spacing;
            # simplified rural edges can span hundreds of metres even when a
            # track hugs the road, so 100 m silently matches nothing
            max_dist=500.0,
            only_edges=True,
        )

    def _load_graph(self) -> nx.MultiDiGraph:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            return ox.load_graphml(self.cache_path)
        if self.place:
            graph = ox.graph_from_place(
                self.place, network_type="drive", simplify=True,
                retain_all=False)
        else:
            raise ValueError("no graph source (place or bbox) configured")
        graph = ox.projection.project_graph(graph)
        ox.save_graphml(graph, self.cache_path)
        return graph

    @classmethod
    def from_bbox(cls, north, south, east, west, cache_path,
                  obs_noise=5.0, dist_noise=5.0, margin_deg=0.008):
        """Build a matcher over a cached local drive graph (specs/04).

        Downloads from OSM the drive network covering the lat/lon box, with
        up to two margin doublings when the box contains almost no roads
        (rural recordings), then projects and caches it. The graph CRS rides
        in ``graph.graph["crs"]`` so trajectories can be transformed into it.
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            for margin in (margin_deg, 2.0 * margin_deg, 5.0 * margin_deg):
                graph = ox.graph_from_bbox(
                    north=north + margin, south=south - margin,
                    east=east + margin, west=west - margin,
                    network_type="drive", simplify=True)
                if graph.number_of_nodes() >= 5:
                    break
            graph = ox.projection.project_graph(graph)
            ox.save_graphml(graph, cache_path)
        matcher = object.__new__(cls)
        matcher.place = None
        matcher.cache_path = cache_path
        matcher.obs_noise = obs_noise
        matcher.dist_noise = dist_noise
        matcher.graph = matcher._load_graph()
        matcher._edge_attrs = {
            (u, v, k): dict(data)
            for u, v, k, data in matcher.graph.edges(keys=True, data=True)
        }
        matcher.map = matcher._make_map()
        matcher.matcher = DistanceMatcher(
            matcher.map, obs_noise=obs_noise, dist_noise=dist_noise,
            max_dist=500.0, only_edges=True)
        return matcher

    @staticmethod
    def enu_to_graph_xy(enu_xy, lat0, lon0, graph_crs):
        """Project ENU metres (origin lat0/lon0) into the graph CRS.

        ENU trajectories and projected OSM graphs never share a coordinate
        frame; without this transform no candidate is ever within the
        matcher search radius. Uses the local-tangent inverse followed by a
        per-point projection so UTM grid convergence is handled exactly.
        """
        from pyproj import Transformer
        enu_xy = np.asarray(enu_xy, dtype=np.float64)
        lat = np.deg2rad(lat0) + enu_xy[:, 1] / _EARTH_RADIUS_M
        lon = np.deg2rad(lon0) + enu_xy[:, 0] / (
            _EARTH_RADIUS_M * np.cos(np.deg2rad(lat0)))
        transformer = Transformer.from_crs(
            "EPSG:4326", graph_crs, always_xy=True)
        xs, ys = transformer.transform(
            np.rad2deg(lon), np.rad2deg(lat))
        return np.column_stack((np.asarray(xs), np.asarray(ys)))

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
        trajectory_xy = np.asarray(trajectory_xy, dtype=np.float64)
        points = [(float(y), float(x)) for x, y in trajectory_xy]
        _, last = self.matcher.match(points, unique=False)
        if last < 0 or not self.matcher.lattice_best:
            return []
        samples = []
        for j, m in enumerate(self.matcher.lattice_best):
            if not m.is_emitting():
                continue
            obs = trajectory_xy[min(j, trajectory_xy.shape[0] - 1)]
            samples.append(self._sample_from_matching(m, obs))
        return samples

    def _sample_from_matching(self, matching: Any, obs_xy) -> MatchedSample:
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
        heading = math.atan2(y1 - y0, x1 - x0)
        # snap point on the matched edge at the along-edge fraction
        if hasattr(geometry, "interpolate"):
            snap = geometry.interpolate(fraction * geometry.length)
            residual = float(np.hypot(obs_xy[0] - snap.x, obs_xy[1] - snap.y))
        else:
            sx = x0 + fraction * (x1 - x0)
            sy = y0 + fraction * (y1 - y0)
            residual = float(np.hypot(obs_xy[0] - sx, obs_xy[1] - sy))
        return MatchedSample(key, fraction, heading, residual)

    def _edge_key(self, u: Any, v: Any) -> tuple[Any, Any, Any]:
        keys = list(self.graph[u][v])
        if len(keys) != 1:
            for key in keys:
                if (u, v, key) in self._edge_attrs:
                    return (u, v, key)
        return (u, v, keys[0])
