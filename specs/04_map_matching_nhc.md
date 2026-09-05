# Map Matching and Non-Holonomic Constraints Specification

This specification covers OSM road context, HMM matching, and NHC coupling
for the selected synchronised IO-VNBD outage windows. `osmnx` and
`leuvenmapmatching` are not installed in the current environment; therefore
no library signature below is treated as confirmed.

## 1. Road graph acquisition

The IO-VNBD README identifies recordings on public roads in the United
Kingdom, Nigeria, and France, but does not identify a city for a selected test
area.

**Test area city:** ⚠️ VERIFY from the selected IO-VNBD recording and its GNSS
coordinates.

After selecting a city/area, acquire the drivable OSM graph with the
appropriate `osmnx` graph-acquisition call. Candidate API name and signature:
`osmnx.graph_from_place(...)` ⚠️ VERIFY against the installed `osmnx` source
before use. If the test area is defined by a bounding box rather than a place
name, the corresponding `osmnx` call is ⚠️ VERIFY.

Project the graph from geographic coordinates to a local metric CRS. Derive
the UTM zone from the representative test-area longitude by dividing the
longitude-plus-180 by 6, taking the integer zone index, and adding 1; select
the northern or southern UTM hemisphere from latitude. The exact CRS
authority/string and whether the graph is projected with
`osmnx.projection.project_graph(...)` ⚠️ VERIFY against the installed source.

Graph simplification settings:

- Preserve the graph's directed travel topology.
- Retain edge geometry and metadata needed for headings and along-edge
  positions.
- Do not simplify away geometry needed to distinguish parallel or spur roads.
- Exact simplification flag, call order, and edge attribute preservation are
  ⚠️ VERIFY.

TODO: Install or otherwise inspect the pinned/target `osmnx` source, verify
the graph acquisition and projection signatures, then run them on one
selected IO-VNBD coordinate envelope and record the resulting CRS, node/edge
counts, geometry retention, and simplification behavior.

## 2. HMM map matching with leuvenmapmatching

During a GNSS outage, use the inertial-only trajectory produced by the EKF as
the matcher input. Do not feed held-out GNSS or ground truth into matching.

Construct the matcher using the library's graph/network and map-matching
classes. Candidate names such as `leuvenmapmatching.map.inmem.MapInMem` and
`leuvenmapmatching.matcher.distance.DistanceMatcher` are ⚠️ VERIFY; their
constructors, required graph format, and keyword names must be read from the
installed library source before coding.

Noise parameters shall include position/measurement noise σ values and any
transition/emission parameters required by the verified matcher API. Starting
values are ⚠️ VERIFY and must be tuned on training outage windows only.

Expected logical output for each matched sample:

1. Matched edge identifier `(u, v, key)` or the verified library equivalent.
2. Along-edge position.
3. Edge heading in the local metric frame.

The exact matcher output object, callback/iteration API, edge identifier
format, along-edge units, and heading convention are ⚠️ VERIFY.

Candidate calls such as `matcher.match(...)`,
`matcher.match_map(...)`, or `matcher.path` are **not approved signatures**:
each is ⚠️ VERIFY until confirmed in the installed
`leuvenmapmatching` source.

TODO: Inspect the installed package source for matcher construction, input
sequence format, matching invocation, and result accessors; run one short
inertial trajectory through the verified API and document the exact output
mapping.

Map matching is an offline/Python responsibility. Android consumes only a
compact accepted-match record containing timestamp, edge identifier, road
heading in ENU radians, and match-quality status. Android must not load or
search the OSM graph.

## 3. NHC as an EKF pseudo-measurement

Apply NHC only when a road match passes quality checks. In plain English, the
vehicle velocity perpendicular to the matched road heading should be
approximately zero. Inject this constraint with a moderate measurement
covariance so the filter is pulled toward the road and is never snapped
directly onto the road geometry.

For an ENU road heading `ψ` in radians, define the horizontal along-road and
left-lateral unit vectors:

```latex
\[
\mathbf{t}(\psi)=
\begin{bmatrix}\cos\psi\\\sin\psi\\0\end{bmatrix},
\qquad
\mathbf{n}(\psi)=
\begin{bmatrix}-\sin\psi\\\cos\psi\\0\end{bmatrix}.
\]
```

The scalar pseudo-measurement is:

```latex
\[
z_{nhc}=0,\qquad
h_{nhc}(\mathbf{x})=\mathbf{n}(\psi)^T\mathbf{v}_{enu},\qquad
r_{nhc}=z_{nhc}-h_{nhc}(\mathbf{x}).
\]
```

For the project state ordering
`[δθ, δv, δp, δb_g, δb_a]`, use:

```latex
\[
\mathbf{H}_{nhc}=
\begin{bmatrix}
\mathbf{0}_{1\times3} & \mathbf{n}(\psi)^T &
\mathbf{0}_{1\times3} & \mathbf{0}_{1\times3} & \mathbf{0}_{1\times3}
\end{bmatrix},
\qquad
\mathbf{R}_{nhc}=[\sigma_{nhc}^2].
\]
```

Apply this through the generic EKF update. The Kalman update pulls the lateral
velocity toward zero; it must not directly overwrite velocity, position, or
the along-road component. The match must pass quality checks before NHC is
applied. `σ_nhc`, match gating, heading-residual rejection, and covariance
inflation remain data-calibration items and must not be inferred here.

TODO: Use the human-filled EKF conventions from `specs/02_ekf.md`, derive the
lateral-velocity residual for one matched edge, and add a known-answer test
for a velocity parallel to and perpendicular to that edge.

## 4. GNSS↔INS state machine

States:

- **FUSED:** GNSS updates and inertial propagation are both accepted.
- **INERTIAL:** GNSS is considered unavailable/unreliable; inertial
  propagation and eligible matching/NHC continue.
- **REACQUIRING:** A fix has returned and is being checked before normal fusion.

Transitions:

- `FUSED → INERTIAL` when GNSS is lost for more than 2 s or reported accuracy
  jumps.
- `INERTIAL → REACQUIRING` when a GNSS fix returns with accuracy `<10 m`.
- `REACQUIRING → FUSED` after a covariance-consistency check passes.
- A failed reacquisition consistency check returns to `INERTIAL`.
- Inflate covariance on every state transition.

The GNSS accuracy field, definition of “lost,” accuracy-jump threshold,
covariance-consistency test, and inflation factors are ⚠️ VERIFY.

```text
                 GNSS lost > 2 s
              or accuracy jumps
        +------------------------------+
        |                              v
   +---------+                   +-----------+
   |  FUSED  |                   | INERTIAL  |
   +---------+                   +-----------+
        ^                              |
        | covariance check              | fix returns,
        | passes                        | accuracy < 10 m
        |                              v
        |                       +---------------+
        +-----------------------| REACQUIRING   |
          covariance check      +---------------+
          passes                       |
                                       | check fails
                                       +----------> INERTIAL
```

TODO: Replay one synchronised segment with GNSS updates removed and restored,
measure transition timing and covariance before/after each transition, then
set and document all ⚠️ VERIFY thresholds and inflation factors.

## 5. Failure modes and fallbacks

Known operational failure modes include:

- Wrong-edge lock-in.
- Spur-road matches.
- Duplicate parallel edges.
- Heading disagreement between inertial trajectory and matched edge.

Fallback rule: reject a match when the heading residual exceeds a configured
threshold, mark the match unusable, and continue inertial propagation without
the NHC constraint. The heading-residual threshold is ⚠️ VERIFY.

Additional match score thresholds, temporal hysteresis, rematch frequency,
and whether a rejected match may be retried with another candidate are
⚠️ VERIFY.

TODO: Build a failure corpus from held-out outage windows, label wrong-edge,
spur, duplicate-edge, and heading-residual cases, then measure rejection and
recovery rates while tuning the threshold on training windows only.

## 6. Acceptance

On the same outage windows used for the Phase-3 baseline, adding map matching
and NHC must reduce horizontal drift by at least 50%:

```text
drift_with_matching_and_nhc <= 0.50 * drift_phase_3
```

The exact Phase-3 artifact, horizontal-drift definition, alignment method,
window inclusion rules, and treatment of rejected matches are ⚠️ VERIFY.

Report one row per outage window with at least:

| Window ID | Segment/driver | Outage length | Baseline horizontal drift | Matching + NHC drift | Reduction | Match accepted | Failure/fallback |
|---|---|---:|---:|---:|---:|---|---|
| ⚠️ VERIFY | ⚠️ VERIFY | 30/60/90 s | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |

Summarize results separately for 30 s, 60 s, and 90 s outages and report the
fraction meeting the ≥50% reduction criterion.

TODO: Run the verified graph/matcher and NHC implementation on exactly the
Phase-3 outage-window identifiers, compute the per-window table, and confirm
the ≥50% horizontal-drift reduction without using ground truth as a filter
input.
