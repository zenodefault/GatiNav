# Final evaluation metrics

specs/03 section 6 metrics, no alignment: ATE = RMS horizontal position error, RPE = RMS one-step horizontal displacement error, drift = final horizontal error (lower is better).

## Per-window comparison (specs/04 section 6)

| Window | Segment | Outage s | Baseline drift (+CNN) m | Matching+NHC drift m | Reduction | Match accepted | Fallback |
|---|---|---:|---:|---:|---:|---|---|
| W01-30s | S-Vta11 | 30 | 2129.52 | 2129.52 | 0.0% | no | no road match accepted |
| W02-30s | S-Vtb4 | 30 | 1956.99 | 1956.99 | 0.0% | no | no road match accepted |
| W03-30s | S-Vta12 | 30 | 2881.10 | 2881.10 | 0.0% | no | no road match accepted |
| W06-60s | S-Vw12 | 60 | 39158.08 | 39158.08 | 0.0% | no | no road match accepted |
| W04-60s | S-Vta15 | 60 | 553.08 | 553.08 | 0.0% | no | no road match accepted |
| W05-60s | S-Vta7 | 60 | 9279.56 | 9279.56 | 0.0% | no | no road match accepted |
| W07-90s | S-Vta23 | 90 | 22979.07 | 22979.07 | 0.0% | no | no road match accepted |
| W08-90s | S-Vw16b | 90 | 17522.01 | 17522.01 | 0.0% | no | no road match accepted |
| W09-90s | S-Vta24 | 90 | 26142.98 | 26142.98 | 0.0% | no | no road match accepted |

## ATE / RPE / drift by outage duration (specs/03 section 6)

| Duration s | Config | ATE m | RPE m | Drift m | Drift/distance |
|---|---|---:|---:|---:|---:|
| 30 | raw-INS | 1010.57 | 0.99 | 2322.54 | nan% |
| 30 | +ZUPT | 1010.57 | 0.99 | 2322.54 | nan% |
| 30 | +CNN | 1010.57 | 0.99 | 2322.54 | nan% |
| 30 | +NHC+matching | 1010.57 | 0.99 | 2322.54 | nan% |
| 30 | +CNN beats +ZUPT (ties not wins) | 0/3 | - | - | - |

| 60 | raw-INS | 7083.73 | 3.46 | 16330.24 | nan% |
| 60 | +ZUPT | 7083.73 | 3.46 | 16330.24 | nan% |
| 60 | +CNN | 7083.73 | 3.46 | 16330.24 | nan% |
| 60 | +NHC+matching | 7083.73 | 3.46 | 16330.24 | nan% |
| 60 | +CNN beats +ZUPT (ties not wins) | 0/3 | - | - | - |

| 90 | raw-INS | 9866.16 | 2.98 | 22214.69 | nan% |
| 90 | +ZUPT | 9866.16 | 2.98 | 22214.69 | nan% |
| 90 | +CNN | 9866.16 | 2.98 | 22214.69 | nan% |
| 90 | +NHC+matching | 9866.16 | 2.98 | 22214.69 | nan% |
| 90 | +CNN beats +ZUPT (ties not wins) | 0/3 | - | - | - |

