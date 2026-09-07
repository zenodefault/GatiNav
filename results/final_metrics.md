# Final evaluation metrics

specs/03 section 6 metrics, no alignment: ATE = RMS horizontal position error, RPE = RMS one-step horizontal displacement error, drift = final horizontal error (lower is better).

## Per-window comparison (specs/04 section 6)

| Window | Segment | Outage s | Baseline drift (+CNN) m | Matching+NHC drift m | Reduction | Match accepted | Fallback |
|---|---|---:|---:|---:|---:|---|---|
| W01-30s | S-Vw15 | 30 | 4.38 | 4.38 | 0.0% | no | no road match accepted |
| W03-30s | S-Vta6 | 30 | 758.24 | 758.24 | 0.0% | no | no road match accepted |
| W04-30s | S-Vta10 | 30 | 1198.56 | 1198.56 | 0.0% | no | no road match accepted |
| W02-60s | S-Vw15 | 60 | 4.39 | 4.39 | 0.0% | no | no road match accepted |
| W05-60s | S-Vta22 | 60 | 241.36 | 241.36 | 0.0% | no | no road match accepted |
| W06-60s | S-Vta26 | 60 | 75.16 | 75.16 | 0.0% | no | no road match accepted |
| W07-90s | S-Vta20 | 90 | 4.05 | 4.05 | 0.0% | no | no road match accepted |
| W08-90s | S-Vw14a | 90 | 33336.12 | 33336.12 | 0.0% | no | no road match accepted |
| W09-90s | S-Vta8 | 90 | 3896.44 | 3896.44 | 0.0% | no | no road match accepted |

## ATE / RPE / drift by outage duration (specs/03 section 6)

| Duration s | Config | ATE m | RPE m | Drift m | Drift/distance |
|---|---|---:|---:|---:|---:|
| 30 | raw-INS | 435.92 | 0.13 | 599.92 | 42160.6% |
| 30 | +ZUPT | 438.88 | 0.15 | 570.13 | 1468.5% |
| 30 | +CNN | 476.07 | 0.16 | 653.73 | 619.6% |
| 30 | +NHC+matching | 476.07 | 0.16 | 653.73 | 619.6% |
| 30 | +CNN beats +ZUPT (ties not wins) | 1/3 | - | - | - |

| 60 | raw-INS | 753.51 | 0.29 | 1214.56 | nan% |
| 60 | +ZUPT | 383.05 | 0.26 | 484.59 | nan% |
| 60 | +CNN | 257.49 | 0.23 | 106.97 | nan% |
| 60 | +NHC+matching | 257.49 | 0.23 | 106.97 | nan% |
| 60 | +CNN beats +ZUPT (ties not wins) | 3/3 | - | - | - |

| 90 | raw-INS | 6258.83 | 1.84 | 14062.07 | 15465.9% |
| 90 | +ZUPT | 3868.23 | 47.76 | 5185.75 | 346.3% |
| 90 | +CNN | 5625.99 | 7.80 | 12412.20 | 679.3% |
| 90 | +NHC+matching | 5625.99 | 7.80 | 12412.20 | 679.3% |
| 90 | +CNN beats +ZUPT (ties not wins) | 1/3 | - | - | - |

