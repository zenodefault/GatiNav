# Final evaluation metrics

specs/03 section 6 metrics, no alignment: ATE = RMS horizontal position error, RPE = RMS one-step horizontal displacement error, drift = final horizontal error (lower is better).

## Per-window comparison (specs/04 section 6)

| Window | Segment | Outage s | Baseline drift (+CNN) m | Matching+NHC drift m | Reduction | Match accepted | Fallback |
|---|---|---:|---:|---:|---:|---|---|
| W01-30s | S-Vw15 | 30 | 4.53 | 4.53 | 0.0% | no | no road match accepted |
| W03-30s | S-Vta6 | 30 | 967.99 | 967.99 | 0.0% | no | no road match accepted |
| W04-30s | S-Vta10 | 30 | 1045.13 | 1045.13 | 0.0% | no | no road match accepted |
| W02-60s | S-Vw15 | 60 | 4.58 | 4.58 | 0.0% | no | no road match accepted |
| W05-60s | S-Vta22 | 60 | 1113.55 | 1113.55 | 0.0% | no | no road match accepted |
| W06-60s | S-Vta26 | 60 | 43.08 | 43.08 | 0.0% | no | no road match accepted |
| W07-90s | S-Vta20 | 90 | 6.01 | 6.01 | 0.0% | no | no road match accepted |
| W08-90s | S-Vw14a | 90 | 5443.99 | 5443.99 | 0.0% | no | no road match accepted |
| W09-90s | S-Vta8 | 90 | 695.60 | 695.60 | 0.0% | no | no road match accepted |

## ATE / RPE / drift by outage duration (specs/03 section 6)

| Duration s | Config | ATE m | RPE m | Drift m | Drift/distance |
|---|---|---:|---:|---:|---:|
| 30 | raw-INS | 435.92 | 0.13 | 599.92 | 42160.6% |
| 30 | +ZUPT | 438.88 | 0.15 | 570.13 | 1468.5% |
| 30 | +CNN | 470.91 | 0.28 | 672.55 | 643.9% |
| 30 | +NHC+matching | 470.91 | 0.28 | 672.55 | 643.9% |
| 30 | +CNN beats +ZUPT (ties not wins) | 2/3 | - | - | - |

| 60 | raw-INS | 753.51 | 0.29 | 1214.56 | nan% |
| 60 | +ZUPT | 383.05 | 0.26 | 484.59 | nan% |
| 60 | +CNN | 317.24 | 0.24 | 387.07 | nan% |
| 60 | +NHC+matching | 317.24 | 0.24 | 387.07 | nan% |
| 60 | +CNN beats +ZUPT (ties not wins) | 3/3 | - | - | - |

| 90 | raw-INS | 6258.83 | 1.84 | 14062.07 | 15465.9% |
| 90 | +ZUPT | 3868.23 | 47.76 | 5185.75 | 346.3% |
| 90 | +CNN | 3338.73 | 69.66 | 2048.53 | 159.8% |
| 90 | +NHC+matching | 3338.73 | 69.66 | 2048.53 | 159.8% |
| 90 | +CNN beats +ZUPT (ties not wins) | 3/3 | - | - | - |

