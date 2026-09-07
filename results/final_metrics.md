# Final evaluation metrics

specs/03 section 6 metrics, no alignment: ATE = RMS horizontal position error, RPE = RMS one-step horizontal displacement error, drift = final horizontal error (lower is better).

## Per-window comparison (specs/04 section 6)

| Window | Segment | Outage s | Baseline drift (+CNN) m | Matching+NHC drift m | Reduction | Match accepted | Fallback |
|---|---|---:|---:|---:|---:|---|---|
| W01-30s | S-Vw15 | 30 | 10.30 | 9.09 | 11.8% | no | no road match accepted |
| W03-30s | S-Vta6 | 30 | 1019.94 | 1566.10 | -53.5% | no | no road match accepted |
| W04-30s | S-Vta10 | 30 | 1149.91 | 1151.93 | -0.2% | no | no road match accepted |
| W02-60s | S-Vw15 | 60 | 10.00 | 7.20 | 28.1% | no | no road match accepted |
| W05-60s | S-Vta22 | 60 | 1051.52 | 1018.96 | 3.1% | no | no road match accepted |
| W06-60s | S-Vta26 | 60 | 156.92 | 132.68 | 15.4% | no | no road match accepted |
| W07-90s | S-Vta20 | 90 | 9.17 | 9.16 | 0.1% | no | no road match accepted |
| W08-90s | S-Vw14a | 90 | 11339.37 | 11383.20 | -0.4% | no | no road match accepted |
| W09-90s | S-Vta8 | 90 | 994.58 | 977.49 | 1.7% | no | no road match accepted |

## ATE / RPE / drift by outage duration (specs/03 section 6)

| Duration s | Config | ATE m | RPE m | Drift m | Drift/distance |
|---|---|---:|---:|---:|---:|
| 30 | raw-INS | 435.92 | 0.13 | 599.92 | 42160.6% |
| 30 | +ZUPT | 438.88 | 0.15 | 570.13 | 1468.5% |
| 30 | +CNN | 489.06 | 0.23 | 726.72 | 1338.6% |
| 30 | +NHC+matching | 600.77 | 0.28 | 909.04 | 1228.1% |
| 30 | +CNN beats +ZUPT (ties not wins) | 2/3 | - | - | - |

| 60 | raw-INS | 753.51 | 0.29 | 1214.56 | nan% |
| 60 | +ZUPT | 383.05 | 0.26 | 484.59 | nan% |
| 60 | +CNN | 332.15 | 0.24 | 406.15 | nan% |
| 60 | +NHC+matching | 329.59 | 0.25 | 386.28 | nan% |
| 60 | +CNN beats +ZUPT (ties not wins) | 3/3 | - | - | - |

| 90 | raw-INS | 6258.83 | 1.84 | 14062.07 | 15465.9% |
| 90 | +ZUPT | 3868.23 | 47.76 | 5185.75 | 346.3% |
| 90 | +CNN | 3662.93 | 55.91 | 4114.37 | 284.7% |
| 90 | +NHC+matching | 3663.89 | 55.82 | 4123.28 | 284.6% |
| 90 | +CNN beats +ZUPT (ties not wins) | 2/3 | - | - | - |

