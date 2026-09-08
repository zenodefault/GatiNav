# Final evaluation metrics

specs/03 section 6 metrics, no alignment: ATE = RMS horizontal position error, RPE = RMS one-step horizontal displacement error, drift = final horizontal error (lower is better).

## Per-window comparison (specs/04 section 6)

| Window | Segment | Outage s | Distance m | Baseline drift (+CNN) m | Matching+NHC drift m | Reduction | Match accepted | Fallback |
|---|---|---:|---:|---:|---:|---:|---|---|
| W01-30s | S-Vw15 | 30 | 0.3 | 9.59 | 9.59 | 0.0% | no | no road match accepted |
| W03-30s | S-Vta6 | 30 | 527.4 | 466.30 | 405.93 | 12.9% | yes | - |
| W04-30s | S-Vta10 | 30 | 798.9 | 1494.38 | 1421.58 | 4.9% | yes | - |
| W02-60s | S-Vw15 | 60 | 0.3 | 9.70 | 9.70 | 0.0% | no | no road match accepted |
| W05-60s | S-Vta22 | 60 | 0.0 | 2903.18 | 1355.18 | 53.3% | yes | - |
| W06-60s | S-Vta26 | 60 | 24.1 | 94.13 | 94.13 | 0.0% | no | no road match accepted |
| W07-90s | S-Vta20 | 90 | 3.8 | 3.27 | 3.27 | 0.0% | no | match unavailable |
| W08-90s | S-Vw14a | 90 | 2315.5 | 12066.43 | 4208.01 | 65.1% | yes | - |
| W09-90s | S-Vta8 | 90 | 790.7 | 5762.35 | 5762.35 | 0.0% | no | - |

## ATE / RPE / drift by outage duration (specs/03 section 6)

| Duration s | Config | ATE m | RPE m | Drift m | Distance m | Drift/distance |
|---|---|---:|---:|---:|---:|---:|
| 30 | raw-INS | 602.55 | 0.36 | 978.25 | 442.2 | 221.2% |
| 30 | +ZUPT | 385.17 | 0.16 | 479.76 | 442.2 | 108.5% |
| 30 | +CNN | 413.13 | 0.26 | 656.76 | 442.2 | 148.5% |
| 30 | +NHC+matching | 373.53 | 0.29 | 612.36 | 442.2 | 138.5% |
| 30 | +CNN beats +ZUPT (ties not wins) | 2/3 | - | - | - |

| 60 | raw-INS | 1042.11 | 0.38 | 2063.59 | 8.1 | 25442.9% |
| 60 | +ZUPT | 739.46 | 0.56 | 1666.75 | 8.1 | 20550.1% |
| 60 | +CNN | 471.32 | 0.55 | 1002.34 | 8.1 | 12358.3% |
| 60 | +NHC+matching | 266.23 | 1.17 | 486.34 | 8.1 | 5996.3% |
| 60 | +CNN beats +ZUPT (ties not wins) | 3/3 | - | - | - |

| 90 | raw-INS | 5243.13 | 1.46 | 11321.25 | 1036.7 | 1092.1% |
| 90 | +ZUPT | 1913.39 | 24.86 | 2074.94 | 1036.7 | 200.2% |
| 90 | +CNN | 3512.26 | 23.31 | 5944.02 | 1036.7 | 573.4% |
| 90 | +NHC+matching | 1757.04 | 15.56 | 3324.54 | 1036.7 | 320.7% |
| 90 | +CNN beats +ZUPT (ties not wins) | 1/3 | - | - | - |

