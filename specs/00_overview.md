# SIH26168 AI-ML Intelligent Dead Reckoning

## Goal

Build an edge-deployable intelligent dead-reckoning system that combines an
error-state EKF, learned IMU-noise adaptation, zero-velocity updates (ZUPT),
OpenStreetMap (OSM) matching, and non-holonomic constraints (NHC).

## Problem Statement Summary

- **SIH:** SIH26168
- **Organization:** ISRO
- **Theme:** Miscellaneous
- **Category:** Software

The system estimates continuous pedestrian or vehicle position during GNSS
outages using inertial sensing and map context, with a native Android capture
and deployment path plus a Python training and evaluation engine.

## Deliverables

- Edge-deployable Python engine for data IO, training, replay, and evaluation.
- Native Android application for sensor recording and on-device deployment.
- IO-VNBD position plots included in the proposal.

## Architecture Summary

1. **Sensing:** Capture phone inertial and positioning observations.
2. **AI preprocessing:** Prepare measurements and adapt noise estimates.
3. **EKF:** Propagate and correct the navigation state.
4. **Map matching/NHC:** Apply OSM context and motion constraints.
5. **Output:** Produce position tracks, diagnostics, and evaluation plots.

**Phase A** records on the phone and replays through Python. **Phase B** moves
the Kotlin EKF and TFLite model onto the device.

## Accuracy Target

Maintain less than 1–2% horizontal error over a 30–60 second GNSS outage.

## Team Ownership

- **Person A:** `python/ekf`, `python/ml`, `python/eval`
- **Person B:** `python/io`, `android/`, `results/`

## References

- Brossard, Barrau, Bonnabel, “AI-IMU Dead-Reckoning,” IEEE T-IV (2020),
  arXiv:1904.06064, reference repository `mbrossar/ai-imu-dr`.
- Onyekpe et al., IO-VNBD dataset, repository `onyekpeu/IO-VNBD`.
- Dataset statistics: ⚠️ VERIFY against the source dataset and paper before use.
