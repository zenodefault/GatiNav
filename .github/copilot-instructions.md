# Project rules — SIH26168 Dead Reckoning
- Frames: ENU (East-North-Up) local frame only. NEVER use NED unless the spec file says so.
- Units: gyro rad/s, accel m/s², timestamps seconds (float64). Convert at IO boundary only.
- Gravity: +9.80665 m/s² pointing UP in ENU. Remove from accel before integration.
- Quaternions: Hamilton convention (w,x,y,z). Rotation body→world via R(q).
- No code without a corresponding test in python/tests/. Math functions need known-answer tests.
- Allowed libs in python/: numpy, scipy, torch, osmnx, leuvenmapmatching,
  matplotlib, pandas, ahrs. Do NOT invent or import other libraries.
  gtsam is EXCLUDED from this project (optional in PS text; not used).
- pykalman is for LINEAR Kalman filters only — NEVER use it for the
  error-state EKF core.
- Mobile: native Android only (Kotlin + Jetpack Compose + osmdroid).
  No Flutter, no React Native, no cross-platform suggestions.
  All timestamp math lives in the service layer; the UI is display-only
  and never does math.
- NEVER run git commands (add/commit/push). Suggest a commit message at
  most; the team commits manually.
- Do not modify files in specs/ or results/.
- If a spec file conflicts with your training knowledge, the spec wins.
  State the conflict explicitly, don't silently choose.

