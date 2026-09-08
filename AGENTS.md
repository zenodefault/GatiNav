GatiNav — Agent Rules (SIH: Intelligent Dead Reckoning + GNSS Fusion)
Mission

Make this repo meet the SIH benchmark:

    DR drift < 10% of distance travelled during GNSS outage (≤5 m over 50 m; ≤100 m over 1 km @ 60 km/h; tested at 30/60/90 s outages)
    10 Hz fused output on Android; ≥200 Hz on the edge engine
    Deliverables: calibration engine, AI speed & vibration filter, map matching + NHC,AI-based GNSS+INS fusion, seamless GNSS↔DR switching, live nav UI, edge engine.

Physics budget — design every change against this

drift² ≈ along-track² + cross-track²; cross-track ≈ distance × sin(avg heading error)For 1 km / <100 m: along-track ≤3–5% (learned odometry), avg heading error ≤4–5°(init yaw <2° at outage start + gyro yaw drift <1°/60 s + map-heading re-anchoring).
Current measured state (verified 2025; re-verify by running, do not re-derive)

    Best drift: 570 m@30s, 386 m@60s, 4,114 m@90s (results/final_metrics.md)
    Known defects:
        NoiseNet speed head (output col 3) is computed but DISCARDED in python/ekf/ekf_cnn.py::update_zupt
        Yaw unobservable: magnetometer is loaded (python/io/iovnb_loader.py) but never used;no GNSS course-over-ground update in the EKF
        Held-out CNN gate "NOT VERIFIED" (results/metrics_table.md): Vw04 vehicle/smartphonerow-length mismatch in python/io/iovnb_loader.py blocks session loading
        Map matcher accepts 0/9 outage windows (hard 30 m residual gate inpython/eval/run_evaluation.py, MATCH_RESIDUAL_MAX_M)
        Window selection allows near-stationary windows → nan% / 42160% drift rows
        Fixed variance thresholds (ACCEL_VAR_MAX=1.45, GYRO_VAR_MAX=0.006 inpython/eval/engine.py) cannot separate idling/potholes from motion

Hard rules

    During any masked outage window, GNSS and ground truth must NEVER reach the filter.Initialization/anchoring uses only pre-outage data. Every change to python/eval mustkeep a test asserting this leakage guard.
    NEVER weaken a gate, threshold, or metric definition to make it pass. If a gate fails,stop and report the failure plus your top-3 diagnosis. A failed gate is a result.
    Every numeric claim must come from actually running python/eval/run_evaluation.py or acommitted test. No asserted or imagined metrics anywhere (docs, comments, commits).
    Outage windows are frozen: results/windows/*.json (created in Phase 0). Selection-logicchanges require regenerating windows and saying so in WORKLOG.md.
    Splits are frozen leave-one-driver/route/device-out: python/eval/splits.py (Phase 0).
    ML models: <500k params, deterministic inference, exportable to ONNX and TFLite,CPU-only ops. Training scripts set seeds and log curves to results/audit/.
    Update WORKLOG.md every session: date, phase, what changed, gate status (PASS/FAIL),evidence paths. Commit after every green test run.
    pytest must pass before any commit. No new heavyweight dependencies without statingjustification in WORKLOG.md.
    Data: data/IO-VNBD/Synchronised V abd S datasets (V-.csv + S-.csv pairs, 100 Hztarget clock). Never commit data files or weights >10 MB.
    Repo map: python/io (loaders/outages) · python/ekf · python/ml · python/matching ·python/eval · python/tests · android/ · results/ · specs/ (internal specs 01–04 —read the relevant spec before touching its area).
    GATE OWNERSHIP: Agents may not add, remove, reinterpret, or re-scope gates. The gatelist per phase is exactly what the phase prompt states. Extra checks may be run andreported as DIAGNOSTICS (clearly labeled, non-blocking), but a failed diagnostic mustnever block a commit or a phase verdict. Scope changes require human approval in theprompt itself.
    RED-SUITE POLICY: pytest must be green at session end. If a HUMAN-defined gate fails,keep the gate test marked xfail with the failure ID and document it; do not leave thesuite red over an agent-invented check.
    SANITY ARITHMETIC: before reporting a drift number, convert it to implied phantomacceleration (a = 2·drift/t²) and state it alongside. If a ≈ 1 g, suspect gravity/frame handling; if drift implies average error speed > vehicle speed, suspectinitialization. Report the check even when it passes.

Session protocol

On session start: read AGENTS.md and WORKLOG.md. On end: append WORKLOG.md and print agate table (gate | status | evidence path). Never leave the repo on a red test suite.
PHASE 0 — Evaluation integrity

Execute Phase 0 only: "Evaluation integrity + screening artifact".

Objective: make the evaluation trustworthy and reproducible; produce the IO-VNBDposition-plot artifact required for the SIH proposal screening.

Tasks:

    Fix python/io/iovnb_loader.py so the Vw04 vehicle/smartphone pair loads (row-lengthmismatch). Add a regression test loading every V-/S- pair found on disk.
    In python/eval/engine.py::find_windows: add a minimum-motion acceptance rule —≥100 m ground-truth travel inside the window AND no stationary stretch >20 s inside it.(Window discovery may use GT; the filter still may not.)
    Create python/eval/splits.py: frozen leave-one-driver-out split over sessions.
    Freeze and save the exact chosen 30/60/90 s outage windows to results/windows/*.json;run_evaluation.py must load them instead of rediscovering.
    Strengthen the leakage test: within a masked window, assert the filter receives noGNSS and no GT values.
    Regenerate results/final_metrics.md + final_comparison.png. Every row must reporttravelled distance and drift/distance — no nan%, no degenerate percentages.
    Create results/screening/: 3–5 position plots (GT green, GNSS-frozen grey, estimateblue) + a metrics summary suitable for the SIH proposal PDF.

Gate (all must be TRUE, verified by running):

    pytest passes
    Vw04 loads; windows reproduce from saved JSON
    zero nan%/degenerate rows in final_metrics.md
    leakage test red→green demonstrable (temporarily feed GNSS in outage → test fails)

Report: gate table + paths. Do NOT touch ekf math, ML, or matching in this phase.
PHASE 1 — Calibration & alignment engine

Execute Phase 1 only: "In-vehicle alignment & calibration engine". No ML training.

Objective: new module python/calibration/engine.py + python/io/preprocessing.py thattransforms every IMU sample from phone frame to vehicle frame before EKF propagation.

Implement:

    Stationary detection (10–20 s) → gyro bias + accelerometer bias estimation, using theexisting variance detector only to FIND the stationary window, not at runtime.
    Gravity leveling → phone pitch/roll (reuse python/ekf/ekf.py::level_rotation).
    Yaw calibration: during straight GNSS-valid driving (speed >3 m/s, course std <2°,straightness check), estimate phone-frame yaw vs GNSS course-over-ground. Resolve the±forward-axis ambiguity with the dominant gyro axis during turns.
    Mount-movement detector: sustained gravity-direction jump or heading-vs-GNSS-coursedivergence >15° → raise MOUNT_MOVED, pause DR, force recalibration.
    Output a phone→vehicle rotation R_pv applied to accel+gyro in preprocessing; EKF seesvehicle-frame IMU only.
    Unit tests: synthetic rotated IMU (known R_pv) → recovered R_pv within 2°/0.1 m/s².

Gate:

    <2 m drift over a 30 s stationary run (bias-corrected)
    zero MOUNT_MOVED triggers during a ≥60 s engine-idle-only segment
    R_pv stable within 3° across a dashboard-mount session and a holder-mount session

Report gate table + evidence paths. If the yaw gate fails on a session, report which andwhy (insufficient straight driving vs estimator bug) — do not proceed to Phase 2.
PHASE 2 — GNSS-aided EKF + seamless handler

Execute Phase 2 only: "GNSS-aided convergence + Seamless GNSS Deficit Handler".

Objective: converged heading/biases before every outage; millisecond mode switching;smooth reacquisition. All changes in python/ekf/ekf.py, python/eval/engine.py, and a newpython/core/mode_manager.py (framework-agnostic — Android will reuse the logic).

Implement:

    GNSS course-over-ground as a yaw pseudo-measurement when speed >2 m/s (R scaled byspeed and fix accuracy). 
    Magnetometer heading as a secondary aid: hard/soft-iron calibration on the pre-outagestationary window; innovation-gated so urban anomalies are rejected.
    ModeManager states: FUSED → INERTIAL (on GNSS loss, next tick, ≤1 update period) →REACQUIRING (fixes gated by chi-square, covariance contracts gradually) → FUSED.Log every transition with timestamp.
    Outage start hygiene: every outage begins from the last converged state. Persist lastconverged heading + confidence; if outage starts at standstill, reuse it and inflateyaw covariance so later map headings can re-anchor.
    Reacquisition quality metric: max position jump at re-lock ≤2 m; report time-to-reconverge.

Gate:

    transitions fire correctly in replay (unit test with synthetic GNSS dropout)
    covariance grows during blackout, contracts after recovery (assert monotone phases)
    no replay position jump >2 m at re-lock
    pre-outage yaw error <2° averaged over ≥5 windows (vs GT heading)
    mode-switch latency <1 update period (<500 ms at 10 Hz processing)

Report gate table. Do not modify the CNN or matcher in this phase.
PHASE 3 — Motion classifier (AI Speed & Vibration Filter, part 1)

Execute Phase 3 only: "Motion-state classifier, ZUPT policy, shock handling".

Objective: replace fixed variance thresholds with a learned window classifier; wire itinto the EKF policy. Touch python/ml/ (new classifier + data pipeline), python/ekf/ekf_cnn.py.

Implement:

    Window classifier (1D-CNN, <100k params, input 6×100 @100 Hz): classes = stationary/idling, driving, turning, braking/accelerating, shock/pothole, mount-disturbance.Labels from IO-VNBD GT speed + accel/gyro signatures (document labeling rules in thescript); leave-one-driver-out eval via python/eval/splits.py.
    Policy wiring:
        ZUPT fires only on high-confidence stationary (p>0.9), replacing the variance gate
        shock/pothole windows: down-weight accelerometer updates (inflated R) for that window
        mount-disturbance: emit event consumed by Phase 1 recalibration
        classifier probability also scales ZUPT measurement covariance (adaptive noise)
    Keep the existing NoiseNet untouched. Train with seeds; save topython/ml/weights/motion_clf.pt + ONNX; curves and confusion matrices to results/audit/.

Gate:

    held-out idling recall >95%; false-ZUPT rate <1% of driving windows
    shock windows measurably reduce accel-update weight (unit test)
    stationary 30 s drift still <2 m with the new ZUPT path
    inference latency <2 ms/window on CPU

Report gate table + confusion matrix path. Do NOT touch the speed head or matcher.
PHASE 4A — Wire the existing speed head (do this before 4B)

Execute Phase 4A only: "End-to-end wiring of the existing NoiseNet speed head + ablation".

Objective: the speed output (NoiseNet col 3) currently reaches the filter and is dropped.Wire it, measure it, and produce the ablation evidence for/against Phase 4B.

Tasks:

    In python/ekf/ekf_cnn.py: consume col 3 as a body-frame forward-velocity measurementz=[v,0,0] rotated into ENU by current attitude; R per-axis from the held-out fractionalRMSE recorded in results/audit/cnn_training.md (scale: lateral/vertical get NHC R).
    Gate the speed update by the Phase 3 classifier (driving/braking only; not stationary,not shock, not mount-moved).
    Add config key to python/eval configs: "+CNN-var+speed" (cumulative).
    Run the full outage evaluation; add the ablation rows to results/final_metrics.md:raw / +ZUPT / +CNN-var / +CNN-var+speed / +NHC+matching.
    Diagnose: is residual error along-track (speed scale) or cross-track (heading)?State the split explicitly in results/audit/speed_head_ablation.md.

Gate:

    ablation table complete for 30/60/90 s over the frozen windows
    held-out fractional speed RMSE stated numerically (all windows + driving-only)
    leakage test still green

Decision rule (print explicitly at the end): if held-out fractional speed RMSE ≤5%,Phase 4B is DEFERRED; if >10% (expected per prior correlation analysis), Phase 4B isGO. Do not editorialize beyond the numbers.
PHASE 4B — Learned odometry model (make-or-break)

Execute Phase 4B only: "Learned odometry model replacing vibration-RMS speed prediction".

Objective: a model that predicts vehicle forward motion from calibrated IMU history withuncertainty — trained on trajectory loss, fused as a probabilistic EKF measurement.

Model spec (python/ml/odom/):

    Input: gravity-aligned accel+gyro AFTER Phase 1 R_pv, plus current attitude features;window 2–10 s sliding (pick one, justify from data), stride 0.5 s at inference.
    Output heads: forward displacement (m), forward velocity (m/s), yaw increment (rad),per-output heteroscedastic uncertainty (predict log-variance; train with Gaussian NLL —never feed a point estimate to the filter).
    Architecture: GRU or TCN, <500k params, ONNX+TFLite exportable, CPU inference <5 msper window on a mid-range phone (benchmark and record it).

Training (python/ml/odom/train.py):

    Supervision: IO-VNBD wheel-encoder speed / GNSS-derived velocity — TRAINING ONLY.
    Loss: displacement/trajectory loss (dominant) + NLL on uncertainty heads.
    Splits: frozen leave-one-driver-out (python/eval/splits.py). No session appears in bothtrain and eval.
    Augmentation for generalization: mount-pose randomization (random R_pv perturbations),per-device normalization stats carried with the model, noise injection at realistic MEMSlevels. Log augmented-vs-clean held-out delta.

Fusion (python/ekf/):

    Model outputs enter the EKF as probabilistic measurements using predicted σ; gate byPhase 3 classifier; keep NHC and map-heading updates active alongside.

Validation & leakage:

    Unit test: outage-window filter receives no GNSS/GT (still green).
    Hold out entire drivers AND one full route; report fractional forward-distance error(all windows + driving-only), median and p90.

Gate:

    held-out forward-distance error ≤5% (target 3%) on driving windows
    beats calibrated-INS-alone (Phase 2 config) on EVERY held-out session
    ONNX export round-trip: max output deviation <1e-3 vs PyTorch
    CPU latency ≤5 ms/window recorded

If the gate fails: STOP, write results/audit/odom_failure.md with the failure split(bias? scale? turns? device?), and recommend: more data vs hybrid fallback. Do notproceed to Phase 5 with a failing odometry model unless the human overrides.
PHASE 5 — Map matching + NHC

Execute Phase 5 only: "Make map matching actually fire and help".

Current fact: matcher accepts 0/9 windows. Audit first, tune second.

Tasks (python/matching/matcher.py, python/eval/run_evaluation.py):

    Audit: for each frozen window, print the session's GNSS bounding box, downloaded graphnode count, and one pre-outage GNSS track matched directly — if GNSS itself doesn'tmatch, the graph is wrong; fix graph source/region before touching the algorithm.
    Keep the ENU→graph CRS transform and the ~10 s pre-outage GNSS anchor. Replace thehard 30 m acceptance gate with: HMM scored on position + heading + speed + topology(turn feasibility); acceptance = posterior probability threshold, not distance alone.
    Maintain multiple road hypotheses near junctions/parallel roads (do not force-commit).
    NHC (v_lateral≈0, v_vertical≈0) activates only after sustained confident match(≥3 s same road, posterior >0.8). Map heading feeds the EKF as a SOFT yaw update withR growing with residual — never overwrite along-track position.
    Cache graphs per session (already partially done in python/matching/graphs/).

Gate:

    ≥70% of moving windows get an accepted match
    median drift with matching ≤ median drift without (Phase 4 config), per duration
    zero wrong-road snaps on a hand-labeled junction/parallel-road test set (build it:≥5 windows with known ambiguity, documented)
    leakage test still green

Report gate table + per-window match table. If GNSS-anchor matching fails on any window,report it as a data/region issue, not an algorithm failure.
PHASE 6 — Acceptance benchmark

Execute Phase 6 only: "Full offline acceptance benchmark + diagnosis". No new features.

Protocol:

    Run the complete system (Phases 1–5 chain) on ALL held-out IO-VNBD sessions AND ≥2self-collected recordings (one car, one two-wheeler if available) with simulated outages.
    Report per duration (30/60/90 s), separately per dataset: ATE, RPE@1s, RPE@10s, finaldrift, drift/distance — median, p90, p95 — plus match-acceptance rate, ZUPT rate,odometry error, reacquisition jump. Emit results/acceptance/*.md + plots.
    Acceptance (PS benchmark): <5 m over 50 m; <100 m over 1 km @60 km/h; <10%drift/distance on most held-out scenarios; defined low-confidence fallback behavior(state it: what the app shows when match+ZUPT confidence is low).
    Diagnosis procedure — run and report the four-way split:cross-track dominant → heading problem (Phase 2/5);along-track dominant → speed scale problem (Phase 4);slow growth from t=0 → bias problem (Phase 1);junction-only spikes → map ambiguity (Phase 5).

Gate: benchmark table PASS/FAIL per line, with the diagnosis split for any FAIL.If any line fails: STOP. Do not write any "meets the benchmark" claim anywhere.
PHASE 7 — Android application

Execute Phase 7 only: "Android port + live navigation UI". Only start if Phase 6 passed.

Tasks (android/):

    Kotlin port of: preprocessing+calibration (R_pv pipeline), 15-state EKF, ModeManager,motion classifier policy. Numerical parity is the priority: identical operation orderand float32/float64 policy as Python.
    Model export: NoiseNet/motion classifier/odometry → TFLite (fallback ONNX Runtime);verify outputs vs Python within 1e-2 on 100 sampled windows.
    Live loop: SensorRecordingService → 100 Hz ring buffer → fusion worker → 10 Hz posepublish → Compose map with smooth vehicle icon (interpolate between fixes; no jumping).
    Bundle offline map tiles + cached road graph for the demo area.
    Parity harness: record a drive, replay through BOTH Python and Android; max posedivergence must be within tolerance (state it, default 3 m over 60 s).
    Profiling: sustained 10 Hz for 30 min — record latency p50/p95, memory, battery drain,sensor drop rate. Emit results/android_profiling.md.
    Demo mode: manual blackout toggle + GNSS recovery during a vehicle run.

Gate:

    10 Hz sustained on a mid-range phone (name it) for 30 min without ANR/OOM
    parity within tolerance on the same recording
    live demo script rehearsed: 60 s tunnel/blackout + recovery, no UI freeze, no jump >2 m

PHASE 8 — Edge engine

Execute Phase 8 only: "Edge-deployable engine (FOG IMU, ~200 Hz)".

Tasks:

    Package the Python core as a CLI service (python/edge/): ingest IMU via serial or CSVat up to 200 Hz (any external IMU — do not assume a specific vendor), same ONNX models,same EKF/ModeManager; publish pose via UDP/stdout at configurable rate ≥200 Hz.
    Decouple rates: EKF predict at IMU rate; model inference and matching on their owncadence; document the threading model.
    Profile on Raspberry Pi 5 / Jetson-class: sustained ≥200 Hz with model in the loop;record CPU per stage. Emit results/edge_profiling.md.

Gate:

    ≥200 Hz sustained for 10 min on target hardware, zero dropped samples
    same frozen-window replay within tolerance of the phone/Python results
    README section: how to plug a different external IMU (frame, units, rates)