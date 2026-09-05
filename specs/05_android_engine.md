# Native Android Engine Specification

This specification covers a single-process native Android application using
Kotlin, Jetpack Compose, and osmdroid. Filter behavior is defined only by
`specs/02_ekf.md` and the NHC measurement in `specs/04_map_matching_nhc.md`;
this document contains no duplicate filter equations.

## 1. Architecture

```text
+---------------------------------------------------------------+
|                    Single-process Android app                 |
|                                                               |
|  +----------------------+       +--------------------------+  |
|  | ForegroundService    |       | ViewModel                |  |
|  |                      |       |                          |  |
|  | SensorManager        | <---- | StateFlow<UiState>       |  |
|  | GNSS client          |       | session commands         |  |
|  | CSV recorder         | ----> |                          |  |
|  | Phase B EKF + TFLite |       +------------+-------------+  |
|  +----------------------+                    |                |
|                                              v                |
|                                  +--------------------------+ |
|                                  | One Compose screen       | |
|                                  | osmdroid MapView        | |
|                                  +--------------------------+ |
+---------------------------------------------------------------+
```

**Phase A:** record CSV on the phone, run the Python engine offline, and
replay the results onto the map.

**Phase B:** run the Kotlin EKF and TFLite model inside the service and publish
pose updates at 10 Hz.

## 2. Kotlin sensor core contract

### Service and lifecycle

- Acquisition runs in an Android `ForegroundService`.
- The service owns a partial `PowerManager.WakeLock` for active recording.
- The service creates and uses a notification channel and posts a foreground
  notification.
- The service must release the wake lock, unregister sensors, stop location
  updates, and close the recording file on every normal stop path.
- Notification-channel behavior and foreground-service start restrictions are
  ⚠️ VERIFY for the target Android SDK.

### Inertial sensors

Register a `SensorEventListener` with `SensorManager` for:

- `Sensor.TYPE_ACCELEROMETER`
- `Sensor.TYPE_GYROSCOPE`
- `Sensor.TYPE_MAGNETIC_FIELD`

Use `SensorManager.SENSOR_DELAY_FASTEST`. The actual delivered rate,
hardware availability, batching behavior, and event coalescing are
⚠️ VERIFY per device and Android version.

### Timestamp contract

Use `SensorEvent.timestamp` for sensor samples. Never use wall-clock time for
filtering or interval calculations. Convert timestamps to seconds relative to
a monotonic session clock.

Android sensor timestamp timebases have changed or differ across Android
versions and devices (for example, event uptime versus
`SystemClock.elapsedRealtimeNanos()` alignment). The exact API-24 boundary and
the behavior of each supported device are ⚠️ VERIFY.

Runtime detection procedure:

1. At session start, capture `SensorEvent.timestamp`,
   `SystemClock.elapsedRealtimeNanos()`, and `System.nanoTime()` close together.
2. Compare the sensor timestamp against each monotonic candidate over an
   initial calibration interval.
3. Select the candidate with the stable offset and low residual jitter.
4. Store the detected timebase name and offset in the session manifest.
5. Convert all accepted sensor timestamps to seconds relative to the selected
   monotonic session origin.
6. Reject or flag events that move backward after conversion.

The exact calibration duration, residual threshold, and fallback behavior are
⚠️ VERIFY. `SystemClock.elapsedRealtimeNanos()` is the preferred session-clock
reference because it is monotonic and includes time spent asleep; confirm this
choice on the target devices.

Sensor FIFO and hardware batching differ across Android versions and sensor
implementations. FIFO depth, maximum report latency, and whether
`SENSOR_DELAY_FASTEST` causes batching are ⚠️ VERIFY.

### GNSS

Use `FusedLocationProviderClient` for location updates containing position,
accuracy, and bearing. This client is supplied by Google Play services rather
than the Android framework; dependency version, availability, permissions, and
provider behavior are ⚠️ VERIFY.

GNSS timestamps must be aligned to the monotonic sensor session clock. Prefer
the location timestamp's elapsed-realtime field when present; the exact
Android `Location` timestamp accessor and conversion procedure are
⚠️ VERIFY for the target API levels.

TODO: On each supported Android version, record one sensor event and one
location update, compare their monotonic timestamps, and document the measured
offset, jitter, provider, and delivered rates.

### Ring buffer

- Maintain a ring buffer covering 10 s at the maximum measured sensor rate.
- When full, drop the oldest sample.
- Expose a monotonically increasing drop counter through the service state.
- Record buffer capacity, current occupancy, and drop count in live statistics.

The maximum measured rate and resulting device-specific capacity are
⚠️ VERIFY.

## 3. Recording mode (Phase A)

Write session files under the app's internal `filesDir`. The primary CSV
schema is:

```text
t_s,gx,gy,gz,ax,ay,az,mx,my,mz,gnss_lat,gnss_lon,gnss_acc,gnss_t_s,engine_state
```

The CSV writer shall preserve the canonical units and timestamp convention
from `specs/01_data_iovnbd.md`. Missing GNSS fields are represented using one
documented CSV representation; that representation is ⚠️ VERIFY.

Write a session manifest JSON alongside the CSV containing:

```text
device model
Android version
measured sampling rates
start time
app version
runtime-detected timestamp timebase
```

The JSON key names, start-time representation, sampling-rate summary format,
and timestamp-timebase metadata schema are ⚠️ VERIFY.

## 4. UI contract

The app has one Compose screen. The screen contains an osmdroid `MapView` and
three simple polylines:

- Raw GNSS: grey.
- Inertial-only trajectory: red.
- Fused output: blue.

The exact Compose/osmdroid interoperability wrapper and osmdroid dependency
version are ⚠️ VERIFY. `MapView` and `Polyline` are candidate osmdroid APIs
that must be checked against the selected dependency source before coding.

Controls and display:

- Blackout toggle button: mute GNSS measurements inside the engine. This is
  for airplane-mode-free demonstrations and must not delete recorded GNSS.
- Start session.
- Stop session.
- Live sampling rate (Hz).
- Buffer drops.
- Processing latency.

The UI reads `ViewModel` `StateFlow` only. Compose code must never compute
timestamps, positions, filtering, map matching, distances, or metrics.

The exact `StateFlow` exposure pattern, map invalidation/update threading, and
blackout visual indicator are ⚠️ VERIFY.

## 5. Phase B engine

Implement a Kotlin transcription of the Python EKF specified in
`specs/02_ekf.md`. It must use the same human-filled equations and port the
known-answer tests where practical. No alternate filter formulation may be
introduced in the Android layer.

Road graph acquisition, projection, and HMM matching remain offline/Python
work. The service may consume a compact matched-road record:

```text
timestamp_s, matched_edge_id, road_heading_rad, match_quality, match_accepted
```

When `match_accepted` is true, apply the one-row NHC update from
`specs/04_map_matching_nhc.md`. Do not run `osmnx` or
`leuvenmapmatching` on the phone.

Decision note: default to pure Kotlin. Escalate to C++/JNI only if profiling
shows that pure Kotlin cannot meet the measured latency or battery budget.
The profiling threshold and escalation decision are ⚠️ VERIFY.

Run the TFLite interpreter in the foreground service using the TFLite Android
SDK only after the fixed-noise EKF and NHC path are accepted. Perform CNN inference every 0.5 s window. The exact interpreter package,
delegate, tensor API, model input/output names, and threading configuration
are ⚠️ VERIFY.

Before enabling Phase B deployment, compare the Android model outputs against
the exported model and complete the operator-compatibility check. This check
must occur after model export in week 3, not week 6. The final result is
⚠️ VERIFY.

## 6. Battery and performance budget

Target battery consumption is less than 8% per 10-minute session.

- All sensor handling, file IO, EKF work, match-record handling, and TFLite
  inference run off the main thread. Python map matching is offline and is not
  an Android workload.
- The partial wake lock is held only while an active session requires it.
- The service must continue collecting while the screen is off.
- Screen-off operation must not drop samples; any observed drops fail the
  recording acceptance test.
- Release the wake lock and service resources promptly when recording stops.

The battery-measurement device, baseline state, thermal conditions, latency
budget, and acceptable transient wake-lock duration are ⚠️ VERIFY.

TODO: Measure a 10-minute screen-on and screen-off session on each target
device, record battery percentage, delivered sensor rate, drop count,
processing latency, and wake-lock duration.

## 7. Manifest permissions

Request only permissions required by the selected target SDK and features:

```xml
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
<uses-permission android:name="android.permission.HIGH_SAMPLING_RATE_SENSORS" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
```

`FOREGROUND_SERVICE_LOCATION`,
`HIGH_SAMPLING_RATE_SENSORS`, and `POST_NOTIFICATIONS` are
version/target-SDK-sensitive; exact permission names, declaration
requirements, and runtime-request requirements are ⚠️ VERIFY.

TODO: Validate this manifest against the target compile/target SDK and run a
fresh install test for permission prompts, foreground-service startup, GNSS,
notifications, and high-rate sensor delivery.

## 8. Sensor-HAL interface

The phone sensor implementation shall be replaceable by an external Bluetooth
IMU through a narrow interface. The interface carries already normalized
timestamps and sensor samples; unit conversion remains at the IO boundary.

```kotlin
interface SensorHal {
    data class Sample(
        val tSeconds: Double,
        val gyroRadPerSecond: Triple<Double, Double, Double>?,
        val accelMetersPerSecondSquared: Triple<Double, Double, Double>?,
        val magnetometer: Triple<Double, Double, Double>?
    )

    fun start(onSample: (Sample) -> Unit)
    fun stop()
    fun samplingRatesHz(): Map<String, Double>
    fun droppedSampleCount(): Long
}
```

Bluetooth transport, connection lifecycle, packet framing, clock
synchronization, reconnection policy, and whether a nullable channel is
acceptable are ⚠️ VERIFY.

## 9. Test plan

### Two-minute recording and replay

Record a two-minute walk session on a target device, replay the CSV through
the Python engine, and assert:

- Zero dropped samples.
- Strictly monotonic `t_s` values.
- Manifest contains the runtime-detected timestamp timebase.
- Replay accepts the CSV schema and produces map-displayable trajectories.

The walk protocol, target device, acceptable latency, and whether GNSS may be
present during the recording are ⚠️ VERIFY.

### CSV round trip

Unit-test writing and reading a representative session, preserving:

- Header order.
- Numeric values within a documented tolerance.
- Missing GNSS representation.
- `engine_state`.
- Timestamp monotonicity.

The numeric tolerance and missing-value representation are ⚠️ VERIFY.

### Blackout toggle

Unit-test a mid-session GNSS blackout toggle:

1. Feed GNSS updates before the toggle.
2. Toggle blackout on.
3. Confirm GNSS measurements are muted inside the engine while recording
   continues.
4. Confirm the engine state transitions to `INERTIAL`.
5. Toggle blackout off and feed a valid GNSS fix.
6. Confirm reacquisition follows the state machine in
   `specs/04_map_matching_nhc.md`.

The exact test harness, transition timing, and state-observation API are
⚠️ VERIFY.

Map choice: osmdroid is the primary OSM tile/offline-cache/simple-Polyline
choice. MapLibre is a fallback only if osmdroid blocks required behavior,
because changing map SDKs would otherwise add integration and rendering
complexity.
