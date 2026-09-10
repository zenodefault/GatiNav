package com.gatinav.android.engine

import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.hypot

/**
 * Real-time 10 Hz GNSS+INS fusion engine (Blueprint Phase 1.3 + 2.2,
 * on-device). Consumes raw IMU samples (accel/gyro/mag) and GNSS fixes,
 * runs the AHRS attitude compensation, the residual denoiser, the 15-state
 * error-state EKF and the FUSED/INERTIAL/REACQUIRING state machine, and
 * publishes the fused pose.
 *
 * The engine runs on a single background thread; samples are queued by the
 * service and drained in bursts, so the 10 Hz position output stays smooth
 * even when sensor callbacks burst at 100-200 Hz.
 */
data class EngineSample(
    val t: Float,           // session seconds
    val gyro: FloatArray,   // rad/s
    val accel: FloatArray,  // m/s^2
    val mag: FloatArray,    // unit vector
)

data class GnssFix(
    val t: Float,
    val latDeg: Double,
    val lonDeg: Double,
    val accuracyM: Float,
    val pdop: Float? = null,
    val satellites: Int? = null,
    val snr: Float? = null,
)

data class FusedPose(
    val t: Float,
    val latDeg: Double,
    val lonDeg: Double,
    val headingRad: Float,
    val speedMps: Float,
    val mode: FusionMode,
    val gnssValid: Boolean,
)

/** Local-tangent ENU frame, origin at the first valid GNSS fix. */
class EnuOrigin {
    var lat0Rad = 0.0
    var lon0Rad = 0.0
    var set = false

    companion object {
        private const val A = 6378137.0
        private const val EARTH_CUTOFF = 0.0
    }

    fun toEnu(latDeg: Double, lonDeg: Double, altM: Double = 0.0): FloatArray {
        val lat = Math.toRadians(latDeg)
        val lon = Math.toRadians(lonDeg)
        if (!set) {
            lat0Rad = lat
            lon0Rad = lon
            set = true
        }
        val x = (lon - lon0Rad) * Math.cos(lat0Rad) * A
        val y = (lat - lat0Rad) * A
        return floatArrayOf(x.toFloat(), y.toFloat(), altM.toFloat())
    }

    fun toLatLon(enu: FloatArray): Pair<Double, Double> {
        val lat = lat0Rad + enu[1] / A
        val lon = lon0Rad + enu[0] / (A * Math.cos(lat0Rad))
        return Pair(Math.toDegrees(lat), Math.toDegrees(lon))
    }
}

class NavigationEngine(
    private val denoiser: ResidualDenoiser,
    private val denoiseStrideS: Float = 0.5f,
    private val fs: Float = 100.0f,
) {
    private val samples = ConcurrentLinkedQueue<EngineSample>()
    private val fixes = ConcurrentLinkedQueue<GnssFix>()

    private val ekf = ErrorStateEKF()
    private val ahrs = Ahrs()
    private val state = FusionStateMachine()
    private val origin = EnuOrigin()

    private var lastProcessedT = 0f
    private var lastDenoiseT = -1e9f
    private var window = FloatArray(0)
    private var windowIdx = 0
    private var residual = 0f
    private var lastGnssT = -1e9f
    private var lastGnssEnu = FloatArray(3)
    private var lastGnssSpeed = 0f
    private var ahrsInitialised = false
    private var staticAccel = mutableListOf<FloatArray>()
    private var staticMag = mutableListOf<FloatArray>()
    private var staticGyro = mutableListOf<FloatArray>()
    private var staticCount = 0
    private val staticTarget = 500 // 5 s @ 100 Hz

    // publish state (volatile for cross-thread visibility)
    @Volatile var pose: FusedPose? = null
        private set
    @Volatile var gnssBlackout = false
        private set

    fun queueSample(sample: EngineSample) = samples.offer(sample)
    fun queueFix(fix: GnssFix) = fixes.offer(fix)

    /** Signal a simulated/real GNSS blackout (Blueprint seamless handler). */
    fun setBlackout(active: Boolean) {
        gnssBlackout = active
        if (active) {
            // transition to inertial only via the state machine on next tick
        }
    }

    /**
     * Drain pending samples, fuse, and publish a pose. Call at ~10 Hz from
     * the service thread. Returns true if a new pose was published.
     */
    fun tick(): Boolean {
        val nowSamples = mutableListOf<EngineSample>()
        while (true) {
            val s = samples.poll() ?: break
            nowSamples.add(s)
        }
        if (nowSamples.isEmpty()) return false

        // AHRS static initialisation from the first 5 s of samples
        for (s in nowSamples) {
            if (!ahrsInitialised) {
                staticAccel.add(s.accel)
                staticMag.add(s.mag)
                staticGyro.add(s.gyro)
                staticCount++
                if (staticCount >= staticTarget) {
                    ahrs.initStatic(
                        staticAccel.toTypedArray(),
                        staticMag.toTypedArray(),
                        staticGyro.toTypedArray(),
                    )
                    ahrsInitialised = true
                    staticAccel.clear(); staticMag.clear(); staticGyro.clear()
                }
                continue
            }
            break
        }
        if (!ahrsInitialised) return false

        for (s in nowSamples) {
            if (s.t <= lastProcessedT) continue
            val dt = (s.t - lastProcessedT).coerceAtLeast(1e-4f)
            ahrs.update(s.gyro, s.accel, s.mag, dt)

            // Phase 1.3: denoise the forward-axis acceleration
            val forward = ahrs.forwardAccel(s.accel)
            if (s.t - lastDenoiseT >= denoiseStrideS) {
                val windowSize = (denoiseStrideS * 5.0f * fs).toInt().coerceAtLeast(2)
                if (window.size != windowSize) {
                    window = FloatArray(windowSize)
                    windowIdx = 0
                }
                window[windowIdx % windowSize] = forward
                windowIdx++
                if (windowIdx >= windowSize) {
                    residual = denoiser.predict(window)
                    lastDenoiseT = s.t
                }
            }
            val level = ahrs.toLevel(s.accel)
            val denoised = level.copyOf()
            denoised[1] -= residual
            val accelCorrected = Mat.matVec(Mat.matTranspose(ahrs.rotation, 3), denoised, 3)

            ekf.predict(s.gyro, accelCorrected, dt)
            lastProcessedT = s.t
        }

        // GNSS fusion (skipped during blackout)
        val validFix = fixes.poll()
        if (validFix != null && !gnssBlackout) {
            val enu = origin.toEnu(validFix.latDeg, validFix.lonDeg)
            ekf.updateGnss(enu, kotlin.math.max(validFix.accuracyM, 1e-3f))
            val speed = if (lastGnssT > 0 && validFix.t > lastGnssT) {
                val d = hypot(enu[0] - lastGnssEnu[0], enu[1] - lastGnssEnu[1]).toFloat()
                d / (validFix.t - lastGnssT)
            } else 0f
            lastGnssEnu = enu
            lastGnssT = validFix.t
            lastGnssSpeed = speed
            state.reportHealth(
                validFix.t, validFix.pdop, validFix.satellites, validFix.snr,
                validFix.accuracyM,
            )
        } else {
            state.advance(lastProcessedT)
        }

        val (lat, lon) = origin.toLatLon(ekf.position)
        pose = FusedPose(
            t = lastProcessedT,
            latDeg = lat,
            lonDeg = lon,
            headingRad = ekf.yaw(),
            speedMps = hypot(ekf.velocity[0], ekf.velocity[1]),
            mode = state.mode,
            gnssValid = (validFix != null && !gnssBlackout),
        )
        return true
    }
}