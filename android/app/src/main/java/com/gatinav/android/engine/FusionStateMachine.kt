package com.gatinav.android.engine

/**
 * FUSED/INERTIAL/REACQUIRING state machine (Blueprint Phase 2.2) — port of
 * python/ekf/fusion_state.py including report_health (PDOP/sat/SNR).
 * Pure logic; the caller drives it with monotonic time.
 */
enum class FusionMode { FUSED, INERTIAL, REACQUIRING }

class FusionStateMachine(
    private val lossTimeoutS: Float = 2.0f,
    private val regainAccuracyM: Float = 10.0f,
    private val accuracyJumpRatio: Float = 3.0f,
    private val inflationFactor: Float = 10.0f,
    private val consistencyGate: Float = 9.0f,
) {
    var mode: FusionMode = FusionMode.FUSED
        private set

    // covariance is held as a 15x15 FloatArray (15x15 identity default);
    // inflated on every transition
    var covariance: FloatArray = Mat.eye(15)
        private set

    private var clock = 0f
    private var lastFixT = 0f
    private var acceptedAccuracyM = 5.0f
    private var pendingAccuracyM: Float? = null

    private fun requireTime(t: Float) {
        if (t <= clock) throw IllegalArgumentException("time must be strictly increasing")
        clock = t
    }

    private fun transition(next: FusionMode) {
        for (i in covariance.indices) covariance[i] *= inflationFactor
        mode = next
    }

    /** FUSED -> INERTIAL once GNSS lost > lossTimeoutS. */
    fun advance(t: Float): FusionMode {
        requireTime(t)
        if (mode == FusionMode.FUSED && t - lastFixT > lossTimeoutS) {
            transition(FusionMode.INERTIAL)
        }
        return mode
    }

    /** Accuracy-driven fix report (back-compat path). */
    fun reportFix(t: Float, accuracyM: Float): FusionMode {
        requireTime(t)
        lastFixT = t
        when (mode) {
            FusionMode.FUSED -> {
                if (accuracyM > accuracyJumpRatio * acceptedAccuracyM) {
                    transition(FusionMode.INERTIAL)
                } else {
                    acceptedAccuracyM = accuracyM
                }
            }
            FusionMode.INERTIAL -> {
                if (accuracyM < regainAccuracyM) {
                    pendingAccuracyM = accuracyM
                    transition(FusionMode.REACQUIRING)
                }
            }
            FusionMode.REACQUIRING -> pendingAccuracyM = accuracyM
        }
        return mode
    }

    /** Health-metric report: PDOP, satellite count, SNR (any can be null). */
    fun reportHealth(
        t: Float,
        pdop: Float? = null,
        satellites: Int? = null,
        snr: Float? = null,
        accuracyM: Float? = null,
        pdopMax: Float = 4.0f,
        minSatellites: Int = 4,
        minSnr: Float = 25.0f,
    ): FusionMode {
        requireTime(t)
        var healthy = true
        if (pdop != null) healthy = healthy && pdop <= pdopMax
        if (satellites != null) healthy = healthy && satellites >= minSatellites
        if (snr != null) healthy = healthy && snr >= minSnr
        when (mode) {
            FusionMode.FUSED -> if (!healthy) transition(FusionMode.INERTIAL)
            FusionMode.INERTIAL -> if (healthy) {
                pendingAccuracyM = accuracyM ?: acceptedAccuracyM
                transition(FusionMode.REACQUIRING)
            }
            FusionMode.REACQUIRING -> if (accuracyM != null) pendingAccuracyM = accuracyM
        }
        return mode
    }

    /** NIS = e^T S^-1 e consistency gate: REACQUIRING -> FUSED/INERTIAL. */
    fun resolveConsistency(innovation: FloatArray, innovationCovariance: FloatArray): FusionMode {
        if (mode != FusionMode.REACQUIRING) {
            throw IllegalStateException("resolveConsistency valid only in REACQUIRING")
        }
        // NIS via Cholesky (S assumed positive definite 3x3)
        val L = cholesky3(innovationCovariance)
        val y = solveLowerTri3(L, innovation)
        val nis = y[0] * y[0] + y[1] * y[1] + y[2] * y[2]
        if (nis <= consistencyGate) {
            acceptedAccuracyM = pendingAccuracyM ?: acceptedAccuracyM
            transition(FusionMode.FUSED)
        } else {
            transition(FusionMode.INERTIAL)
        }
        pendingAccuracyM = null
        return mode
    }

    private fun cholesky3(m: FloatArray): FloatArray {
        val l = FloatArray(9)
        for (i in 0 until 3) for (j in 0..i) {
            var sum = m[i * 3 + j]
            for (k in 0 until j) sum -= l[i * 3 + k] * l[j * 3 + k]
            l[i * 3 + j] = if (i == j) kotlin.math.sqrt(kotlin.math.max(sum, 0f)) else sum / l[j * 3 + j]
        }
        return l
    }

    private fun solveLowerTri3(l: FloatArray, b: FloatArray): FloatArray {
        val y = FloatArray(3)
        for (i in 0 until 3) {
            var sum = b[i]
            for (k in 0 until i) sum -= l[i * 3 + k] * y[k]
            y[i] = sum / l[i * 3 + i]
        }
        return y
    }
}