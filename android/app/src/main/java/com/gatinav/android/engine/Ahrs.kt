package com.gatinav.android.engine

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Complementary-filter AHRS (Blueprint Phase 1.1) — port of
 * python/calibration/ahrs.py. Fuses accelerometer (gravity -> pitch/roll),
 * magnetometer (magnetic north -> yaw) and gyroscope (propagation).
 *
 * Static init aligns the phone z-axis with gravity and y-axis with
 * magnetic north; during motion the gyro propagates and accel+mag correct.
 * Output: body->level rotation (level = R @ body, z up, y mag-north).
 */
class Ahrs(
    private val kp: Float = 0.5f,
    private val ki: Float = 0.0f,
    private val magGain: Float = 0.3f,
) {
    private var q = floatArrayOf(1f, 0f, 0f, 0f) // Hamilton (w, x, y, z)
    private var gyroBias = FloatArray(3)
    private var integral = FloatArray(3)
    var initialised = false
        private set

    /** Quaternion product (Hamilton). */
    private fun quatMultiply(a: FloatArray, b: FloatArray): FloatArray {
        val aw = a[0]; val ax = a[1]; val ay = a[2]; val az = a[3]
        val bw = b[0]; val bx = b[1]; val by = b[2]; val bz = b[3]
        return floatArrayOf(
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    }

    fun quaternionToRotation(qq: FloatArray): FloatArray {
        val w = qq[0]; val x = qq[1]; val y = qq[2]; val z = qq[3]
        val n = sqrt(w * w + x * x + y * y + z * z)
        if (n == 0f) return Mat.eye(3)
        val (w2, x2, y2, z2) = floatArrayOf(w / n, x / n, y / n, z / n)
        return floatArrayOf(
            1 - 2 * (y2 * y2 + z2 * z2), 2 * (x2 * y2 - z2 * w2), 2 * (x2 * z2 + y2 * w2),
            2 * (x2 * y2 + z2 * w2), 1 - 2 * (x2 * x2 + z2 * z2), 2 * (y2 * z2 - x2 * w2),
            2 * (x2 * z2 - y2 * w2), 2 * (y2 * z2 + x2 * w2), 1 - 2 * (x2 * x2 + y2 * y2),
        )
    }

    /** Static initialisation from a stretch of phone-frame accel+mag (N, 3). */
    fun initStatic(accel: Array<FloatArray>, mag: Array<FloatArray>, gyro: Array<FloatArray>? = null): Ahrs {
        if (accel.size < 5 || mag.size < 5) throw IllegalArgumentException("need >= 5 static samples")
        val gravity = floatArrayOf(0f, 0f, 0f)
        for (a in accel) for (i in 0 until 3) gravity[i] += a[i]
        for (i in 0 until 3) gravity[i] /= accel.size
        val gNorm = Mat.norm(gravity)
        if (gNorm < 1e-6f) throw IllegalArgumentException("static accelerometer magnitude ~0")
        val z = floatArrayOf(gravity[0] / gNorm, gravity[1] / gNorm, gravity[2] / gNorm)
        val field = floatArrayOf(0f, 0f, 0f)
        for (m in mag) for (i in 0 until 3) field[i] += m[i]
        for (i in 0 until 3) field[i] /= mag.size
        val dot = field[0] * z[0] + field[1] * z[1] + field[2] * z[2]
        var y = floatArrayOf(field[0] - dot * z[0], field[1] - dot * z[1], field[2] - dot * z[2])
        val yNorm = Mat.norm(y)
        if (yNorm < 1e-6f) throw IllegalArgumentException("magnetometer has no horizontal component")
        y = floatArrayOf(y[0] / yNorm, y[1] / yNorm, y[2] / yNorm)
        val x = floatArrayOf(
            y[1] * z[2] - y[2] * z[1],
            y[2] * z[0] - y[0] * z[2],
            y[0] * z[1] - y[1] * z[0],
        )
        val xNorm = Mat.norm(x)
        val xUnit = floatArrayOf(x[0] / xNorm, x[1] / xNorm, x[2] / xNorm)
        // rotation rows = level axes in body coords; convert to quaternion via
        // the standard orthonormal->quaternion formula
        q = rotationToQuaternion(floatArrayOf(
            xUnit[0], y[0], z[0],
            xUnit[1], y[1], z[1],
            xUnit[2], y[2], z[2],
        ))
        if (gyro != null) {
            val mean = FloatArray(3)
            for (g in gyro) for (i in 0 until 3) mean[i] += g[i]
            for (i in 0 until 3) mean[i] /= gyro.size
            gyroBias = mean
        }
        integral = FloatArray(3)
        initialised = true
        return this
    }

    private fun rotationToQuaternion(r: FloatArray): FloatArray {
        val trace = r[0] + r[4] + r[8]
        if (trace > 0f) {
            var s = 2f * sqrt(trace + 1f)
            val qq = floatArrayOf(
                0.25f * s,
                (r[7] - r[5]) / s,
                (r[2] - r[6]) / s,
                (r[3] - r[1]) / s,
            )
            return normalizeQuat(qq)
        }
        val i = maxOf(0, maxOf(1, 2).let { 0 }) // pick the largest diagonal
        var i0 = 0
        var maxDiag = r[0]
        for (idx in 1..2) if (r[idx * 3 + idx] > maxDiag) { maxDiag = r[idx * 3 + idx]; i0 = idx }
        val j = (i0 + 1) % 3
        val k = (i0 + 2) % 3
        var s = 2f * sqrt(1f + r[i0 * 3 + i0] - r[j * 3 + j] - r[k * 3 + k])
        val qq = FloatArray(4)
        qq[i0 + 1] = 0.25f * s
        qq[0] = (r[k * 3 + j] - r[j * 3 + k]) / s
        qq[j + 1] = (r[j * 3 + i0] + r[i0 * 3 + j]) / s
        qq[k + 1] = (r[k * 3 + i0] + r[i0 * 3 + k]) / s
        return normalizeQuat(qq)
    }

    private fun normalizeQuat(qq: FloatArray): FloatArray {
        val n = sqrt(qq.sumOf { it * it })
        return if (n < 1e-12f) qq else floatArrayOf(qq[0] / n, qq[1] / n, qq[2] / n, qq[3] / n)
    }

    /** Mahony step: gyro propagation + accel/mag correction. */
    fun update(gyro: FloatArray, accel: FloatArray, mag: FloatArray? = null, dt: Float = 0.01f) {
        if (!initialised) throw IllegalStateException("call initStatic before update")
        val rotation = quaternionToRotation(q)
        // predicted gravity/north in body frame
        val gravityBody = Mat.matVec(Mat.matTranspose(rotation, 3), floatArrayOf(0f, 0f, 1f), 3)
        val northBody = Mat.matVec(Mat.matTranspose(rotation, 3), floatArrayOf(0f, 1f, 0f), 3)
        val error = FloatArray(3)
        val aNorm = Mat.norm(accel)
        if (aNorm > 1e-6f) {
            val aHat = floatArrayOf(accel[0] / aNorm, accel[1] / aNorm, accel[2] / aNorm)
            val cross = floatArrayOf(
                gravityBody[1] * aHat[2] - gravityBody[2] * aHat[1],
                gravityBody[2] * aHat[0] - gravityBody[0] * aHat[2],
                gravityBody[0] * aHat[1] - gravityBody[1] * aHat[0],
            )
            for (i in 0 until 3) error[i] += cross[i]
        }
        if (mag != null) {
            val aRef = if (aNorm > 1e-6f) floatArrayOf(accel[0] / aNorm, accel[1] / aNorm, accel[2] / aNorm) else gravityBody
            val dotM = mag[0] * aRef[0] + mag[1] * aRef[1] + mag[2] * aRef[2]
            val horizontal = floatArrayOf(mag[0] - dotM * aRef[0], mag[1] - dotM * aRef[1], mag[2] - dotM * aRef[2])
            val hNorm = Mat.norm(horizontal)
            if (hNorm > 1e-6f) {
                val hUnit = floatArrayOf(horizontal[0] / hNorm, horizontal[1] / hNorm, horizontal[2] / hNorm)
                val cross = floatArrayOf(
                    northBody[1] * hUnit[2] - northBody[2] * hUnit[1],
                    northBody[2] * hUnit[0] - northBody[0] * hUnit[2],
                    northBody[0] * hUnit[1] - northBody[1] * hUnit[0],
                )
                for (i in 0 until 3) error[i] += magGain * cross[i]
            }
        }
        for (i in 0 until 3) integral[i] += ki * error[i] * dt
        val omega = floatArrayOf(
            (gyro[0] - gyroBias[0]) + kp * error[0] + integral[0],
            (gyro[1] - gyroBias[1]) + kp * error[1] + integral[1],
            (gyro[2] - gyroBias[2]) + kp * error[2] + integral[2],
        )
        val delta = quatMultiply(q, floatArrayOf(0f, omega[0], omega[1], omega[2]))
        val dtHalf = 0.5f * dt
        q = normalizeQuat(floatArrayOf(
            q[0] + delta[0] * dtHalf,
            q[1] + delta[1] * dtHalf,
            q[2] + delta[2] * dtHalf,
            q[3] + delta[3] * dtHalf,
        ))
    }

    val rotation: FloatArray get() = quaternionToRotation(q)

    fun toLevel(accel: FloatArray): FloatArray = Mat.matVec(rotation, accel, 3)

    /** Forward-axis (level y) acceleration scalar. */
    fun forwardAccel(accel: FloatArray): Float = toLevel(accel)[1]
}