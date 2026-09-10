package com.gatinav.android.engine

import kotlin.math.cos
import kotlin.math.sin

/**
 * 15-state ENU error-state EKF, a faithful port of
 * python/ekf/ekf.py (see that module for the derivation and the VERIFY
 * notes). States: [roll,pitch,yaw | v_e,v_n,v_u | p_e,p_n,p_u |
 * gyro_bias | accel_bias].
 *
 * Thread-safety: this class is NOT thread-safe; call it from a single
 * thread (the NavigationEngine loop).
 */
class ErrorStateEKF {
    companion object {
        private const val N = 15
        val DEFAULT_NOISE_VARIANCES = floatArrayOf(1e-6f, 2.5e-3f, 1e-10f, 1e-6f)
        val GRAVITY = floatArrayOf(0f, 0f, -9.80665f)
    }

    val errorState = FloatArray(N)
    var P = FloatArray(N * N)
    var rotation = Mat.eye(3)
    var velocity = FloatArray(3)
    var position = FloatArray(3)
    var gyroBias = FloatArray(3)
    var accelBias = FloatArray(3)
    var noiseVariances = DEFAULT_NOISE_VARIANCES.copyOf()
    var nhcSigma = 1.0f

    init {
        // DEFAULT_P0 from the Python port
        val p0 = FloatArray(N * N)
        val diag = floatArrayOf(
            1e-4f, 1e-4f, 1.0f,   // orientation
            4.0f, 4.0f, 1.0f,     // velocity
            9.0f, 9.0f, 9.0f,     // position
            1e-4f, 1e-4f, 1e-4f,  // gyro bias
            1e-2f, 1e-2f, 1e-2f,  // accel bias
        )
        for (i in 0 until N) p0[i * N + i] = diag[i]
        P = p0
    }

    fun predict(gyro: FloatArray, accel: FloatArray, dt: Float) {
        val omega = floatArrayOf(gyro[0] - gyroBias[0], gyro[1] - gyroBias[1], gyro[2] - gyroBias[2])
        val specificForce = floatArrayOf(accel[0] - accelBias[0], accel[1] - accelBias[1], accel[2] - accelBias[2])
        val previousRotation = rotation.copyOf()
        rotation = Mat.matMul(previousRotation, Mat.expRotation(floatArrayOf(omega[0] * dt, omega[1] * dt, omega[2] * dt)))
        val omegaHalf = floatArrayOf(omega[0] * 0.5f * dt, omega[1] * 0.5f * dt, omega[2] * 0.5f * dt)
        val midpointRotation = Mat.matMul(previousRotation, Mat.expRotation(omegaHalf))
        val acceleration = floatArrayOf(0f, 0f, 0f)
        for (i in 0 until 3) {
            acceleration[i] = (midpointRotation[i * 3] * specificForce[0] +
                    midpointRotation[i * 3 + 1] * specificForce[1] +
                    midpointRotation[i * 3 + 2] * specificForce[2]) + GRAVITY[i]
        }
        for (i in 0 until 3) {
            position[i] = position[i] + velocity[i] * dt + 0.5f * acceleration[i] * dt * dt
            velocity[i] = velocity[i] + acceleration[i] * dt
        }
        // state transition matrix
        val phi = Mat.eye(N)
        val omegaSkew = Mat.skew(omega)
        for (i in 0 until 3) for (j in 0 until 3) {
            phi[i * N + j] -= omegaSkew[i * 3 + j] * dt
        }
        for (i in 0 until 3) phi[i * N + (9 + i)] = -dt
        val forceSkew = Mat.skew(specificForce)
        val rotForce = Mat.matMul(rotation, forceSkew)
        for (i in 0 until 3) for (j in 0 until 3) {
            phi[(3 + i) * N + j] -= rotForce[i * 3 + j] * dt
        }
        for (i in 0 until 3) phi[(3 + i) * N + (12 + i)] = -dt
        for (i in 0 until 3) phi[(6 + i) * N + (3 + i)] = dt

        val phiT = Mat.matTranspose(phi, N)
        P = Mat.matMul(Mat.matMul(phi, P, N), phiT, N)
        for (i in 0 until N) P[i * N + i] += processNoise(dt)[i * N + i]
        Mat.symmetrize(P, N)
    }

    private fun processNoise(dt: Float): FloatArray {
        val q = FloatArray(N * N)
        val blocks = intArrayOf(0, 3, 9, 12)
        for (b in 0 until 4) {
            val start = blocks[b]
            for (i in 0 until 3) q[(start + i) * N + (start + i)] = noiseVariances[b] * dt
        }
        return q
    }

    /** Generic measurement update: z = measurement, h = predicted, H, R */
    fun update(measurement: FloatArray, predicted: FloatArray, H: FloatArray, R: FloatArray): FloatArray {
        val z = measurement.copyOf()
        val h = predicted.copyOf()
        if (H.size / N != z.size) throw IllegalArgumentException("H rows must match measurement size")
        val m = z.size
        // S = H P H^T + R
        val HPT = Mat.matMul(H, Mat.matTranspose(P, N), m) // (m x N) x (N x N) -> use generic
        // generic (m x N) @ (N x N)
        val HP = FloatArray(m * N)
        for (i in 0 until m) for (j in 0 until N) {
            var sum = 0f
            for (k in 0 until N) sum += H[i * N + k] * P[k * N + j]
            HP[i * N + j] = sum
        }
        val S = FloatArray(m * m)
        for (i in 0 until m) for (j in 0 until m) {
            var sum = 0f
            for (k in 0 until N) sum += HP[i * N + k] * Mat.matTranspose(H, m)[k * m + j]
            S[i * m + j] = sum + R[i * m + j]
        }
        // K = P H^T S^-1  (N x m)
        val PHT = FloatArray(N * m)
        for (i in 0 until N) for (j in 0 until m) {
            var sum = 0f
            for (k in 0 until N) sum += P[i * N + k] * H[j * N + k]
            PHT[i * m + j] = sum
        }
        // solve S K = P H^T  -> K = solve(S^T, (PHT)^T)^T ; S symmetric so solve directly
        val K = FloatArray(N * m)
        for (col in 0 until m) {
            val b = FloatArray(m) { PHT[it * m + col] }
            val sol = Mat.solve(S, b, m)
            for (row in 0 until m) K[row * N + col] = sol[row]
        }
        // dx = K (z - h)
        val innovation = FloatArray(m) { z[it] - h[it] }
        val dx = Mat.matVec(Mat.matTranspose(K, m), innovation, N)
        rotation = Mat.matMul(Mat.expRotation(floatArrayOf(dx[0], dx[1], dx[2])), rotation)
        for (i in 0 until 3) velocity[i] += dx[3 + i]
        for (i in 0 until 3) position[i] += dx[6 + i]
        for (i in 0 until 3) gyroBias[i] += dx[9 + i]
        for (i in 0 until 3) accelBias[i] += dx[12 + i]
        // Joseph form: P = (I - K H) P (I - K H)^T + K R K^T
        val KH = FloatArray(N * N)
        for (i in 0 until N) for (j in 0 until N) {
            var sum = 0f
            for (k in 0 until m) sum += K[i * m + k] * H[k * N + j]
            KH[i * N + j] = sum
        }
        val IKH = FloatArray(N * N)
        for (i in 0 until N) for (j in 0 until N) IKH[i * N + j] = (if (i == j) 1f else 0f) - KH[i * N + j]
        val IKHt = Mat.matTranspose(IKH, N)
        val p1 = Mat.matMul(Mat.matMul(IKH, P, N), IKHt, N)
        val KRT = Mat.matMul(K, Mat.matTranspose(R, m), m) // (N x m) @ (m x m)
        val KRKt = Mat.matMul(KRT, Mat.matTranspose(K, m), N)
        for (i in 0 until N) for (j in 0 until N) P[i * N + j] = p1[i * N + j] + KRKt[i * N + j]
        Mat.symmetrize(P, N)
        return dx
    }

    fun updateGnss(positionEnu: FloatArray, accuracy: Float): FloatArray {
        val H = Mat.zeros(3, N)
        for (i in 0 until 3) H[i * N + (6 + i)] = 1f
        val R = FloatArray(9)
        for (i in 0 until 3) R[i * 3 + i] = accuracy * accuracy
        return update(positionEnu, position, H, R)
    }

    fun updateGnssVelocity(velocityEnu: FloatArray, velocityAccuracy: Float): FloatArray {
        val H = Mat.zeros(3, N)
        for (i in 0 until 3) H[i * N + (3 + i)] = 1f
        val R = FloatArray(9)
        for (i in 0 until 3) R[i * 3 + i] = velocityAccuracy * velocityAccuracy
        return update(velocityEnu, velocity, H, R)
    }

    fun updateZupt(covariance: FloatArray = floatArrayOf(1f, 1f, 1f)): FloatArray {
        val H = Mat.zeros(3, N)
        for (i in 0 until 3) H[i * N + (3 + i)] = 1f
        val R = FloatArray(9)
        for (i in 0 until 3) R[i * 3 + i] = covariance[i]
        return update(FloatArray(3), velocity, H, R)
    }

    /** NHC: pull ENU velocity toward the matched road's along-heading axis */
    fun updateNhc(roadHeading: Float): FloatArray {
        val lateral = floatArrayOf(-sin(roadHeading), cos(roadHeading), 0f)
        val H = Mat.zeros(1, N)
        for (i in 0 until 3) H[i] = lateral[i]
        val predicted = floatArrayOf(lateral[0] * velocity[0] + lateral[1] * velocity[1] + lateral[2] * velocity[2])
        return update(FloatArray(1), predicted, H, floatArrayOf(nhcSigma * nhcSigma))
    }

    /** Position-domain road constraint with confidence-scaled gain (plan 3.3) */
    fun updateNhcRoad(roadHeading: Float, lateralSigma: Float = 5.0f, confidence: Float = 1.0f): FloatArray {
        val lateral = floatArrayOf(-sin(roadHeading), cos(roadHeading), 0f)
        val H = Mat.zeros(1, N)
        for (i in 0 until 3) H[i * N + (6 + i)] = lateral[i]
        val predicted = floatArrayOf(lateral[0] * position[0] + lateral[1] * position[1] + lateral[2] * position[2])
        val sigma = lateralSigma / kotlin.math.max(confidence, 1e-3f)
        return update(FloatArray(1), predicted, H, floatArrayOf(sigma * sigma))
    }

    fun yaw(): Float = kotlin.math.atan2(rotation[1 * 3 + 0], rotation[0 * 3 + 0])
}