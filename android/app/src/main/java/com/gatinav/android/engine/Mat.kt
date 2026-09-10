package com.gatinav.android.engine

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Minimal matrix/vector helpers for the 15-state error-state EKF port
 * (mirrors the Python engine in python/ekf/ekf.py). All matrices are
 * FloatArray in row-major order; vectors are FloatArray.
 */
object Mat {
    fun eye(n: Int): FloatArray {
        val m = FloatArray(n * n)
        for (i in 0 until n) m[i * n + i] = 1f
        return m
    }

    fun zeros(rows: Int, cols: Int): FloatArray = FloatArray(rows * cols)

    fun vec3(vararg v: Float): FloatArray = floatArrayOf(v[0], v[1], v[2])

    /** skew-symmetric matrix of a 3-vector */
    fun skew(v: FloatArray): FloatArray {
        val x = v[0]; val y = v[1]; val z = v[2]
        return floatArrayOf(
            0f, -z, y,
            z, 0f, -x,
            -y, x, 0f,
        )
    }

    /** rotation matrix from a rotation vector (Hamilton/Rodrigues) */
    fun expRotation(vector: FloatArray): FloatArray {
        val angle = norm(vector)
        if (angle < 1e-12f) {
            val r = eye(3)
            val s = skew(vector)
            for (i in 0 until 9) r[i] += s[i]
            return r
        }
        val axis = floatArrayOf(vector[0] / angle, vector[1] / angle, vector[2] / angle)
        val c = cos(angle); val s = sin(angle)
        val outer = FloatArray(9)
        for (i in 0 until 3) for (j in 0 until 3) outer[i * 3 + j] = axis[i] * axis[j]
        val cross = skew(axis)
        val result = FloatArray(9)
        for (i in 0 until 9) {
            result[i] = c * (if (i % 4 == 0) 1f else 0f) + (1 - c) * outer[i] + s * cross[i]
        }
        return result
    }

    fun matMul(a: FloatArray, b: FloatArray, n: Int): FloatArray {
        val out = FloatArray(n * n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                var sum = 0f
                for (k in 0 until n) sum += a[i * n + k] * b[k * n + j]
                out[i * n + j] = sum
            }
        }
        return out
    }

    fun matVec(a: FloatArray, v: FloatArray, n: Int): FloatArray {
        val out = FloatArray(n)
        for (i in 0 until n) {
            var sum = 0f
            for (k in 0 until n) sum += a[i * n + k] * v[k]
            out[i] = sum
        }
        return out
    }

    fun matTranspose(a: FloatArray, n: Int): FloatArray {
        val out = FloatArray(n * n)
        for (i in 0 until n) for (j in 0 until n) out[j * n + i] = a[i * n + j]
        return out
    }

    /** solve A x = b for square A via Gaussian elimination with partial pivoting */
    fun solve(a: FloatArray, b: FloatArray, n: Int): FloatArray {
        val m = a.copyOf()
        val x = b.copyOf()
        for (col in 0 until n) {
            var pivot = col
            var max = kotlin.math.abs(m[col * n + col])
            for (row in col + 1 until n) {
                val v = kotlin.math.abs(m[row * n + col])
                if (v > max) { max = v; pivot = row }
            }
            if (pivot != col) {
                for (k in 0 until n) {
                    val tmp = m[col * n + k]; m[col * n + k] = m[pivot * n + k]; m[pivot * n + k] = tmp
                }
                val tmp = x[col]; x[col] = x[pivot]; x[pivot] = tmp
            }
            val d = m[col * n + col]
            if (kotlin.math.abs(d) < 1e-20f) throw IllegalArgumentException("singular matrix in solve")
            for (row in col + 1 until n) {
                val factor = m[row * n + col] / d
                for (k in col until n) m[row * n + k] -= factor * m[col * n + k]
                x[row] -= factor * x[col]
            }
        }
        val result = FloatArray(n)
        for (i in n - 1 downTo 0) {
            var sum = x[i]
            for (k in i + 1 until n) sum -= m[i * n + k] * result[k]
            result[i] = sum / m[i * n + i]
        }
        return result
    }

    fun norm(v: FloatArray): Float = sqrt(v.sumOf { it * it })

    /** symmetric half: m = (m + m^T) / 2 in place (n x n) */
    fun symmetrize(m: FloatArray, n: Int) {
        for (i in 0 until n) for (j in i + 1 until n) {
            val v = (m[i * n + j] + m[j * n + i]) / 2f
            m[i * n + j] = v
            m[j * n + i] = v
        }
    }
}