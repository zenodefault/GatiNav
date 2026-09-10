package com.gatinav.android.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class AhrsTest {

    private fun rotX(a: Float): FloatArray = floatArrayOf(
        1f, 0f, 0f,
        0f, cos(a), -sin(a),
        0f, sin(a), cos(a),
    )

    private fun rotY(a: Float): FloatArray = floatArrayOf(
        cos(a), 0f, sin(a),
        0f, 1f, 0f,
        -sin(a), 0f, cos(a),
    )

    private fun rotZ(a: Float): FloatArray = floatArrayOf(
        cos(a), -sin(a), 0f,
        sin(a), cos(a), 0f,
        0f, 0f, 1f,
    )

    private fun rotToBody(vectors: Array<FloatArray>, r: FloatArray): Array<FloatArray> {
        val rt = Mat.matTranspose(r, 3)
        return Array(vectors.size) { i -> Mat.matVec(rt, vectors[i], 3) }
    }

    @Test
    fun initStaticRecoversKnownRotation() {
        val roll = Math.toRadians(20.0).toFloat()
        val pitch = Math.toRadians(10.0).toFloat()
        val yaw = Math.toRadians(30.0).toFloat()
        val rotation = Mat.matMul(rotZ(yaw), Mat.matMul(rotY(pitch), rotX(roll)))

        val levelAccel = Array(50) { floatArrayOf(0f, 0f, 9.80665f) }
        val levelMag = Array(50) { floatArrayOf(0f, 1f, 0f) }
        val accel = rotToBody(levelAccel, rotation)
        val mag = rotToBody(levelMag, rotation)

        val ahrs = Ahrs().initStatic(accel, mag)
        val recovered = ahrs.rotation
        val residual = Mat.matMul(recovered, Mat.matTranspose(rotation, 3))
        // recovered @ rotation^T ~= identity
        val maxDiff = (0 until 9).maxOf { abs(residual[it] - (if (it % 4 == 0) 1f else 0f)) }
        assertTrue("attitude recovery error $maxDiff", maxDiff < 2e-2f)
    }

    @Test
    fun stationaryUpdateStaysStable() {
        val accel = Array(20) { floatArrayOf(0f, 0f, 9.80665f) }
        val mag = Array(20) { floatArrayOf(0f, 1f, 0f) }
        val ahrs = Ahrs().initStatic(accel, mag)
        val r0 = ahrs.rotation
        val rng = kotlin.random.Random(1)
        for (i in 0 until 100) {
            val gyro = floatArrayOf(rng.nextFloat() * 2e-3f - 1e-3f, rng.nextFloat() * 2e-3f - 1e-3f, rng.nextFloat() * 2e-3f - 1e-3f)
            val a = floatArrayOf(rng.nextFloat() * 0.04f - 0.02f, rng.nextFloat() * 0.04f - 0.02f, 9.80665f + rng.nextFloat() * 0.04f - 0.02f)
            ahrs.update(gyro, a, floatArrayOf(0f, 1f, 0f), 0.01f)
        }
        val drift = (0 until 9).maxOf { abs(ahrs.rotation[it] - r0[it]) }
        assertTrue("attitude drift $drift", drift < 0.05f)
    }

    @Test
    fun forwardAccelHasNoGravityLeak() {
        val accel = Array(20) { floatArrayOf(0f, 0f, 9.80665f) }
        val mag = Array(20) { floatArrayOf(0f, 1f, 0f) }
        val ahrs = Ahrs().initStatic(accel, mag)
        assertEquals(0f, ahrs.forwardAccel(floatArrayOf(0f, 0f, 9.80665f)), 1e-6f)
    }

    @Test
    fun updateRequiresInit() {
        val ahrs = Ahrs()
        var threw = false
        try {
            ahrs.update(floatArrayOf(0f, 0f, 0f), floatArrayOf(0f, 0f, 9.80665f))
        } catch (e: IllegalStateException) {
            threw = true
        }
        assertTrue(threw)
    }

    @Test
    fun initStaticRejectsVerticalField() {
        val accel = Array(10) { floatArrayOf(0f, 0f, 9.80665f) }
        val mag = Array(10) { floatArrayOf(0f, 0f, 1f) }
        var threw = false
        try {
            Ahrs().initStatic(accel, mag)
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertTrue(threw)
    }
}