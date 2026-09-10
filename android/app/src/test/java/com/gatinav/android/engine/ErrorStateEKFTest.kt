package com.gatinav.android.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ErrorStateEKFTest {

    @Test
    fun stationaryPredictionStaysFinite() {
        val ekf = ErrorStateEKF()
        val gravity = floatArrayOf(0f, 0f, 9.80665f)
        for (i in 0 until 200) {
            ekf.predict(floatArrayOf(0f, 0f, 0f), gravity, 0.01f)
        }
        assertTrue(ekf.position.all { it.isFinite() })
        assertTrue(ekf.P.all { it.isFinite() })
        // level-levelled stationary body should not accumulate horizontal velocity
        assertTrue(kotlin.math.abs(ekf.velocity[0]) < 1e-3f)
        assertTrue(kotlin.math.abs(ekf.velocity[1]) < 1e-3f)
    }

    @Test
    fun gnssUpdatePullsPositionToFix() {
        val ekf = ErrorStateEKF()
        ekf.predict(floatArrayOf(0f, 0f, 0f), floatArrayOf(0f, 0f, 9.80665f), 0.01f)
        ekf.updateGnss(floatArrayOf(10f, 0f, 0f), 2f)
        assertTrue(ekf.position[0] > 0f)
        assertTrue(ekf.position.all { it.isFinite() })
    }

    @Test
    fun zuptConstrainsVelocity() {
        val ekf = ErrorStateEKF()
        ekf.velocity = floatArrayOf(3f, 0f, 0f)
        ekf.updateZupt(floatArrayOf(1f, 1f, 1f))
        assertTrue(kotlin.math.abs(ekf.velocity[0]) < 3f)
    }

    @Test
    fun nhcConstrainsLateralVelocity() {
        val ekf = ErrorStateEKF()
        // road heading 0 (east); lateral = north
        ekf.velocity = floatArrayOf(10f, 4f, 0f)
        ekf.updateNhc(0f)
        assertTrue(kotlin.math.abs(ekf.velocity[1]) < 4f)
        // along-track velocity preserved
        assertTrue(ekf.velocity[0] > 8f)
    }

    @Test
    fun nhcRoadConstrainsLateralPosition() {
        val ekf = ErrorStateEKF()
        ekf.position = floatArrayOf(0f, 50f, 0f)
        ekf.updateNhcRoad(0f, lateralSigma = 1f, confidence = 1f)
        assertTrue(kotlin.math.abs(ekf.position[1]) < 10f)
    }

    @Test
    fun nhcRoadConfidenceScales() {
        val weak = ErrorStateEKF().apply { position = floatArrayOf(0f, 50f, 0f) }
        weak.updateNhcRoad(0f, lateralSigma = 5f, confidence = 0.1f)
        val strong = ErrorStateEKF().apply { position = floatArrayOf(0f, 50f, 0f) }
        strong.updateNhcRoad(0f, lateralSigma = 5f, confidence = 1f)
        assertTrue(kotlin.math.abs(strong.position[1]) < kotlin.math.abs(weak.position[1]))
    }

    @Test
    fun gnssVelocityTightensBiasObservability() {
        val ekf = ErrorStateEKF()
        ekf.predict(floatArrayOf(0f, 0f, 0f), floatArrayOf(0f, 0f, 9.80665f), 0.01f)
        ekf.updateGnssVelocity(floatArrayOf(5f, 0f, 0f), 2f)
        assertTrue(ekf.velocity.all { it.isFinite() })
        assertEquals(5f, ekf.velocity[0], 1.5f)
    }

    @Test
    fun yawExtraction() {
        val ekf = ErrorStateEKF()
        val y = ekf.yaw()
        assertTrue(y.isFinite())
    }
}