package com.gatinav.android.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NavigationEngineTest {

    @Test
    fun stationarySessionProducesFinitePose() {
        val engine = NavigationEngine(PassThroughDenoiser())
        // 5 s static init + 1 s of motion-free samples
        for (i in 0 until 600) {
            engine.queueSample(EngineSample(
                t = i * 0.01f,
                gyro = floatArrayOf(0f, 0f, 0f),
                accel = floatArrayOf(0f, 0f, 9.80665f),
                mag = floatArrayOf(0f, 1f, 0f),
            ))
        }
        var published = false
        for (i in 0 until 70) {
            if (engine.tick()) published = true
        }
        assertTrue("engine never published a pose", published)
        val pose = engine.pose
        assertNotNull(pose)
        assertTrue(pose!!.latDeg.isFinite())
        assertTrue(pose.lonDeg.isFinite())
        assertTrue(pose.headingRad.isFinite())
    }

    @Test
    fun gnssFixFeedsEnuOriginAndPose() {
        val engine = NavigationEngine(PassThroughDenoiser())
        for (i in 0 until 600) {
            engine.queueSample(EngineSample(
                t = i * 0.01f,
                gyro = floatArrayOf(0f, 0f, 0f),
                accel = floatArrayOf(0f, 0f, 9.80665f),
                mag = floatArrayOf(0f, 1f, 0f),
            ))
        }
        engine.queueFix(GnssFix(t = 6.0f, latDeg = 51.5074, lonDeg = -0.1278, accuracyM = 3f))
        for (i in 0 until 70) engine.tick()
        val pose = engine.pose
        assertNotNull(pose)
        // ENU origin = first fix, so the pose should be near the fix
        assertEquals(51.5074, pose!!.latDeg, 1e-3)
        assertEquals(-0.1278, pose.lonDeg, 1e-3)
    }

    @Test
    fun blackoutSwitchesModeToInertial() {
        val engine = NavigationEngine(PassThroughDenoiser())
        for (i in 0 until 600) {
            engine.queueSample(EngineSample(
                t = i * 0.01f,
                gyro = floatArrayOf(0f, 0f, 0f),
                accel = floatArrayOf(0f, 0f, 9.80665f),
                mag = floatArrayOf(0f, 1f, 0f),
            ))
        }
        engine.queueFix(GnssFix(t = 6.0f, latDeg = 51.5074, lonDeg = -0.1278, accuracyM = 3f))
        for (i in 0 until 70) engine.tick()
        assertEquals(FusionMode.FUSED, engine.pose?.mode)
        engine.setBlackout(true)
        for (i in 0 until 70) engine.tick()
        // no new fixes -> advance() drives toward INERTIAL after 2 s
        assertEquals(FusionMode.INERTIAL, engine.pose?.mode)
    }

    @Test
    fun denoiserFallbackIsPassThrough() {
        val denoiser = PassThroughDenoiser()
        assertEquals(0f, denoiser.predict(FloatArray(50)))
    }
}