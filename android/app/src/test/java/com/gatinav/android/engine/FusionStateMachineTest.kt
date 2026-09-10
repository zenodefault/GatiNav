package com.gatinav.android.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FusionStateMachineTest {

    @Test
    fun unhealthyFixDropsFusedToInertial() {
        val machine = FusionStateMachine()
        machine.advance(1f)
        assertEquals(FusionMode.FUSED, machine.mode)
        val state = machine.reportHealth(2f, pdop = 12f, satellites = 3, snr = 18f)
        assertEquals(FusionMode.INERTIAL, state)
    }

    @Test
    fun healthyFixInInertialEntersReacquiring() {
        val machine = FusionStateMachine()
        machine.advance(5f) // lost > 2 s -> INERTIAL
        assertEquals(FusionMode.INERTIAL, machine.mode)
        val state = machine.reportHealth(6f, pdop = 1.5f, satellites = 9, snr = 40f, accuracyM = 3f)
        assertEquals(FusionMode.REACQUIRING, state)
    }

    @Test
    fun resolutionBackToFused() {
        val machine = FusionStateMachine()
        machine.advance(5f)
        machine.reportHealth(6f, pdop = 1.5f, satellites = 9, snr = 40f, accuracyM = 3f)
        val state = machine.resolveConsistency(floatArrayOf(0f, 0f, 0f), Mat.eye(3))
        assertEquals(FusionMode.FUSED, state)
    }

    @Test
    fun partialMetricsAreHealthy() {
        val machine = FusionStateMachine()
        machine.advance(5f)
        val state = machine.reportHealth(6f, accuracyM = 4f)
        assertEquals(FusionMode.REACQUIRING, state)
    }

    @Test
    fun transitionsInflateCovariance() {
        val machine = FusionStateMachine()
        val before = machine.covariance
        machine.advance(5f) // FUSED -> INERTIAL
        val afterInertial = machine.covariance
        machine.reportHealth(6f, pdop = 1.5f, satellites = 9, snr = 40f, accuracyM = 3f)
        val afterReacquiring = machine.covariance
        assertTrue(afterInertial[0] > before[0])
        assertTrue(afterReacquiring[0] > afterInertial[0])
    }

    @Test
    fun timeMustIncrease() {
        val machine = FusionStateMachine()
        machine.advance(5f)
        var threw = false
        try {
            machine.advance(5f)
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertTrue(threw)
    }
}