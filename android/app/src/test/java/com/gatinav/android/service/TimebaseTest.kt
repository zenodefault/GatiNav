package com.gatinav.android.service

import org.junit.Assert.assertEquals
import org.junit.Test

class TimebaseTest {
    @Test
    fun selectsElapsedRealtimeWhenSensorTimestampsTrackElapsedRealtime() {
        val sensorTimestamps = listOf(1_050_000_000L, 1_150_000_000L, 1_250_000_000L)
        val elapsedRealtimeNanos = 1_000_000_000L
        val monotonicNanos = 2_000_000_000L

        val selection = selectTimebase(sensorTimestamps, elapsedRealtimeNanos, monotonicNanos)

        assertEquals(TimebaseKind.ELAPSED_REALTIME, selection.timebase)
        assertEquals(50_000_000L, selection.offsetNanos)
        assertEquals(1_000_000_000L, selection.sessionOriginNanos)
        assertEquals(0.1, sessionSecondsFromNanos(1_100_000_000L, selection.sessionOriginNanos), 1e-9)
    }

    @Test
    fun selectsMonotonicClockWhenSensorTimestampsTrackMonoTime() {
        val sensorTimestamps = listOf(9_800_000_000L, 9_900_000_000L, 10_000_000_000L)
        val elapsedRealtimeNanos = 1_000_000_000L
        val monotonicNanos = 9_900_000_000L

        val selection = selectTimebase(sensorTimestamps, elapsedRealtimeNanos, monotonicNanos)

        assertEquals(TimebaseKind.MONOTONIC, selection.timebase)
        assertEquals(-100_000_000L, selection.offsetNanos)
        assertEquals(9_900_000_000L, selection.sessionOriginNanos)
        assertEquals(0.05, sessionSecondsFromNanos(9_950_000_000L, selection.sessionOriginNanos), 1e-9)
    }
}
