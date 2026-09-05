package com.gatinav.android.service

import org.junit.Assert.assertEquals
import org.junit.Test

class RingBufferTest {
    @Test
    fun dropsOldestSampleWhenBufferIsFull() {
        val buffer = SensorRingBuffer(3)

        buffer.push(10L)
        buffer.push(20L)
        buffer.push(30L)
        buffer.push(40L)

        assertEquals(listOf(20L, 30L, 40L), buffer.snapshot())
        assertEquals(1L, buffer.droppedSampleCount())
        assertEquals(3, buffer.size())
    }
}
