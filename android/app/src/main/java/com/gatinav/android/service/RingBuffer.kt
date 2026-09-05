package com.gatinav.android.service

class SensorRingBuffer(private val capacity: Int) {
    private val values = LongArray(capacity)
    private var size = 0
    private var head = 0
    private var droppedSamples = 0L

    fun push(value: Long) {
        if (size == capacity) {
            values[head] = value
            head = (head + 1) % capacity
            droppedSamples += 1L
            return
        }

        values[(head + size) % capacity] = value
        size += 1
    }

    fun snapshot(): List<Long> {
        val snapshot = mutableListOf<Long>()
        for (index in 0 until size) {
            snapshot.add(values[(head + index) % capacity])
        }
        return snapshot
    }

    fun droppedSampleCount(): Long = droppedSamples

    fun size(): Int = size
}
