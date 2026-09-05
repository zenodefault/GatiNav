package com.gatinav.android.service

import kotlin.math.abs
import kotlin.math.pow

enum class TimebaseKind {
    ELAPSED_REALTIME,
    MONOTONIC
}

data class TimebaseSelection(
    val timebase: TimebaseKind,
    val offsetNanos: Long,
    val sessionOriginNanos: Long
)

fun selectTimebase(
    sensorTimestampsNanos: List<Long>,
    elapsedRealtimeNanos: Long,
    monotonicNanos: Long
): TimebaseSelection {
    if (sensorTimestampsNanos.isEmpty()) {
        return TimebaseSelection(TimebaseKind.ELAPSED_REALTIME, 0L, 0L)
    }

    val elapsedOffsets = sensorTimestampsNanos.map { it - elapsedRealtimeNanos }
    val monotonicOffsets = sensorTimestampsNanos.map { it - monotonicNanos }
    val elapsedResidual = residual(elapsedOffsets)
    val monotonicResidual = residual(monotonicOffsets)
    val elapsedMagnitude = abs(elapsedOffsets.first().toDouble())
    val monotonicMagnitude = abs(monotonicOffsets.first().toDouble())

    return if (
        monotonicResidual < elapsedResidual ||
        (monotonicResidual == elapsedResidual && monotonicMagnitude < elapsedMagnitude)
    ) {
        val offsetNanos = monotonicOffsets.first()
        TimebaseSelection(
            timebase = TimebaseKind.MONOTONIC,
            offsetNanos = offsetNanos,
            sessionOriginNanos = sensorTimestampsNanos.first() - offsetNanos
        )
    } else {
        val offsetNanos = elapsedOffsets.first()
        TimebaseSelection(
            timebase = TimebaseKind.ELAPSED_REALTIME,
            offsetNanos = offsetNanos,
            sessionOriginNanos = sensorTimestampsNanos.first() - offsetNanos
        )
    }
}

fun sessionSecondsFromNanos(timestampNanos: Long, sessionOriginNanos: Long): Double {
    return (timestampNanos - sessionOriginNanos).toDouble() / 1_000_000_000.0
}

private fun residual(offsets: List<Long>): Double {
    if (offsets.isEmpty()) {
        return Double.POSITIVE_INFINITY
    }

    val mean = offsets.map(Long::toDouble).average()
    return offsets.fold(0.0) { total, value ->
        total + ((value.toDouble() - mean).pow(2.0))
    } / offsets.size
}
