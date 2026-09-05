package com.gatinav.android.ui

import com.gatinav.android.service.SensorFrame
import com.gatinav.android.service.SensorRingBuffer
import com.gatinav.android.service.SessionRecorder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class SessionViewModelTest {
    @Test
    fun toggleFlipsEngineGnssFeedState() {
        val viewModel = SessionViewModel()
        assertFalse(viewModel.uiState.value.gnssBlackout)

        viewModel.toggleBlackout()
        assertTrue(viewModel.uiState.value.gnssBlackout)

        viewModel.toggleBlackout()
        assertFalse(viewModel.uiState.value.gnssBlackout)
    }

    @Test
    fun statsUpdateFromFakeEngineEmissions() {
        val viewModel = SessionViewModel()

        viewModel.updateStats(sampleHz = 12.5, bufferDrops = 7L, processingLatencyMs = 33L)

        val state = viewModel.uiState.value
        assertEquals(12.5, state.sampleHz, 0.0)
        assertEquals(7L, state.bufferDrops)
        assertEquals(33L, state.processingLatencyMs)
    }

    @Test
    fun loadSessionReadsCsvAndTrajectoryJson() {
        val tempDir = File.createTempFile("session_load_test", "")
        tempDir.delete()
        tempDir.mkdirs()
        val csvFile = File(tempDir, "session.csv")
        csvFile.writeText(
            "t_s,gx,gy,gz,ax,ay,az,mx,my,mz,gnss_lat,gnss_lon,gnss_acc,gnss_t_s,engine_state\n" +
                "0.0,0,0,0,1,2,3,4,5,6,51.5,-0.1,5.0,0.0,RECORDING\n" +
                "0.1,0,0,0,1,2,3,4,5,6,51.51,-0.11,5.0,0.1,RECORDING\n"
        )
        val jsonFile = File(tempDir, "trajectories.json")
        jsonFile.writeText("{\"raw\":[[51.5,-0.1],[51.51,-0.11]],\"inertial\":[[51.52,-0.12]],\"fused\":[[51.53,-0.13]]}")

        val viewModel = SessionViewModel()
        viewModel.loadSession(csvFile, jsonFile)

        val state = viewModel.uiState.value
        assertEquals(2, state.rawGnssPath.size)
        assertEquals(1, state.inertialPath.size)
        assertEquals(1, state.fusedPath.size)
        assertEquals(51.5, state.rawGnssPath[0].first, 1e-9)
    }

    @Test
    fun twoMinuteWalkHasZeroDropsAndMonotonicTimestamps() {
        val tempDir = File.createTempFile("two_min_walk", "")
        tempDir.delete()
        tempDir.mkdirs()
        val recorder = SessionRecorder(tempDir)
        val ring = SensorRingBuffer(2000)
        val timestamps = mutableListOf<Double>()

        recorder.start()
        for (i in 0 until 1200) {
            val seconds = i / 10.0
            timestamps += seconds
            ring.push(i.toLong())
            recorder.append(
                SensorFrame(
                    tSeconds = seconds,
                    gx = 0.0,
                    gy = 0.0,
                    gz = 0.0,
                    ax = 1.0,
                    ay = 0.0,
                    az = 0.0,
                    mx = 0.0,
                    my = 0.0,
                    mz = 1.0,
                    gnssLat = 51.5 + (i / 1000.0),
                    gnssLon = -0.1 - (i / 1000.0),
                    gnssAcc = 5.0,
                    gnssTS = seconds,
                    engineState = "RECORDING"
                )
            )
        }
        val fromRecorder = recorder.stop()
        val csvRows = fromRecorder.csvFile.readLines().drop(1)

        assertEquals(0L, ring.droppedSampleCount())
        assertEquals(1200, csvRows.size)
        assertTrue(timestamps.zipWithNext().all { (a, b) -> b > a })
        tempDir.deleteRecursively()
    }
}
