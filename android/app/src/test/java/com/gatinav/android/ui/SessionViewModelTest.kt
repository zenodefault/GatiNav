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
    fun startRecordStopWritesCsvWithSpecSchemaAndManifest() {
        val tempDir = File.createTempFile("vm_recording_test", "")
        tempDir.delete()
        tempDir.mkdirs()

        val viewModel = SessionViewModel()
        viewModel.startSession(tempDir)
        assertTrue(viewModel.uiState.value.isSessionRunning)

        for (index in 0 until 50) {
            viewModel.recordSample(
                SensorFrame(
                    tSeconds = index.toDouble() / 10.0,
                    gx = 0.0,
                    gy = 0.0,
                    gz = 0.0,
                    ax = 1.0,
                    ay = 2.0,
                    az = 3.0,
                    mx = 4.0,
                    my = 5.0,
                    mz = 6.0,
                    gnssLat = 51.5,
                    gnssLon = -0.1,
                    gnssAcc = 3.5,
                    gnssTS = index.toDouble() / 10.0,
                    engineState = "RECORDING"
                )
            )
        }

        viewModel.stopSession()
        assertFalse(viewModel.uiState.value.isSessionRunning)

        val sessionDir = tempDir.listFiles()!!.first { it.isDirectory && it.name.startsWith("session_") }
        val csvLines = File(sessionDir, "session.csv").readLines()
        assertEquals(51, csvLines.size)
        assertEquals(
            "t_s,gx,gy,gz,ax,ay,az,mx,my,mz,gnss_lat,gnss_lon,gnss_acc,gnss_t_s,engine_state",
            csvLines[0]
        )
        assertTrue(csvLines[1].startsWith("0.0,0.0,0.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,51.5,-0.1,3.5,0.0,RECORDING"))
        assertTrue(csvLines[50].startsWith("4.9,"))

        val manifestText = File(sessionDir, "session_manifest.json").readText()
        assertTrue(manifestText.contains("\"device_model\""))
        assertTrue(manifestText.contains("\"android_version\""))
        assertTrue(manifestText.contains("\"measured_sampling_rates\""))
        assertTrue(manifestText.contains("\"start_time\""))
        assertTrue(manifestText.contains("\"app_version\""))
        assertTrue(manifestText.contains("\"runtime_detected_timestamp_timebase\""))

        tempDir.deleteRecursively()
    }

    @Test
    fun loadLatestSessionReadsRecordedCsvAndPythonTrajectories() {
        val tempDir = File.createTempFile("vm_load_latest", "")
        tempDir.delete()
        tempDir.mkdirs()

        val viewModel = SessionViewModel()
        viewModel.startSession(tempDir)
        for (index in 0 until 3) {
            viewModel.recordSample(
                SensorFrame(
                    tSeconds = index / 10.0,
                    gx = 0.0, gy = 0.0, gz = 0.0,
                    ax = 1.0, ay = 0.0, az = 0.0,
                    mx = 0.0, my = 0.0, mz = 1.0,
                    gnssLat = 51.5 + index * 1e-4,
                    gnssLon = -0.1 - index * 1e-4,
                    gnssAcc = 5.0,
                    gnssTS = index / 10.0,
                    engineState = "RECORDING"
                )
            )
        }
        viewModel.stopSession()

        val sessionDir = tempDir.listFiles()!!.first { it.isDirectory && it.name.startsWith("session_") }
        File(sessionDir, "trajectories.json").writeText(
            "{\"session_id\":\"s1\",\"raw\":[[51.5,-0.1]]," +
                "\"inertial\":[[51.52,-0.12],[51.53,-0.13]]," +
                "\"fused\":[[51.54,-0.14],[51.55,-0.15]]}"
        )

        assertTrue(viewModel.loadLatestSession(tempDir))
        val state = viewModel.uiState.value
        assertEquals(3, state.rawGnssPath.size)   // from the recorded CSV
        assertEquals(2, state.inertialPath.size)  // inertial from the Python JSON
        assertEquals(2, state.fusedPath.size)     // fused from the Python JSON
        assertFalse(state.isSessionRunning)

        tempDir.deleteRecursively()
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
