package com.gatinav.android.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class SessionRecorderTest {
    @Test
    fun writesCsvAndManifestForFiveSecondsOfSimulatedSensors() {
        val tempDir = File.createTempFile("session_recorder_test", "")
        tempDir.delete()
        tempDir.mkdirs()

        val recorder = SessionRecorder(tempDir)
        recorder.start()

        for (index in 0 until 50) {
            recorder.append(
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

        val finished = recorder.stop()
        val csvText = finished.csvFile.readText()
        val lines = csvText.split("\n").filter { it.isNotEmpty() }

        assertEquals(51, lines.size)
        assertEquals("t_s,gx,gy,gz,ax,ay,az,mx,my,mz,gnss_lat,gnss_lon,gnss_acc,gnss_t_s,engine_state", lines[0])
        assertTrue(lines[1].startsWith("0.0,0.0,0.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,51.5,-0.1,3.5,0.0,RECORDING"))

        val manifestText = finished.manifestFile.readText()
        assertTrue(manifestText.contains("\"device_model\""))
        assertTrue(manifestText.contains("\"android_version\""))
        assertTrue(manifestText.contains("\"measured_sampling_rates\""))
        assertTrue(manifestText.contains("\"start_time\""))
        assertTrue(manifestText.contains("\"app_version\""))
        assertTrue(manifestText.contains("\"runtime_detected_timestamp_timebase\""))

        tempDir.deleteRecursively()
    }
}
