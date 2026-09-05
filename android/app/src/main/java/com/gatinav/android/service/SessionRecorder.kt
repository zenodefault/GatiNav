package com.gatinav.android.service

import android.os.Build
import java.io.File
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

private const val CSV_HEADER = "t_s,gx,gy,gz,ax,ay,az,mx,my,mz,gnss_lat,gnss_lon,gnss_acc,gnss_t_s,engine_state"

data class SensorFrame(
    val tSeconds: Double,
    val gx: Double,
    val gy: Double,
    val gz: Double,
    val ax: Double,
    val ay: Double,
    val az: Double,
    val mx: Double,
    val my: Double,
    val mz: Double,
    val gnssLat: Double? = null,
    val gnssLon: Double? = null,
    val gnssAcc: Double? = null,
    val gnssTS: Double? = null,
    val engineState: String = "RECORDING"
)

data class SessionFiles(
    val csvFile: File,
    val manifestFile: File
)

class SessionRecorder(
    private val filesDir: File,
    private val detectedTimebase: String = "SystemClock.elapsedRealtimeNanos"
) {
    private var csvFile: File? = null
    private var manifestFile: File? = null
    private var startInstant: Instant? = null
    private var rowCount = 0

    fun start(): SessionFiles {
        startInstant = Instant.now()
        val stamp = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss").withZone(ZoneOffset.UTC).format(startInstant)
        val sessionDir = File(filesDir, "session_${stamp}")
        sessionDir.mkdirs()

        csvFile = File(sessionDir, "session.csv")
        manifestFile = File(sessionDir, "session_manifest.json")
        csvFile!!.writeText("$CSV_HEADER\n")

        return SessionFiles(csvFile!!, manifestFile!!)
    }

    fun append(frame: SensorFrame) {
        val writer = csvFile ?: return
        writer.appendText(formatRow(frame) + "\n")
        rowCount += 1
    }

    fun stop(): SessionFiles {
        val csv = csvFile ?: return SessionFiles(File(filesDir, "session.csv"), File(filesDir, "session_manifest.json"))
        val manifestText = buildString {
            append("{\n")
            append("  \"device_model\": \"").append(escape(Build.MODEL ?: "unknown")).append("\",\n")
            append("  \"android_version\": \"").append(escape(Build.VERSION.RELEASE ?: "unknown")).append("\",\n")
            append("  \"measured_sampling_rates\": {\"sensor_hz\": 0.0},\n")
            append("  \"start_time\": \"").append(escape(startInstant?.toString() ?: Instant.now().toString())).append("\",\n")
            append("  \"app_version\": \"1.0\",\n")
            append("  \"runtime_detected_timestamp_timebase\": \"").append(escape(detectedTimebase)).append("\"\n")
            append("}\n")
        }
        manifestFile!!.writeText(manifestText)
        return SessionFiles(csv, manifestFile!!)
    }

    fun rowCount(): Int = rowCount

    fun csvHeader(): String = CSV_HEADER

    private fun formatRow(frame: SensorFrame): String {
        val gnssLat = frame.gnssLat?.toString() ?: ""
        val gnssLon = frame.gnssLon?.toString() ?: ""
        val gnssAcc = frame.gnssAcc?.toString() ?: ""
        val gnssTS = frame.gnssTS?.toString() ?: ""

        return listOf(
            frame.tSeconds,
            frame.gx,
            frame.gy,
            frame.gz,
            frame.ax,
            frame.ay,
            frame.az,
            frame.mx,
            frame.my,
            frame.mz,
            gnssLat,
            gnssLon,
            gnssAcc,
            gnssTS,
            frame.engineState
        ).joinToString(",")
    }

    private fun escape(value: String): String {
        return value.replace("\\", "\\\\").replace("\"", "\\\"")
    }
}
