package com.gatinav.android.ui

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.io.File

data class SessionUiState(
    val gnssBlackout: Boolean = false,
    val isSessionRunning: Boolean = false,
    val sampleHz: Double = 0.0,
    val bufferDrops: Long = 0L,
    val processingLatencyMs: Long = 0L,
    val rawGnssPath: List<Pair<Double, Double>> = emptyList(),
    val inertialPath: List<Pair<Double, Double>> = emptyList(),
    val fusedPath: List<Pair<Double, Double>> = emptyList()
)

class SessionViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(SessionUiState())
    val uiState: StateFlow<SessionUiState> = _uiState.asStateFlow()

    fun toggleBlackout() {
        _uiState.update { it.copy(gnssBlackout = !it.gnssBlackout) }
    }

    fun startSession() {
        _uiState.update { it.copy(isSessionRunning = true) }
    }

    fun stopSession() {
        _uiState.update { it.copy(isSessionRunning = false) }
    }

    fun updateStats(sampleHz: Double, bufferDrops: Long, processingLatencyMs: Long) {
        _uiState.update { it.copy(sampleHz = sampleHz, bufferDrops = bufferDrops, processingLatencyMs = processingLatencyMs) }
    }

    fun loadSession(csvFile: File, trajectoryFile: File) {
        val csvPoints = parseSessionCsv(csvFile)
        val trajectoryPoints = parseTrajectoryJson(trajectoryFile)
        _uiState.update {
            it.copy(
                rawGnssPath = csvPoints,
                inertialPath = trajectoryPoints["inertial"] ?: emptyList(),
                fusedPath = trajectoryPoints["fused"] ?: emptyList(),
                isSessionRunning = false
            )
        }
    }

    private fun parseSessionCsv(csvFile: File): List<Pair<Double, Double>> {
        if (!csvFile.exists()) return emptyList()
        val rows = csvFile.readLines().drop(1)
        val points = mutableListOf<Pair<Double, Double>>()
        for (line in rows) {
            if (line.isBlank()) continue
            val cells = line.split(',')
            if (cells.size < 14) continue
            val lat = cells[10].toDoubleOrNull() ?: continue
            val lon = cells[11].toDoubleOrNull() ?: continue
            points.add(lat to lon)
        }
        return points
    }

    private fun parseTrajectoryJson(trajectoryFile: File): Map<String, List<Pair<Double, Double>>> {
        if (!trajectoryFile.exists()) return emptyMap()
        val text = trajectoryFile.readText()
        val output = mutableMapOf<String, List<Pair<Double, Double>>>()
        for (key in listOf("raw", "inertial", "fused")) {
            val keyIndex = text.indexOf("\"$key\"")
            if (keyIndex < 0) continue
            val arrayStart = text.indexOf('[', keyIndex)
            if (arrayStart < 0) continue
            var depth = 0
            var arrayEnd = -1
            for (index in arrayStart until text.length) {
                when (text[index]) {
                    '[' -> depth += 1
                    ']' -> {
                        depth -= 1
                        if (depth == 0) {
                            arrayEnd = index
                            break
                        }
                    }
                }
            }
            if (arrayEnd < 0) continue
            val tokenList = text.substring(arrayStart + 1, arrayEnd)
            val points = mutableListOf<Pair<Double, Double>>()
            var cursor = 0
            while (cursor < tokenList.length) {
                val openIndex = tokenList.indexOf('[', cursor)
                if (openIndex < 0) break
                val commaIndex = tokenList.indexOf(',', openIndex)
                val closeIndex = tokenList.indexOf(']', commaIndex)
                if (commaIndex < 0 || closeIndex < 0) break
                val latText = tokenList.substring(openIndex + 1, commaIndex).trim()
                val lonText = tokenList.substring(commaIndex + 1, closeIndex).trim()
                val lat = latText.toDoubleOrNull()
                if (lat == null) {
                    cursor = closeIndex + 1
                    continue
                }
                val lon = lonText.toDoubleOrNull()
                if (lon == null) {
                    cursor = closeIndex + 1
                    continue
                }
                points.add(lat to lon)
                cursor = closeIndex + 1
            }
            output[key] = points
        }
        return output
    }
}
