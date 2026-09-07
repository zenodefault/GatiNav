package com.gatinav.android.ui

import android.graphics.Color
import android.view.ViewGroup
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Polyline

@Composable
fun GatiNavMapScreen(viewModel: SessionViewModel = remember { SessionViewModel() }) {
    val state by viewModel.uiState.collectAsState()
    val context = LocalContext.current

    Column(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            factory = { context ->
                MapView(context).apply {
                    setTileSource(TileSourceFactory.MAPNIK)
                    setMultiTouchControls(true)
                    controller.setZoom(18.0)
                    controller.setCenter(GeoPoint(51.5074, -0.1278))
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }
            },
            update = { mapView ->
                mapView.overlays.clear()
                val raw = Polyline().apply {
                    color = Color.GRAY
                    setPoints(ArrayList(state.rawGnssPath.map { GeoPoint(it.first, it.second) }))
                }
                val inertial = Polyline().apply {
                    color = Color.RED
                    setPoints(ArrayList(state.inertialPath.map { GeoPoint(it.first, it.second) }))
                }
                val fused = Polyline().apply {
                    color = Color.BLUE
                    setPoints(ArrayList(state.fusedPath.map { GeoPoint(it.first, it.second) }))
                }
                mapView.overlays.addAll(listOf(raw, inertial, fused))
                mapView.invalidate()
            }
        )

        Row(
            modifier = Modifier.fillMaxWidth().padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(onClick = { viewModel.toggleBlackout() }) {
                Text(if (state.gnssBlackout) "Blackout: ON" else "Blackout: OFF")
            }
            Button(onClick = { viewModel.startSession(context.filesDir) }) {
                Text("Start")
            }
            Button(onClick = { viewModel.stopSession() }) {
                Text("Stop")
            }
            Button(onClick = { viewModel.loadLatestSession(context.filesDir) }) {
                Text("Load session")
            }
        }

        Column(Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Text("Live sampling rate: ${state.sampleHz} Hz")
            Text("Buffer drops: ${state.bufferDrops}")
            Text("Processing latency: ${state.processingLatencyMs} ms")
        }
    }
}
