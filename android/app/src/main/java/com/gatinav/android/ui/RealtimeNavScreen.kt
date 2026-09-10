package com.gatinav.android.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.ViewGroup
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.gatinav.android.engine.FusedPose
import com.gatinav.android.engine.FusionMode
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

/**
 * Real-time navigation screen (Blueprint plan 4.3): an OSM map with a
 * smooth animated vehicle icon reflecting the fused GNSS/INS pose, plus
 * ISO 15008-style high-contrast readouts (mode, speed, GNSS health) and
 * the blackout toggle for the seamless deficit handler.
 */
@Composable
fun RealtimeNavScreen(
    viewModel: RealtimeNavViewModel = remember { RealtimeNavViewModel(LocalContext.current.applicationContext as android.app.Application) }
) {
    val pose by viewModel.pose.collectAsState()
    val connected by viewModel.serviceConnected.collectAsState()
    val blackout by viewModel.blackout.collectAsState()
    val context = LocalContext.current

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                MapView(ctx).apply {
                    setTileSource(TileSourceFactory.MAPNIK)
                    setMultiTouchControls(true)
                    controller.setZoom(18.0)
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    overlays.add(Marker(this).apply {
                        setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
                        icon = vehicleIcon(36f, Color.rgb(33, 150, 243))
                        id = "vehicle"
                    })
                }
            },
            update = { mapView ->
                val marker = mapView.overlays.filterIsInstance<Marker>()
                    .firstOrNull { it.id == "vehicle" }
                val vehicle = pose
                if (vehicle != null && marker != null) {
                    marker.position = GeoPoint(vehicle.latDeg, vehicle.lonDeg)
                    marker.setRotation(Math.toDegrees(vehicle.headingRad.toDouble()).toFloat())
                    val poly = mapView.overlays.filterIsInstance<Polyline>().firstOrNull()
                    if (poly != null) {
                        val points = ArrayList<GeoPoint>(poly.points)
                        points.add(GeoPoint(vehicle.latDeg, vehicle.lonDeg))
                        poly.setPoints(points.takeLast(500))
                    }
                    mapView.controller.animateTo(GeoPoint(vehicle.latDeg, vehicle.lonDeg))
                    mapView.invalidate()
                }
            }
        )

        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(12.dp)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            StatusPill("Engine: ${if (connected) "LIVE" else "offline"}",
                if (connected) Color.rgb(46, 125, 50) else Color.rgb(158, 158, 158))
            val mode = pose?.mode ?: FusionMode.FUSED
            StatusPill("Mode: ${mode.name} ${if (blackout) "(blackout)" else ""}",
                modeColor(mode))
            StatusPill("Speed: ${"%.1f".format(pose?.speedMps ?: 0.0)} m/s",
                Color.rgb(33, 33, 33))
            StatusPill("GNSS: ${if (pose?.gnssValid == true) "OK" else "lost"}",
                if (pose?.gnssValid == true) Color.rgb(46, 125, 50) else Color.rgb(211, 47, 47))
        }

        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(onClick = { viewModel.toggleBlackout() }) {
                Text(if (blackout) "Blackout: ON" else "Blackout: OFF")
            }
        }
    }
}

@Composable
private fun StatusPill(text: String, color: Int) {
    Text(
        text = text,
        color = Color.WHITE,
        style = MaterialTheme.typography.titleMedium,
        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
    )
}

private fun modeColor(mode: FusionMode): Int = when (mode) {
    FusionMode.FUSED -> Color.rgb(46, 125, 50)
    FusionMode.INERTIAL -> Color.rgb(230, 81, 0)
    FusionMode.REACQUIRING -> Color.rgb(255, 152, 0)
}

private fun vehicleIcon(sizePx: Float, color: Int): Bitmap {
    val bmp = Bitmap.createBitmap(sizePx.toInt(), sizePx.toInt(), Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bmp)
    val cx = sizePx / 2
    val cy = sizePx / 2
    // direction arrow: a triangle pointing up, rotated by the marker heading
    val arrow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color = color
        style = Paint.Style.FILL
    }
    val outline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = sizePx * 0.06f
    }
    val r = sizePx * 0.34f
    val tip = android.graphics.Path().apply {
        moveTo(cx, cy - r)
        lineTo(cx - r * 0.7f, cy + r * 0.7f)
        lineTo(cx, cy + r * 0.35f)
        lineTo(cx + r * 0.7f, cy + r * 0.7f)
        close()
    }
    canvas.drawPath(tip, arrow)
    canvas.drawPath(tip, outline)
    return bmp
}