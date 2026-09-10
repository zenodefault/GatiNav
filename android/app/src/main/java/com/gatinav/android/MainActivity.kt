package com.gatinav.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.core.content.ContextCompat
import com.gatinav.android.service.RealtimeNavigationService
import com.gatinav.android.ui.RealtimeNavScreen

class MainActivity : ComponentActivity() {

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        startRealtimeNavigation()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            GatiNavTheme {
                RealtimeNavScreen()
            }
        }
        if (hasRequiredPermissions()) {
            startRealtimeNavigation()
        } else {
            permissionLauncher.launch(requiredPermissions())
        }
    }

    private fun hasRequiredPermissions(): Boolean =
        requiredPermissions().all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

    private fun requiredPermissions(): Array<String> {
        val permissions = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.FOREGROUND_SERVICE,
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            permissions.add(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions.add(Manifest.permission.FOREGROUND_SERVICE_LOCATION)
        }
        return permissions.toTypedArray()
    }

    private fun startRealtimeNavigation() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(Intent(this, RealtimeNavigationService::class.java))
        } else {
            startService(Intent(this, RealtimeNavigationService::class.java))
        }
    }
}

@Composable
private fun GatiNavTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = MaterialTheme.colorScheme,
        typography = MaterialTheme.typography,
        content = content
    )
}