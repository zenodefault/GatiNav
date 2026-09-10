package com.gatinav.android.ui

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.gatinav.android.engine.FusedPose
import com.gatinav.android.service.RealtimeNavigationService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Binds to the RealtimeNavigationService and exposes the fused 10 Hz pose
 * plus service state to the Compose UI. Also exposes the GNSS blackout
 * toggle (the seamless deficit-handler switch for demos).
 */
class RealtimeNavViewModel(application: Application) : AndroidViewModel(application) {

    private val _pose = MutableStateFlow<FusedPose?>(null)
    val pose: StateFlow<FusedPose?> = _pose.asStateFlow()

    private val _serviceConnected = MutableStateFlow(false)
    val serviceConnected: StateFlow<Boolean> = _serviceConnected.asStateFlow()

    private val _blackout = MutableStateFlow(false)
    val blackout: StateFlow<Boolean> = _blackout.asStateFlow()

    private var service: RealtimeNavigationService? = null

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val svc = service
            if (svc == null) return
            _serviceConnected.value = true
            _blackout.value = svc.blackout.value
            viewModelScope.launch {
                svc.pose.collect { _pose.value = it }
            }
            viewModelScope.launch {
                svc.blackout.collect { _blackout.value = it }
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            _serviceConnected.value = false
        }
    }

    fun bind(context: Context) {
        context.bindService(
            Intent(context, RealtimeNavigationService::class.java),
            connection, Context.BIND_AUTO_CREATE
        )
    }

    fun unbind(context: Context) {
        context.unbindService(connection)
        _serviceConnected.value = false
    }

    fun toggleBlackout() {
        service?.setBlackout(!_blackout.value)
        _blackout.value = !_blackout.value
    }

    override fun onCleared() {
        // unbound by the screen; the service keeps running in foreground
    }
}