package com.gatinav.android.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import com.gatinav.android.engine.EngineSample
import com.gatinav.android.engine.FusedPose
import com.gatinav.android.engine.GnssFix
import com.gatinav.android.engine.NavigationEngine
import com.gatinav.android.engine.DenoiserFactory
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Real-time navigation foreground service (Blueprint Phase B / plan 4.3).
 *
 * Subscribes to the phone's accelerometer/gyroscope/magnetometer at
 * maximum rate plus GNSS at 1 Hz, feeds the on-device NavigationEngine
 * (AHRS + residual denoiser + 15-state EKF + fusion state machine) and
 * publishes the fused 10 Hz pose to a StateFlow the UI collects.
 */
class RealtimeNavigationService : Service(), SensorEventListener {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var engineJob: Job? = null
    private lateinit var sensorManager: SensorManager
    private lateinit var notificationManager: NotificationManager
    private lateinit var powerManager: PowerManager
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var engine: NavigationEngine? = null

    private var sessionOriginNanos = 0L
    private var lastSessionSeconds = Double.NEGATIVE_INFINITY
    private var accel = FloatArray(3)
    private var gyro = FloatArray(3)
    private var mag = FloatArray(3)
    private var hasAccel = false
    private var hasGyro = false
    private var hasMag = false

    private val _pose = MutableStateFlow<FusedPose?>(null)
    val pose: StateFlow<FusedPose?> = _pose.asStateFlow()

    private val _blackout = MutableStateFlow(false)
    val blackout: StateFlow<Boolean> = _blackout.asStateFlow()

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SensorManager::class.java)
        notificationManager = getSystemService(NotificationManager::class.java)
        powerManager = getSystemService(PowerManager::class.java)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        createNotificationChannel()
        engine = NavigationEngine(DenoiserFactory.create(this))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, buildNotification())
        acquireWakeLock()
        registerSensors()
        startLocationUpdates()
        startEngineLoop()
        return START_STICKY
    }

    override fun onDestroy() {
        engineJob?.cancel()
        scope.cancel()
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(locationCallback)
        wakeLock?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    /** Toggle the GNSS blackout for tunnel/outage simulation (plan seamless handler). */
    fun setBlackout(active: Boolean) {
        _blackout.value = active
        engine?.setBlackout(active)
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accel = event.values.copyOf(3)
                hasAccel = true
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyro = event.values.copyOf(3)
                hasGyro = true
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                val norm = kotlin.math.sqrt(event.values[0] * event.values[0] +
                        event.values[1] * event.values[1] + event.values[2] * event.values[2])
                mag = if (norm > 1e-6f) floatArrayOf(
                    event.values[0] / norm, event.values[1] / norm, event.values[2] / norm
                ) else event.values.copyOf(3)
                hasMag = true
            }
        }
        if (!(hasAccel && hasGyro && hasMag)) return

        if (sessionOriginNanos == 0L) {
            val elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos()
            val monotonicNanos = System.nanoTime()
            val selection = selectTimebase(listOf(event.timestamp), elapsedRealtimeNanos, monotonicNanos)
            sessionOriginNanos = selection.sessionOriginNanos
        }
        val sessionSeconds = sessionSecondsFromNanos(event.timestamp, sessionOriginNanos)
        if (sessionSeconds < lastSessionSeconds) return
        lastSessionSeconds = sessionSeconds

        engine?.queueSample(EngineSample(
            t = sessionSeconds.toFloat(),
            gyro = gyro,
            accel = accel,
            mag = mag,
        ))
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private var wakeLock: PowerManager.WakeLock? = null

    private fun acquireWakeLock() {
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "GatiNav:Realtime")
        wakeLock?.acquire()
    }

    private fun registerSensors() {
        listOf(
            sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
            sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
            sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD),
        ).filterNotNull().forEach { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        }
    }

    private fun startLocationUpdates() {
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L).build()
        fusedLocationClient.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
    }

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val location = result.lastLocation ?: return
            if (sessionOriginNanos == 0L) return
            val t = sessionSecondsFromNanos(SystemClock.elapsedRealtimeNanos(), sessionOriginNanos).toFloat()
            engine?.queueFix(GnssFix(
                t = t,
                latDeg = location.latitude,
                lonDeg = location.longitude,
                accuracyM = location.accuracy,
            ))
        }
    }

    private fun startEngineLoop() {
        engineJob = scope.launch {
            while (isActive) {
                engine?.tick()
                _pose.value = engine?.pose
                delay(100L) // 10 Hz publish
            }
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            CHANNEL_ID, "GatiNav Real-time Navigation", NotificationManager.IMPORTANCE_LOW
        )
        notificationManager.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("GatiNav")
            .setContentText("Real-time GNSS+INS fusion active")
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "gatinav_realtime"
        private const val NOTIFICATION_ID = 43
    }
}