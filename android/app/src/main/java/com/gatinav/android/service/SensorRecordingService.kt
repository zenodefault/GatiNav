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
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

class SensorRecordingService : Service(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private lateinit var notificationManager: NotificationManager
    private lateinit var powerManager: PowerManager
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    private val ringBuffer = SensorRingBuffer(128)
    private var wakeLock: PowerManager.WakeLock? = null
    private var sessionOriginNanos: Long = 0L
    private var lastSessionSeconds = Double.NEGATIVE_INFINITY

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SensorManager::class.java)
        notificationManager = getSystemService(NotificationManager::class.java)
        powerManager = getSystemService(PowerManager::class.java)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, buildNotification())
        acquireWakeLock()
        registerSensors()
        startLocationUpdates()
        return START_STICKY
    }

    override fun onDestroy() {
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(locationCallback)
        wakeLock?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onSensorChanged(event: SensorEvent) {
        if (sessionOriginNanos == 0L) {
            val elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos()
            val monotonicNanos = System.nanoTime()
            val selection = selectTimebase(listOf(event.timestamp), elapsedRealtimeNanos, monotonicNanos)
            sessionOriginNanos = selection.sessionOriginNanos
        }

        val sessionSeconds = sessionSecondsFromNanos(event.timestamp, sessionOriginNanos)
        if (sessionSeconds < lastSessionSeconds) {
            return
        }

        lastSessionSeconds = sessionSeconds
        ringBuffer.push(event.timestamp)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun acquireWakeLock() {
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "GatiNav:SensorRecording")
        wakeLock?.acquire()
    }

    private fun registerSensors() {
        val sensors = listOf(
            sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
            sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
            sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        ).filterNotNull()

        sensors.forEach { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        }
    }

    private fun startLocationUpdates() {
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L).build()
        fusedLocationClient.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
    }

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.lastLocation ?: return
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }

        val channel = NotificationChannel(
            CHANNEL_ID,
            "GatiNav Sensor Recording",
            NotificationManager.IMPORTANCE_LOW
        )
        notificationManager.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("GatiNav")
            .setContentText("Recording sensor stream")
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "gatinav_sensor_recording"
        private const val NOTIFICATION_ID = 42
    }
}
