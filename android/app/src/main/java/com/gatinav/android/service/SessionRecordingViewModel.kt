package com.gatinav.android.service

import android.content.Context
import androidx.lifecycle.ViewModel

class SessionRecordingViewModel : ViewModel() {
    private var recorder: SessionRecorder? = null
    private var activeSession: SessionFiles? = null

    fun startSession(context: Context) {
        recorder = SessionRecorder(context.filesDir)
        activeSession = recorder?.start()
    }

    fun recordSample(frame: SensorFrame) {
        recorder?.append(frame)
    }

    fun stopSession(): SessionFiles? {
        val session = recorder?.stop()
        recorder = null
        activeSession = session
        return session
    }
}
