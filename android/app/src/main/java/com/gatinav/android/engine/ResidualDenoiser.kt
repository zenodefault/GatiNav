package com.gatinav.android.engine

import android.content.Context

/**
 * Residual denoiser interface (Blueprint Phase 1.2/1.3 on-device).
 *
 * The exported model (python/ml/export_denoise_onnx.py -> .tflite) maps a
 * (1, 1, W) forward-acceleration window to a scalar residual correction:
 * ``a_denoised = a_measured - residual``. The app ships a TFLite runner
 * when the model asset is present and falls back to a pass-through
 * (residual = 0) otherwise, so the real-time loop always runs.
 */
interface ResidualDenoiser {
    /** residual correction for the current forward-accel window (m/s^2) */
    fun predict(window: FloatArray): Float
}

/** No-op denoiser: the engine runs pure AHRS+EKF without neural correction. */
class PassThroughDenoiser : ResidualDenoiser {
    override fun predict(window: FloatArray): Float = 0f
}

/**
 * TFLite-backed denoiser. Loads "denoise.tflite" from assets on first
 * use; mean/std are baked into the graph by the exporter, so the caller
 * only feeds the raw forward-accel window.
 */
class TfliteDenoiser(context: Context) : ResidualDenoiser {
    private var interpreter: org.tensorflow.lite.Interpreter? = null
    private val output = Array(1) { FloatArray(1) }

    init {
        try {
            val buffer = context.assets.open("denoise.tflite").use { it.readBytes() }
            interpreter = org.tensorflow.lite.Interpreter(java.nio.ByteBuffer.wrap(buffer))
        } catch (e: Exception) {
            interpreter = null
        }
    }

    val isAvailable: Boolean get() = interpreter != null

    override fun predict(window: FloatArray): Float {
        val interp = interpreter ?: return 0f
        val input = Array(1) { Array(1) { window } }
        try {
            interp.run(input, output)
            return output[0][0]
        } catch (e: Exception) {
            return 0f
        }
    }
}

object DenoiserFactory {
    /** Prefer TFLite when the model asset exists; always fall back safely. */
    fun create(context: Context): ResidualDenoiser {
        val tflite = TfliteDenoiser(context)
        return if (tflite.isAvailable) tflite else PassThroughDenoiser()
    }
}