// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

struct AudioVADCalibration: Equatable, Sendable {
    let threshold: Double
    let minSpeechDurationMS: Int
    let minSilenceDurationMS: Int
    let maxSpeechDurationSeconds: Double
    let speechPadMS: Int

    var requestOptions: [(String, String)] {
        [
            ("threshold", String(threshold)),
            ("min_speech_duration_ms", String(minSpeechDurationMS)),
            ("min_silence_duration_ms", String(minSilenceDurationMS)),
            ("max_speech_duration_sec", String(maxSpeechDurationSeconds)),
            ("speech_pad_ms", String(speechPadMS))
        ]
    }

    static func custom(threshold: Double, minSpeechDurationMS: Double,
                       minSilenceDurationMS: Double, maxSpeechDurationSeconds: Double,
                       speechPadMS: Double) -> AudioVADCalibration {
        AudioVADCalibration(
            threshold: min(0.95, max(0.1, threshold)),
            minSpeechDurationMS: Int(min(2_000, max(100, minSpeechDurationMS)).rounded()),
            minSilenceDurationMS: Int(min(2_000, max(50, minSilenceDurationMS)).rounded()),
            maxSpeechDurationSeconds: min(120, max(10, maxSpeechDurationSeconds)),
            speechPadMS: Int(min(500, max(0, speechPadMS)).rounded())
        )
    }
}
