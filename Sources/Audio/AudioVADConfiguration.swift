// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

struct AudioVADConfiguration: Equatable, Sendable {
    let mode: AudioVADMode
    let calibration: AudioVADCalibration?

    var isEnabled: Bool { mode.isEnabled }

    /// Silero leaves speech unbounded by default; each span becomes one ASR request, so the
    /// standard mode caps it like the calibrated profiles do.
    static let standardMaxSpeechSeconds = 30.0

    /// `--request-option` pairs for the Silero VAD task, in a stable order.
    var requestOptions: [(String, String)] {
        guard isEnabled else { return [] }
        guard mode == .calibrated, let calibration else {
            return [("max_speech_duration_sec", String(Self.standardMaxSpeechSeconds))]
        }
        return calibration.requestOptions
    }
}
