// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

enum SpeechInputMethod: String, CaseIterable, Identifiable {
    case apple
    case audioCpp

    var id: String { rawValue }
}

enum SpeechLoadPolicy: String, CaseIterable, Identifiable {
    case onDemand
    case alwaysLoaded

    var id: String { rawValue }
}

enum SpeechPreferences {
    /// The spoken language chosen in the studio, used by dictation when the model cannot
    /// detect the language on its own.
    static var spokenLanguage: String {
        UserDefaults.standard.string(forKey: SettingsKeys.audioLanguage) ?? "auto"
    }

    /// Carries the choices made for the previous speech engine over to audio.cpp. Its model
    /// files cannot be reused, so only the input method and the loading policy survive.
    static func migrateLegacyDefaults(_ defaults: UserDefaults = .standard) {
        if defaults.string(forKey: SettingsKeys.speechInputMethod) == "whisper" {
            defaults.set(SpeechInputMethod.audioCpp.rawValue, forKey: SettingsKeys.speechInputMethod)
        }
        if defaults.object(forKey: SettingsKeys.speechLoadPolicy) == nil,
           let policy = defaults.string(forKey: legacyLoadPolicyKey),
           SpeechLoadPolicy(rawValue: policy) != nil {
            defaults.set(policy, forKey: SettingsKeys.speechLoadPolicy)
        }
        defaults.removeObject(forKey: legacyLoadPolicyKey)
        defaults.removeObject(forKey: legacyModelKey)
        if defaults.string(forKey: SettingsKeys.audioOperation) == legacyTranslateEnglish {
            defaults.set(AudioStudioOperation.translateLocal.rawValue, forKey: SettingsKeys.audioOperation)
            defaults.set("English", forKey: SettingsKeys.audioTargetLanguage)
        }
    }

    private static let legacyModelKey = "whisperModel"
    private static let legacyLoadPolicyKey = "whisperLoadPolicy"
    private static let legacyTranslateEnglish = "translateEnglish"
}
