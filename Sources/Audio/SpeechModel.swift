// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// A speech-to-text checkpoint run by the bundled audio.cpp engine.
struct SpeechModel: Identifiable, Hashable, Sendable {
    enum Tier: Sendable {
        case fast, balanced, recommended, maximum, advanced
    }

    /// The audio.cpp loader that runs the checkpoint (`--family`).
    enum Family: String, Sendable {
        case senseVoice = "sense_asr"
        case parakeet = "parakeet_tdt"
        case qwen3 = "qwen3_asr"
        case crisperWhisper = "crisperwhisper"
    }

    /// A companion checkpoint a family loads to time words, because the model that
    /// transcribes cannot produce them itself.
    struct Aligner: Hashable, Sendable {
        let repository: String
        let revision: String
        /// Path of the GGUF inside the Hugging Face repository.
        let remotePath: String
        let sizeMB: Int

        var fileName: String { (remotePath as NSString).lastPathComponent }

        var downloadURL: String {
            "https://huggingface.co/\(repository)/resolve/\(revision)/\(remotePath)"
        }

        func url(in directory: URL) -> URL {
            directory.appendingPathComponent(fileName)
        }
    }

    let id: String
    let name: String
    let family: Family
    let repository: String
    let revision: String
    /// Path of the GGUF inside the Hugging Face repository.
    let remotePath: String
    let sizeMB: Int
    let tier: Tier
    let detailES: String
    let detailEN: String
    /// Set when the family needs this extra checkpoint to time words.
    var aligner: Aligner?

    var fileName: String { (remotePath as NSString).lastPathComponent }

    var downloadURL: String {
        "https://huggingface.co/\(repository)/resolve/\(revision)/\(remotePath)"
    }

    /// Qwen3-ASR decodes text only; word timings come from the separate force aligner.
    var needsAligner: Bool { aligner != nil }

    /// Whether the checkpoint detects the spoken language itself. CrisperWhisper takes an
    /// explicit Whisper language code and defaults to English when none is sent.
    var detectsLanguage: Bool { family != .crisperWhisper }

    /// The language to use when the stored choice cannot be honored.
    var fallbackLanguage: String { detectsLanguage ? "auto" : "en" }

    /// Qwen3-ForcedAligner 0.6B, shared by both Qwen3-ASR checkpoints.
    static let qwen3Aligner = Aligner(
        repository: "audio-cpp/audio.cpp-gguf", revision: "main",
        remotePath: "Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf",
        sizeMB: 1_078
    )

    static let catalog: [SpeechModel] = [
        .init(id: "sensevoice-small", name: "SenseVoice Small", family: .senseVoice,
              repository: "FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp",
              revision: "5c3fcfe748a8714216bc135476d5863084fddb72",
              remotePath: "sensevoice-small-q8-audiocpp-v1.gguf",
              sizeMB: 243, tier: .fast,
              detailES: "Rápido · chino, inglés, japonés, coreano y más",
              detailEN: "Fast · Chinese, English, Japanese, Korean and more"),
        .init(id: "parakeet-tdt-0.6b-v3", name: "Parakeet TDT 0.6B v3", family: .parakeet,
              repository: "audio-cpp/audio.cpp-gguf", revision: "main",
              remotePath: "Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf",
              sizeMB: 874, tier: .balanced,
              detailES: "El más rápido en europeo · 25 idiomas, detecta el idioma solo",
              detailEN: "Fastest for European languages · 25 languages, detects the language itself"),
        .init(id: "qwen3-asr-0.6b", name: "Qwen3-ASR 0.6B", family: .qwen3,
              repository: "audio-cpp/audio.cpp-gguf", revision: "main",
              remotePath: "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf",
              sizeMB: 1_098, tier: .advanced,
              detailES: "Compacto · 30 idiomas",
              detailEN: "Compact · 30 languages",
              aligner: qwen3Aligner),
        .init(id: "qwen3-asr-1.7b", name: "Qwen3-ASR 1.7B", family: .qwen3,
              repository: "audio-cpp/audio.cpp-gguf", revision: "main",
              remotePath: "Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf",
              sizeMB: 2_359, tier: .recommended,
              detailES: "Máxima precisión · 30 idiomas, incluidos chino, japonés y coreano",
              detailEN: "Maximum accuracy · 30 languages, including Chinese, Japanese and Korean",
              aligner: qwen3Aligner),
        .init(id: "crisperwhisper-2-large", name: "CrisperWhisper 2.0 Large", family: .crisperWhisper,
              repository: "audio-cpp/audio.cpp-gguf", revision: "main",
              remotePath: "CrisperWhisper2.0-GGUF/crisperwhisper-2-large-q8_0.gguf",
              sizeMB: 1_637, tier: .advanced,
              detailES: "Literal y limpio · 99 idiomas, elige el idioma · licencia no comercial",
              detailEN: "Verbatim and clean · 99 languages, pick the language · non-commercial licence")
    ]

    static let recommendedID = "qwen3-asr-1.7b"

    static func model(id: String) -> SpeechModel {
        catalog.first { $0.id == id }
            ?? catalog.first { $0.id == recommendedID }
            ?? catalog[0]
    }

    func url(in directory: URL) -> URL {
        directory.appendingPathComponent(fileName)
    }

    /// The aligner checkpoint next to the model file, when the user downloaded it.
    /// Timings come with the model itself for every other family.
    func installedAlignerURL(beside modelURL: URL) -> URL? {
        guard let aligner else { return nil }
        let url = aligner.url(in: modelURL.deletingLastPathComponent())
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Whether a run of this model can time words: it does so itself, or its aligner is present.
    func providesWordTimings(beside modelURL: URL) -> Bool {
        aligner == nil || installedAlignerURL(beside: modelURL) != nil
    }

    /// The code to ask for: the stored one when the model accepts it, its own default otherwise.
    func effectiveLanguage(_ stored: String) -> String {
        supports(language: stored) ? stored : fallbackLanguage
    }

    /// The language a dictation run sends. Dictation has no picker of its own, so a model that
    /// cannot detect the language uses the app's spoken-language setting.
    func dictationLanguage(_ stored: String) -> String {
        detectsLanguage ? "auto" : effectiveLanguage(stored)
    }

    /// Whether the checkpoint recognizes a language the app offers; `auto` only when it detects.
    func supports(language code: String) -> Bool {
        if code == "auto" { return detectsLanguage }
        switch family {
        case .parakeet: return Self.parakeetLanguages.contains(code)
        case .senseVoice, .qwen3, .crisperWhisper: return requestLanguage(for: code) != nil
        }
    }

    /// The value of the `language` request option, or nil to let the model detect it.
    /// Parakeet always detects the language itself and takes no such option.
    func requestLanguage(for code: String) -> String? {
        guard code != "auto", let language = SpeechLanguage.common.first(where: { $0.id == code }) else {
            return nil
        }
        switch family {
        case .senseVoice, .crisperWhisper: return language.id
        case .qwen3: return language.qwenName
        case .parakeet: return nil
        }
    }

    private static let parakeetLanguages: Set<String> = [
        "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt",
        "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"
    ]
}
