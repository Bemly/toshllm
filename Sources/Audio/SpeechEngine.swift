// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Locations and command lines of the bundled audio.cpp speech engine.
enum SpeechEngine {
    /// Directory holding the binaries and the Silero VAD assets they resolve relative to the
    /// working directory: the app's bin-audio, or the engine build when running from the tree.
    static var resourceDirectory: URL {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("bin-audio", isDirectory: true),
           FileManager.default.isExecutableFile(atPath: bundled.appendingPathComponent("audiocpp_cli").path) {
            return bundled
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("vendor/audio.cpp/build-static/bin", isDirectory: true)
    }

    static var cliBinary: String { resourceDirectory.appendingPathComponent("audiocpp_cli").path }
    static var serverBinary: String { resourceDirectory.appendingPathComponent("audiocpp_server").path }

    static var installed: Bool { FileManager.default.isExecutableFile(atPath: cliBinary) }
    static var serverInstalled: Bool { FileManager.default.isExecutableFile(atPath: serverBinary) }

    static var vadModelDirectory: URL {
        resourceDirectory.appendingPathComponent("assets/framework/models/silero_vad", isDirectory: true)
    }

    static var vadInstalled: Bool {
        FileManager.default.fileExists(atPath: vadModelDirectory.path)
    }

    /// Rate of the WAV files the app prepares; audio.cpp reports word and segment spans in
    /// samples of its input.
    static let sampleRate = 16_000.0

    static func runtimeEnvironment(base: [String: String]) -> [String: String] {
        var environment = base
        environment["GGML_METAL_CONCURRENCY_DISABLE"] = "1"
        environment["GGML_METAL_SHARED_BUFFERS_DISABLE"] = "1"
        return environment
    }

    static var threadCount: Int {
        max(1, min(8, ProcessInfo.processInfo.activeProcessorCount))
    }

    enum Input: Equatable {
        case file(URL)
        /// Every `.wav` in the directory, transcribed in name order by one model session.
        case directory(URL)
    }

    /// Session options every run of the model gets, on the command line and in the server.
    static func sessionOptions(for model: SpeechModel, vadModel: URL, modelURL: URL) -> [String: String] {
        let family = model.family.rawValue
        var options = ["\(family).vad_model_path": vadModel.path]
        if model.family == .parakeet {
            // Full-context attention grows quadratically; long recordings switch to bounded windows.
            options["\(family).offline_mode"] = "auto"
        }
        // Qwen3-ASR reads word timings from a separate aligner checkpoint; without it the
        // transcript comes back untimed and the studio times each speech segment instead.
        if let aligner = model.installedAlignerURL(beside: modelURL) {
            options["\(family).forced_aligner_path"] = aligner.path
        }
        return options
    }

    static func asrArguments(model: SpeechModel, modelURL: URL, vadModel: URL,
                                         gpuIndex: Int, language: String, input: Input,
                                         textOut: URL, wordsOut: URL?) -> [String] {
        var arguments = [
            "--task", "asr", "--family", model.family.rawValue, "--mode", "offline",
            "--model", modelURL.path,
            "--backend", "metal", "--device", String(max(0, gpuIndex)),
            "--threads", String(threadCount)
        ]
        switch input {
        case .file(let url): arguments += ["--audio", url.path]
        case .directory(let url): arguments += ["--batch-audio-dir", url.path]
        }
        arguments += ["--text-out", textOut.path]
        if let wordsOut, model.providesWordTimings(beside: modelURL) {
            arguments += ["--words-out", wordsOut.path]
        }
        if let value = model.requestLanguage(for: language) {
            arguments += ["--request-option", "language=\(value)"]
        }
        for (key, value) in sessionOptions(for: model, vadModel: vadModel, modelURL: modelURL)
            .sorted(by: { $0.key < $1.key }) {
            arguments += ["--session-option", "\(key)=\(value)"]
        }
        return arguments
    }

    /// Silero VAD runs on the CPU: it is tiny, and the GPU stays free for the ASR model.
    static func vadArguments(audioURL: URL, vadModel: URL, segmentsOut: URL,
                                         configuration: AudioVADConfiguration) -> [String] {
        var arguments = [
            "--task", "vad", "--family", "silero_vad", "--mode", "offline",
            "--model", vadModel.path, "--backend", "cpu",
            "--threads", String(threadCount),
            "--audio", audioURL.path, "--segments-out", segmentsOut.path
        ]
        for (key, value) in configuration.requestOptions {
            arguments += ["--request-option", "\(key)=\(value)"]
        }
        return arguments
    }

    static let serverModelID = "speech"

    static func serverConfiguration(model: SpeechModel, modelURL: URL, vadModel: URL,
                                                gpuIndex: Int, port: Int) throws -> Data {
        let entry: [String: Any] = [
            "id": serverModelID,
            "family": model.family.rawValue,
            "path": modelURL.path,
            "task": "asr",
            "mode": "offline",
            "session_options": sessionOptions(for: model, vadModel: vadModel, modelURL: modelURL)
        ]
        let root: [String: Any] = [
            "host": "127.0.0.1",
            "port": port,
            "backend": "metal",
            "device": max(0, gpuIndex),
            "threads": threadCount,
            "lazy_load": false,
            "max_loaded_models": 1,
            "models": [entry]
        ]
        return try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
    }

    /// Body for `POST /v1/audio/transcriptions/details`, which reads the WAV from a local path.
    static func serverRequest(model: SpeechModel, modelURL: URL, audioURL: URL, language: String,
                                          withTimestamps: Bool) throws -> Data {
        var options: [String: String] = [:]
        if let value = model.requestLanguage(for: language) { options["language"] = value }
        if withTimestamps && model.providesWordTimings(beside: modelURL) {
            options["return_timestamps"] = "true"
        }
        var body: [String: Any] = ["model": serverModelID, "audio": audioURL.path]
        if !options.isEmpty { body["options"] = options }
        return try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
    }

    /// The `error.message` of an audio.cpp error response, if it has one.
    static func serverError(from data: Data) -> String? {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let error = root["error"] as? [String: Any], let message = error["message"] as? String {
            return message
        }
        return root["error"] as? String
    }

    /// The last line audio.cpp wrote before failing, which carries its reason.
    static func failureReason(_ diagnostics: String) -> String {
        let lines = diagnostics.split(whereSeparator: \.isNewline).map(String.init)
        return lines.last { $0.contains("audiocpp_cli failed:") || $0.contains("error") }
            ?? lines.last ?? ""
    }
}
