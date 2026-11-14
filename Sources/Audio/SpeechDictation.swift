// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AVFoundation
import Combine
import Foundation

@MainActor
final class SpeechDictationController: ObservableObject {
    static let shared = SpeechDictationController()

    @Published private(set) var isDictating = false
    @Published private(set) var isTranscribing = false
    @Published private(set) var statusText: String?
    @Published private(set) var isModelKeptLoaded = false
    @Published var error: String?

    private var recorder: AVAudioRecorder?
    private var process: Process?
    private var serverProcess: Process?
    private var serverErrorPipe: Pipe?
    private var serverDirectory: URL?
    private var serverModelURL: URL?
    private var serverModel: SpeechModel?
    private var serverGPUIndex: Int?
    private var serverStartTask: Task<Void, Error>?
    private var inferenceTask: URLSessionDataTask?
    private var audioURL: URL?
    private var outputDirectory: URL?
    private var timestampedOutput = false
    private var onText: ((String) -> Void)?
    private var onFailure: (() -> Void)?
    private var transcriptionID: UUID?
    private static let serverPort = 18_091

    private static let missingEngineMessage =
        "No se encontró el motor de voz audio.cpp. Vuelve a instalar ToshLLM / the audio.cpp speech engine is missing. Reinstall ToshLLM."
    private static let missingServerMessage =
        "No se encontró el servicio persistente de audio.cpp / the persistent audio.cpp service is missing."
    private static let transcriptionFailedMessage =
        "audio.cpp no pudo transcribir con la GPU / audio.cpp could not transcribe with the GPU."

    func toggle(model: SpeechModel, modelURL: URL, gpuIndex: Int, loadPolicy: SpeechLoadPolicy,
                onText: @escaping (String) -> Void) {
        if isTranscribing {
            cancel()
        } else if isDictating {
            stopAndTranscribe(model: model, modelURL: modelURL, gpuIndex: gpuIndex, loadPolicy: loadPolicy)
        } else {
            Task { await start(onText: onText) }
        }
    }

    func transcribe(fileURL: URL, model: SpeechModel, modelURL: URL, gpuIndex: Int,
                    loadPolicy: SpeechLoadPolicy,
                    onText: @escaping (String) -> Void,
                    onFailure: @escaping () -> Void) {
        guard !isDictating, !isTranscribing else {
            error = "Ya hay una transcripción en curso / a transcription is already running."
            onFailure()
            return
        }
        guard SpeechEngine.installed else {
            error = Self.missingEngineMessage
            onFailure()
            return
        }

        let converted = FileManager.default.temporaryDirectory
            .appendingPathComponent("ToshLLM-speech-\(UUID().uuidString).wav")
        audioURL = converted
        self.onText = onText
        self.onFailure = onFailure
        error = nil
        statusText = "Preparando audio… / Preparing audio…"
        isTranscribing = true

        let runID = UUID()
        transcriptionID = runID
        let converter = Process()
        converter.executableURL = URL(fileURLWithPath: "/usr/bin/afconvert")
        converter.arguments = [fileURL.path, converted.path, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1"]
        converter.standardOutput = FileHandle.nullDevice
        let diagnostics = Pipe()
        converter.standardError = diagnostics
        converter.terminationHandler = { [weak self] completed in
            let data = diagnostics.fileHandleForReading.readDataToEndOfFile()
            let details = String(data: data, encoding: .utf8) ?? ""
            Task { @MainActor in
                guard let self, self.transcriptionID == runID else { return }
                self.process = nil
                guard completed.terminationStatus == 0 else {
                    self.fail(runID: runID,
                              message: details.trimmingCharacters(in: .whitespacesAndNewlines))
                    return
                }
                self.runTranscription(audioURL: converted, model: model, modelURL: modelURL,
                                      gpuIndex: gpuIndex, loadPolicy: loadPolicy, withTimestamps: true)
            }
        }
        do {
            try converter.run()
            process = converter
        } catch {
            fail(runID: runID, message: error.localizedDescription)
        }
    }

    private func start(onText: @escaping (String) -> Void) async {
        guard SpeechEngine.installed else {
            error = Self.missingEngineMessage
            return
        }
        guard await AVCaptureDevice.requestAccess(for: .audio) else {
            error = "El acceso al micrófono está denegado. Actívalo en Ajustes del Sistema → Privacidad y seguridad → Micrófono / microphone access was denied. Enable it in System Settings → Privacy & Security → Microphone."
            return
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ToshLLM-speech-\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false
        ]
        do {
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            guard recorder.record() else { throw CocoaError(.fileWriteUnknown) }
            self.recorder = recorder
            audioURL = url
            self.onText = onText
            error = nil
            statusText = nil
            isDictating = true
        } catch {
            self.error = error.localizedDescription
            cleanupFiles()
        }
    }

    private func stopAndTranscribe(model: SpeechModel, modelURL: URL, gpuIndex: Int,
                                   loadPolicy: SpeechLoadPolicy) {
        recorder?.stop()
        recorder = nil
        isDictating = false
        guard let audioURL else { return }

        runTranscription(audioURL: audioURL, model: model, modelURL: modelURL, gpuIndex: gpuIndex,
                         loadPolicy: loadPolicy, withTimestamps: false)
    }

    private func runTranscription(audioURL: URL, model: SpeechModel, modelURL: URL, gpuIndex: Int,
                                  loadPolicy: SpeechLoadPolicy, withTimestamps: Bool) {
        if loadPolicy == .alwaysLoaded {
            runWithPersistentServer(audioURL: audioURL, model: model, modelURL: modelURL,
                                    gpuIndex: gpuIndex, withTimestamps: withTimestamps)
            return
        }
        let runID = UUID()
        transcriptionID = runID
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ToshLLM-transcript-\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            fail(runID: runID, message: error.localizedDescription)
            return
        }
        outputDirectory = directory
        timestampedOutput = withTimestamps

        let textURL = directory.appendingPathComponent("transcript.txt")
        let wordsURL = directory.appendingPathComponent("words.json")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: SpeechEngine.cliBinary)
        p.currentDirectoryURL = SpeechEngine.resourceDirectory
        p.arguments = SpeechEngine.asrArguments(
            model: model, modelURL: modelURL, vadModel: SpeechEngine.vadModelDirectory,
            gpuIndex: gpuIndex, language: model.dictationLanguage(SpeechPreferences.spokenLanguage),
            input: .file(audioURL),
            textOut: textURL, wordsOut: withTimestamps ? wordsURL : nil
        )
        p.environment = SpeechEngine.runtimeEnvironment(base: ProcessInfo.processInfo.environment)
        p.standardOutput = FileHandle.nullDevice
        let diagnostics = Pipe()
        p.standardError = diagnostics
        p.terminationHandler = { [weak self] completed in
            let data = diagnostics.fileHandleForReading.readDataToEndOfFile()
            let details = String(data: data, encoding: .utf8) ?? ""
            Task { @MainActor in
                self?.finish(runID: runID, status: completed.terminationStatus, diagnostics: details,
                             textURL: textURL, wordsURL: wordsURL)
            }
        }

        statusText = "Transcribiendo en la GPU… / Transcribing on the GPU…"
        isTranscribing = true
        do {
            try p.run()
            process = p
        } catch {
            fail(runID: runID, message: error.localizedDescription)
        }
    }

    func preload(model: SpeechModel, modelURL: URL, gpuIndex: Int) {
        guard SpeechEngine.serverInstalled else {
            error = Self.missingServerMessage
            return
        }
        Task {
            do {
                statusText = "Cargando el modelo de voz en la GPU… / Loading the speech model on the GPU…"
                try await ensureServer(model: model, modelURL: modelURL, gpuIndex: gpuIndex)
                statusText = nil
            } catch {
                statusText = nil
                self.error = error.localizedDescription
            }
        }
    }

    func unloadPersistentModel() {
        serverStartTask?.cancel()
        serverStartTask = nil
        if let serverProcess, serverProcess.isRunning { serverProcess.terminate() }
        serverErrorPipe?.fileHandleForReading.readabilityHandler = nil
        serverProcess = nil
        serverErrorPipe = nil
        if let serverDirectory { try? FileManager.default.removeItem(at: serverDirectory) }
        serverDirectory = nil
        serverModelURL = nil
        serverModel = nil
        serverGPUIndex = nil
        isModelKeptLoaded = false
    }

    func shutdown() {
        // Shutdown must not advance the transcription queue.
        onText = nil
        onFailure = nil
        cancel()
        unloadPersistentModel()
    }

    private func runWithPersistentServer(audioURL: URL, model: SpeechModel, modelURL: URL,
                                         gpuIndex: Int, withTimestamps: Bool) {
        guard SpeechEngine.serverInstalled else {
            let runID = UUID()
            transcriptionID = runID
            fail(runID: runID, message: Self.missingServerMessage)
            return
        }
        let runID = UUID()
        transcriptionID = runID
        timestampedOutput = withTimestamps
        statusText = isModelKeptLoaded
            ? "Transcribiendo en la GPU… / Transcribing on the GPU…"
            : "Cargando el modelo de voz en la GPU… / Loading the speech model on the GPU…"
        isTranscribing = true
        Task {
            do {
                try await ensureServer(model: model, modelURL: modelURL, gpuIndex: gpuIndex)
                guard transcriptionID == runID else { return }
                statusText = "Transcribiendo en la GPU… / Transcribing on the GPU…"
                try submitToServer(audioURL: audioURL, model: model, modelURL: modelURL, runID: runID,
                                   withTimestamps: withTimestamps)
            } catch {
                fail(runID: runID, message: error.localizedDescription)
            }
        }
    }

    private func ensureServer(model: SpeechModel, modelURL: URL, gpuIndex: Int) async throws {
        let normalizedGPU = max(0, gpuIndex)
        if serverProcess?.isRunning == true,
           serverModelURL == modelURL, serverModel == model, serverGPUIndex == normalizedGPU,
           isModelKeptLoaded { return }
        if let serverStartTask { try await serverStartTask.value; return }

        unloadPersistentModel()
        let task = Task { @MainActor in
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("ToshLLM-speech-server-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let configURL = directory.appendingPathComponent("server.json")
            try SpeechEngine.serverConfiguration(
                model: model, modelURL: modelURL, vadModel: SpeechEngine.vadModelDirectory,
                gpuIndex: normalizedGPU, port: Self.serverPort
            ).write(to: configURL, options: .atomic)
            serverDirectory = directory

            let server = Process()
            server.executableURL = URL(fileURLWithPath: SpeechEngine.serverBinary)
            server.currentDirectoryURL = SpeechEngine.resourceDirectory
            server.arguments = ["--config", configURL.path, "--no-ui"]
            server.environment = SpeechEngine.runtimeEnvironment(base: ProcessInfo.processInfo.environment)
            server.standardOutput = FileHandle.nullDevice
            let diagnostics = Pipe()
            server.standardError = diagnostics
            diagnostics.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let line = String(data: data, encoding: .utf8) else { return }
                AppLog.speech.debug("audio.cpp service: \(line, privacy: .public)")
            }
            server.terminationHandler = { [weak self] _ in
                Task { @MainActor in
                    guard let self, self.serverProcess === server else { return }
                    self.serverProcess = nil
                    self.serverModelURL = nil
                    self.serverModel = nil
                    self.serverGPUIndex = nil
                    self.isModelKeptLoaded = false
                }
            }
            try server.run()
            serverProcess = server
            serverErrorPipe = diagnostics
            serverModelURL = modelURL
            serverModel = model
            serverGPUIndex = normalizedGPU

            // A non-lazy model loads before the listener opens, so /health answering means
            // the weights are on the GPU.
            let readyURL = URL(string: "http://127.0.0.1:\(Self.serverPort)/health")!
            for _ in 0..<1_200 {
                try Task.checkCancellation()
                guard server.isRunning else {
                    throw CocoaError(.executableLoad,
                                     userInfo: [NSLocalizedDescriptionKey: "audio.cpp no pudo mantener el modelo cargado / couldn't keep the audio.cpp model loaded."])
                }
                var request = URLRequest(url: readyURL)
                request.timeoutInterval = 0.25
                if let (_, response) = try? await URLSession.shared.data(for: request),
                   (response as? HTTPURLResponse)?.statusCode == 200 {
                    isModelKeptLoaded = true
                    return
                }
                try await Task.sleep(for: .milliseconds(100))
            }
            throw URLError(.timedOut)
        }
        serverStartTask = task
        defer { serverStartTask = nil }
        try await task.value
    }

    private func submitToServer(audioURL: URL, model: SpeechModel, modelURL: URL, runID: UUID,
                                withTimestamps: Bool) throws {
        let url = URL(string: "http://127.0.0.1:\(Self.serverPort)/v1/audio/transcriptions/details")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 3_600
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try SpeechEngine.serverRequest(
            model: model, modelURL: modelURL, audioURL: audioURL,
            language: model.dictationLanguage(SpeechPreferences.spokenLanguage),
            withTimestamps: withTimestamps
        )
        inferenceTask = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            Task { @MainActor in
                self?.finishServer(runID: runID, data: data, response: response, error: error)
            }
        }
        inferenceTask?.resume()
    }

    private func finishServer(runID: UUID, data: Data?, response: URLResponse?, error: Error?) {
        guard transcriptionID == runID else { return }
        inferenceTask = nil
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard error == nil, (200..<300).contains(status), let data,
              let details = SpeechTranscript.details(jsonData: data) else {
            let message = error?.localizedDescription
                ?? data.flatMap(SpeechEngine.serverError(from:))
                ?? data.flatMap { String(data: $0, encoding: .utf8) }
                ?? Self.transcriptionFailedMessage
            fail(runID: runID, message: message)
            return
        }
        finishText(runID: runID, raw: rendered(text: details.text, words: details.words))
    }

    private func finish(runID: UUID, status: Int32, diagnostics: String,
                        textURL: URL, wordsURL: URL) {
        guard transcriptionID == runID else { return }

        guard status == 0 else {
            transcriptionID = nil
            process = nil
            isTranscribing = false
            statusText = nil
            let reason = SpeechEngine.failureReason(diagnostics)
            error = reason.isEmpty ? Self.transcriptionFailedMessage : reason
            AppLog.speech.error("audio.cpp failed: \(diagnostics, privacy: .public)")
            let failure = onFailure
            cleanupFiles()
            failure?()
            return
        }
        let text = (try? String(contentsOf: textURL, encoding: .utf8)) ?? ""
        let words = (try? Data(contentsOf: wordsURL)).map { SpeechTranscript.words(jsonData: $0) } ?? []
        finishText(runID: runID, raw: rendered(text: text, words: words))
    }

    /// Plain text for dictation; `[hh:mm:ss]` lines for attached recordings when the model
    /// timed its words.
    private func rendered(text: String, words: [SpeechWord]) -> String {
        guard timestampedOutput, !words.isEmpty, let last = words.last else { return text }
        let cues = SpeechTranscript.cues(text: text, words: words, span: 0...last.end)
        return cues.isEmpty ? text : SpeechTranscript.timestamped(cues)
    }

    private func finishText(runID: UUID, raw: String) {
        guard transcriptionID == runID else { return }
        transcriptionID = nil
        process = nil
        isTranscribing = false
        statusText = nil
        let text = SpeechTranscript.normalized(raw)
        guard !text.isEmpty else {
            error = "No se detectó voz / no speech was detected."
            let failure = onFailure
            cleanupFiles()
            failure?()
            return
        }
        let completion = onText
        cleanupFiles()
        completion?(text)
    }

    private func fail(runID: UUID, message: String) {
        guard transcriptionID == runID else { return }
        transcriptionID = nil
        process = nil
        isTranscribing = false
        statusText = nil
        error = message.isEmpty
            ? "No se pudo preparar el audio / the audio could not be prepared."
            : message
        let failure = onFailure
        cleanupFiles()
        failure?()
    }

    func cancel() {
        transcriptionID = nil
        recorder?.stop()
        recorder = nil
        process?.terminate()
        process = nil
        inferenceTask?.cancel()
        inferenceTask = nil
        isDictating = false
        isTranscribing = false
        statusText = nil
        let failure = onFailure
        cleanupFiles()
        failure?()
    }

    private func cleanupFiles() {
        if let audioURL { try? FileManager.default.removeItem(at: audioURL) }
        if let outputDirectory { try? FileManager.default.removeItem(at: outputDirectory) }
        audioURL = nil
        outputDirectory = nil
        timestampedOutput = false
        onText = nil
        onFailure = nil
    }
}
