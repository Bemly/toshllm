// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AVFoundation
import Combine
import Foundation

@MainActor
final class AudioPlaybackState: ObservableObject {
    @Published private(set) var currentTime: TimeInterval = 0
    @Published private(set) var currentCueID: Int?

    fileprivate func update(time: TimeInterval, cueID: Int?) {
        currentTime = time
        if currentCueID != cueID { currentCueID = cueID }
    }
}

@MainActor
final class AudioStudioController: ObservableObject {
    static let shared = AudioStudioController()

    @Published private(set) var sourceURL: URL?
    @Published private(set) var duration: TimeInterval = 0
    private(set) var currentTime: TimeInterval = 0
    @Published private(set) var fileSize = 0
    @Published private(set) var player: AVPlayer?
    @Published private(set) var isPlaying = false
    @Published private(set) var stage: AudioStudioStage = .idle
    @Published private(set) var progress = 0.0
    @Published private(set) var elapsed: TimeInterval = 0
    @Published private(set) var cues: [SubtitleCue] = []
    @Published private(set) var originalCues: [SubtitleCue] = []
    @Published private(set) var translatedCues: [SubtitleCue] = []
    @Published private(set) var transcriptMode = AudioTranscriptMode.original
    @Published private(set) var translatedBatchCount = 0
    @Published private(set) var translationBatchCount = 0
    @Published private(set) var translationTargetLanguage = ""
    @Published private(set) var translationModel = ""
    @Published private(set) var translationGlossary = ""
    @Published private(set) var vadPreviewState = AudioVADPreviewState.idle
    @Published private(set) var vadPreviewCues: [SubtitleCue] = []
    @Published private(set) var detectedLanguage = ""
    @Published private(set) var error: String?

    let playback = AudioPlaybackState()

    private var process: Process?
    private var stderrPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrBuffer = ""
    private var stdoutBuffer = ""
    private var segmentTotal = 0
    private var finishedSegments = 0
    private var workDirectory: URL?
    private var runID: UUID?
    private var ticker: Task<Void, Never>?
    private var translationTask: Task<Void, Never>?
    private var timeObserver: Any?
    private var restoreModel: SpeechModel?
    private var restoreModelURL: URL?
    private var restoreGPUIndex = 0
    private var shouldRestorePersistentModel = false
    private var undoHistory: [([SubtitleCue], [SubtitleCue])] = []
    private var redoHistory: [([SubtitleCue], [SubtitleCue])] = []

    private static var recoveryURL: URL {
        AppSupport.directory.appending(path: "AudioRecovery.json")
    }

    var isBusy: Bool {
        switch stage {
        case .preparing, .analyzingVAD, .transcribing, .translating: true
        default: false
        }
    }

    var isVideo: Bool {
        guard let ext = sourceURL?.pathExtension.lowercased() else { return false }
        return ["mp4", "mov", "m4v", "webm", "mkv"].contains(ext)
    }

    var remaining: TimeInterval? {
        guard progress > 0.08, progress < 1 else { return nil }
        return max(0, elapsed / progress - elapsed)
    }

    var hasTranslation: Bool { !translatedCues.isEmpty }

    var canRetryTranslation: Bool { !originalCues.isEmpty && !isBusy }

    var hasRecoveryProject: Bool {
        FileManager.default.fileExists(atPath: Self.recoveryURL.path)
    }

    func select(_ url: URL) {
        cancel(clearResult: true)
        player?.pause()
        isPlaying = false
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        timeObserver = nil
        sourceURL = url
        duration = 0
        updatePlaybackTime(0)
        vadPreviewState = .idle
        vadPreviewCues = []
        fileSize = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let player = AVPlayer(url: url)
        self.player = player
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.5, preferredTimescale: 600), queue: .main
        ) { [weak self] time in
            Task { @MainActor in self?.updatePlaybackTime(max(0, time.seconds)) }
        }
        Task { await loadDuration(url) }
    }

    func togglePlayback() {
        guard let player else { return }
        if player.rate == 0 {
            player.play()
            isPlaying = true
        } else {
            player.pause()
            isPlaying = false
        }
    }

    func seek(to seconds: TimeInterval) {
        let target = max(0, seconds)
        updatePlaybackTime(target)
        player?.seek(to: CMTime(seconds: target, preferredTimescale: 600))
    }

    private func updatePlaybackTime(_ time: TimeInterval) {
        currentTime = time
        playback.update(time: time, cueID: Self.cueID(
            at: time,
            in: originalCues.isEmpty ? cues : originalCues
        ))
    }

    func start(model: SpeechModel, modelURL: URL, vadConfiguration: AudioVADConfiguration,
               gpuIndex: Int, language: String,
               operation: AudioStudioOperation, targetLanguage: String,
               chatPort: Int, routerModel: String?, glossary: String,
               restorePersistentModel: Bool) {
        guard let sourceURL, !isBusy else { return }
        guard !SpeechDictationController.shared.isDictating,
              !SpeechDictationController.shared.isTranscribing,
              !AppleSpeechDictationController.shared.isDictating else {
            fail("Termina primero el dictado del micrófono o la transcripción adjunta / finish microphone dictation or the attached transcription first.")
            return
        }
        guard SpeechEngine.installed else {
            fail("No se encontró el motor de voz audio.cpp / the audio.cpp speech engine is missing.")
            return
        }
        guard !vadConfiguration.isEnabled || SpeechEngine.vadInstalled else {
            fail("No se encontró el detector de voz Silero / the Silero voice detector is missing.")
            return
        }
        if operation == .translateLocal && targetLanguage.trimmingCharacters(in: .whitespaces).isEmpty {
            fail("Elige un idioma de destino / choose a target language.")
            return
        }

        cancel(clearResult: true)
        self.sourceURL = sourceURL
        restoreModel = model
        restoreModelURL = modelURL
        restoreGPUIndex = max(0, gpuIndex)
        shouldRestorePersistentModel = restorePersistentModel
        SpeechDictationController.shared.shutdown()

        let id = UUID()
        runID = id
        stage = .preparing
        progress = 0.02
        error = nil
        startTicker()

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ToshLLM-audio-\(id.uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            fail(error.localizedDescription)
            return
        }
        workDirectory = directory
        let job = AudioTranscriptionJob(
            model: model, modelURL: modelURL, vadConfiguration: vadConfiguration,
            gpuIndex: gpuIndex, language: model.effectiveLanguage(language),
            operation: operation, targetLanguage: targetLanguage, chatPort: chatPort,
            routerModel: routerModel, glossary: glossary
        )
        let converted = directory.appendingPathComponent("source.wav")
        convert(sourceURL, to: converted, runID: id) { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let failure): self.fail(failure.message)
            case .success: self.planSpans(audioURL: converted, job: job, runID: id)
            }
        }
    }

    func previewVAD(vadConfiguration: AudioVADConfiguration) {
        guard let sourceURL, !isBusy, vadConfiguration.isEnabled else { return }
        guard SpeechEngine.installed, SpeechEngine.vadInstalled else {
            vadPreviewState = .failed("Falta audio.cpp o Silero VAD / audio.cpp or Silero VAD is missing.")
            return
        }
        let id = UUID()
        runID = id
        stage = .analyzingVAD
        progress = 0.05
        vadPreviewState = .running
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ToshLLM-vad-preview-\(id.uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            finishVADPreview(.failed(error.localizedDescription))
            return
        }
        workDirectory = directory
        let converted = directory.appendingPathComponent("source.wav")
        let sampleURL = directory.appendingPathComponent("preview.wav")
        convert(sourceURL, to: converted, runID: id) { [weak self] result in
            guard let self else { return }
            guard case .success = result else {
                self.finishVADPreview(.failed("No se pudo preparar la muestra / the sample could not be prepared."))
                return
            }
            Task { @MainActor in
                let written = await Task.detached {
                    guard let data = try? Data(contentsOf: converted, options: .alwaysMapped),
                          let wave = PCM16Wave(data: data) else { return false }
                    return (try? wave.wavFile(from: 0, to: Self.vadPreviewSeconds)
                        .write(to: sampleURL, options: .atomic)) != nil
                }.value
                guard self.runID == id else { return }
                guard written else {
                    self.finishVADPreview(.failed("No se pudo preparar la muestra / the sample could not be prepared."))
                    return
                }
                self.progress = 0.5
                self.runVAD(audioURL: sampleURL, configuration: vadConfiguration, runID: id) { result in
                    switch result {
                    case .failure:
                        self.finishVADPreview(.failed("VAD no pudo analizar la muestra / VAD could not analyze the sample."))
                    case .success(let spans):
                        self.vadPreviewCues = spans.enumerated().map { index, span in
                            SubtitleCue(id: index + 1, start: span.lowerBound, end: span.upperBound, text: "")
                        }
                        let speech = spans.reduce(0) { $0 + max(0, $1.upperBound - $1.lowerBound) }
                        let sample = min(Self.vadPreviewSeconds, self.duration > 0 ? self.duration : Self.vadPreviewSeconds)
                        self.finishVADPreview(.ready(segmentCount: spans.count,
                                                     speechSeconds: speech, sampleSeconds: sample))
                    }
                }
            }
        }
    }

    nonisolated private static let vadPreviewSeconds = 30.0

    private func finishVADPreview(_ state: AudioVADPreviewState) {
        runID = nil
        process = nil
        stage = .idle
        progress = 0
        vadPreviewState = state
        cleanupWorkDirectory()
    }

    /// Converts any audio or video track to the 16 kHz mono PCM WAV the engine reads.
    private func convert(_ source: URL, to destination: URL, runID id: UUID,
                         completion: @escaping @MainActor (Result<Void, AudioStudioFailure>) -> Void) {
        let converter = Process()
        converter.executableURL = URL(fileURLWithPath: "/usr/bin/afconvert")
        converter.arguments = [source.path, destination.path, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1"]
        converter.standardOutput = FileHandle.nullDevice
        let diagnostics = Pipe()
        converter.standardError = diagnostics
        converter.terminationHandler = { [weak self] completed in
            let data = diagnostics.fileHandleForReading.readDataToEndOfFile()
            let message = String(decoding: data, as: UTF8.self)
            Task { @MainActor in
                guard let self, self.runID == id else { return }
                self.process = nil
                completion(completed.terminationStatus == 0
                           ? .success(())
                           : .failure(AudioStudioFailure(message.trimmingCharacters(in: .whitespacesAndNewlines))))
            }
        }
        do {
            try converter.run()
            process = converter
        } catch {
            completion(.failure(AudioStudioFailure(error.localizedDescription)))
        }
    }

    /// Runs Silero VAD over a WAV and returns its speech spans in seconds.
    private func runVAD(audioURL: URL, configuration: AudioVADConfiguration, runID id: UUID,
                        completion: @escaping @MainActor (Result<[ClosedRange<TimeInterval>], AudioStudioFailure>) -> Void) {
        let segmentsURL = audioURL.deletingPathExtension().appendingPathExtension("segments.json")
        let vad = Process()
        vad.executableURL = URL(fileURLWithPath: SpeechEngine.cliBinary)
        vad.currentDirectoryURL = SpeechEngine.resourceDirectory
        vad.arguments = SpeechEngine.vadArguments(
            audioURL: audioURL, vadModel: SpeechEngine.vadModelDirectory,
            segmentsOut: segmentsURL, configuration: configuration
        )
        vad.environment = SpeechEngine.runtimeEnvironment(base: ProcessInfo.processInfo.environment)
        vad.standardOutput = FileHandle.nullDevice
        let diagnostics = Pipe()
        vad.standardError = diagnostics
        vad.terminationHandler = { [weak self] completed in
            let data = diagnostics.fileHandleForReading.readDataToEndOfFile()
            let details = String(decoding: data, as: UTF8.self)
            Task { @MainActor in
                guard let self, self.runID == id else { return }
                self.process = nil
                guard completed.terminationStatus == 0 else {
                    AppLog.speech.error("audio.cpp VAD failed: \(details, privacy: .public)")
                    completion(.failure(AudioStudioFailure(SpeechEngine.failureReason(details))))
                    return
                }
                // No segments file means Silero found no speech at all.
                let spans = (try? Data(contentsOf: segmentsURL)).map { SpeechTranscript.speechSpans(jsonData: $0) } ?? []
                completion(.success(spans))
            }
        }
        do {
            try vad.run()
            process = vad
        } catch {
            completion(.failure(AudioStudioFailure(error.localizedDescription)))
        }
    }

    /// Speech spans from Silero VAD, or fixed windows over the whole recording without it.
    private func planSpans(audioURL: URL, job: AudioTranscriptionJob, runID id: UUID) {
        guard job.vadConfiguration.isEnabled else {
            writeSegments(audioURL: audioURL, spans: nil, job: job, runID: id)
            return
        }
        stage = .analyzingVAD
        progress = 0.04
        runVAD(audioURL: audioURL, configuration: job.vadConfiguration, runID: id) { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let failure):
                self.fail(failure.message.isEmpty
                          ? "VAD no pudo analizar el audio / VAD could not analyze the audio."
                          : failure.message)
            case .success(let spans) where spans.isEmpty:
                self.fail("No se detectó voz / no speech was detected.")
            case .success(let spans):
                self.writeSegments(audioURL: audioURL, spans: spans, job: job, runID: id)
            }
        }
    }

    private func writeSegments(audioURL: URL, spans: [ClosedRange<TimeInterval>]?,
                               job: AudioTranscriptionJob, runID id: UUID) {
        guard let workDirectory else { return }
        let segments = workDirectory.appendingPathComponent("segments", isDirectory: true)
        Task { @MainActor in
            let planned: [ClosedRange<TimeInterval>]? = await Task.detached {
                guard let data = try? Data(contentsOf: audioURL, options: .alwaysMapped),
                      let wave = PCM16Wave(data: data),
                      (try? FileManager.default.createDirectory(at: segments,
                                                               withIntermediateDirectories: true)) != nil else {
                    return nil
                }
                let spans = (spans ?? SpeechTranscript.fixedWindows(duration: wave.duration))
                    .filter { $0.upperBound - $0.lowerBound >= 0.1 }
                for (index, span) in spans.enumerated() {
                    let url = segments.appendingPathComponent(Self.segmentName(index) + ".wav")
                    guard (try? wave.wavFile(from: span.lowerBound, to: span.upperBound)
                        .write(to: url, options: .atomic)) != nil else { return nil }
                }
                return spans
            }.value
            guard runID == id else { return }
            guard let planned else {
                fail("No se pudo preparar el audio / the audio could not be prepared.")
                return
            }
            guard !planned.isEmpty else {
                fail("No se detectó voz / no speech was detected.")
                return
            }
            runTranscription(segments: segments, spans: planned, job: job, runID: id)
        }
    }

    nonisolated static func segmentName(_ index: Int) -> String {
        "seg" + String(format: "%05d", index)
    }

    func cancel(clearResult: Bool = false) {
        runID = nil
        process?.terminate()
        process = nil
        stopReadingPipes()
        translationTask?.cancel()
        translationTask = nil
        ticker?.cancel()
        ticker = nil
        cleanupWorkDirectory()
        progress = 0
        elapsed = 0
        stage = .idle
        if clearResult {
            cues = []
            originalCues = []
            translatedCues = []
            transcriptMode = .original
            translatedBatchCount = 0
            translationBatchCount = 0
            detectedLanguage = ""
            error = nil
        }
    }

    func shutdown() {
        player?.pause()
        isPlaying = false
        cancel()
    }

    func content(for format: AudioExportFormat,
                 track: AudioExportTrack = .translated) -> String {
        let selected = exportCues(track: track)
        switch format {
        case .srt: return SubtitleCue.srt(selected)
        case .vtt: return SubtitleCue.vtt(selected)
        case .text: return SubtitleCue.plainText(selected)
        case .json:
            let rows: [[String: Any]] = selected.map {
                ["id": $0.id, "start": $0.start, "end": $0.end, "text": $0.text]
            }
            guard let data = try? JSONSerialization.data(withJSONObject: ["segments": rows], options: [.prettyPrinted, .sortedKeys]) else { return "" }
            return String(decoding: data, as: UTF8.self) + "\n"
        }
    }

    func exportCues(for track: AudioExportTrack) -> [SubtitleCue] {
        exportCues(track: track)
    }

    private func loadDuration(_ url: URL) async {
        let asset = AVURLAsset(url: url)
        guard let value = try? await asset.load(.duration), sourceURL == url else { return }
        duration = max(0, value.seconds)
    }

    /// One model session transcribes every span (`--batch-audio-dir`), and reports each span
    /// as it finishes, which drives the progress bar.
    private func runTranscription(segments: URL, spans: [ClosedRange<TimeInterval>],
                                  job: AudioTranscriptionJob, runID id: UUID) {
        guard let workDirectory else { return }
        stage = .transcribing
        progress = 0.08
        segmentTotal = spans.count
        finishedSegments = 0
        stdoutBuffer = ""
        let output = workDirectory.appendingPathComponent("transcripts", isDirectory: true)

        let asr = Process()
        asr.executableURL = URL(fileURLWithPath: SpeechEngine.cliBinary)
        asr.currentDirectoryURL = SpeechEngine.resourceDirectory
        asr.arguments = SpeechEngine.asrArguments(
            model: job.model, modelURL: job.modelURL, vadModel: SpeechEngine.vadModelDirectory,
            gpuIndex: job.gpuIndex, language: job.language, input: .directory(segments),
            textOut: output.appendingPathComponent("text.txt"),
            wordsOut: output.appendingPathComponent("words.json")
        )
        asr.environment = SpeechEngine.runtimeEnvironment(base: ProcessInfo.processInfo.environment)
        let results = Pipe()
        stdoutPipe = results
        asr.standardOutput = results
        results.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let chunk = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.consumeOutput(chunk, runID: id) }
        }
        let diagnostics = Pipe()
        stderrPipe = diagnostics
        asr.standardError = diagnostics
        diagnostics.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let chunk = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.consumeDiagnostics(chunk, runID: id) }
        }
        asr.terminationHandler = { [weak self] completed in
            Task { @MainActor in
                guard let self, self.runID == id else { return }
                self.stopReadingPipes()
                self.process = nil
                guard completed.terminationStatus == 0 else {
                    AppLog.speech.error("audio.cpp failed: \(self.stderrBuffer, privacy: .public)")
                    let reason = SpeechEngine.failureReason(self.stderrBuffer)
                    self.fail(reason.isEmpty
                              ? "audio.cpp no pudo completar la transcripción / audio.cpp could not complete the transcription."
                              : reason)
                    return
                }
                self.loadTranscription(output: output, spans: spans, job: job)
            }
        }
        do {
            try asr.run()
            process = asr
        } catch {
            fail(error.localizedDescription)
        }
    }

    nonisolated static func cueID(at time: TimeInterval, in cues: [SubtitleCue]) -> Int? {
        guard time >= 0, !cues.isEmpty else { return nil }
        var lower = 0
        var upper = cues.count - 1
        while lower <= upper {
            let middle = lower + (upper - lower) / 2
            let cue = cues[middle]
            if time < cue.start {
                upper = middle - 1
            } else if time >= cue.end {
                lower = middle + 1
            } else {
                return cue.id
            }
        }
        return nil
    }

    private func consumeDiagnostics(_ chunk: String, runID: UUID) {
        guard self.runID == runID else { return }
        stderrBuffer += chunk
        if stderrBuffer.count > 32_000 { stderrBuffer = String(stderrBuffer.suffix(16_000)) }
    }

    /// Counts the `request_index=` line the CLI prints as each span finishes.
    private func consumeOutput(_ chunk: String, runID: UUID) {
        guard self.runID == runID, segmentTotal > 0 else { return }
        stdoutBuffer += chunk
        var lines = stdoutBuffer.components(separatedBy: "\n")
        stdoutBuffer = lines.removeLast()
        finishedSegments += lines.filter { $0.hasPrefix("request_index=") }.count
        progress = 0.08 + min(1, Double(finishedSegments) / Double(segmentTotal)) * 0.67
    }

    private func stopReadingPipes() {
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe = nil
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stdoutPipe = nil
    }

    private func loadTranscription(output: URL, spans: [ClosedRange<TimeInterval>],
                                   job: AudioTranscriptionJob) {
        var cues: [SubtitleCue] = []
        for (index, span) in spans.enumerated() {
            let name = Self.segmentName(index)
            let text = (try? String(contentsOf: output.appendingPathComponent("text_\(name).txt"),
                                    encoding: .utf8)) ?? ""
            let words = (try? Data(contentsOf: output.appendingPathComponent("words_\(name).json")))
                .map { SpeechTranscript.words(jsonData: $0, offset: span.lowerBound) } ?? []
            cues += SpeechTranscript.cues(text: text, words: words, span: span)
        }
        let parsed = SpeechTranscript.numbered(cues)
        guard !parsed.isEmpty else {
            fail("No se detectó voz / no speech was detected.")
            return
        }
        originalCues = parsed
        translatedCues = []
        transcriptMode = .original
        refreshVisibleCues()
        detectedLanguage = job.language == "auto" ? "" : job.language
        saveRecovery()
        if job.operation == .translateLocal {
            translate(parsed, targetLanguage: job.targetLanguage, port: job.chatPort,
                      routerModel: job.routerModel, glossary: job.glossary)
        } else {
            complete()
        }
    }


    func retryTranslation(targetLanguage: String, port: Int,
                          routerModel: String?, glossary: String) {
        guard canRetryTranslation else { return }
        switch stage {
        case .failed: break
        default: translatedCues = []
        }
        error = nil
        translate(originalCues, targetLanguage: targetLanguage, port: port,
                  routerModel: routerModel, glossary: glossary)
    }

    private func translate(_ source: [SubtitleCue], targetLanguage: String,
                           port: Int, routerModel: String?, glossary: String) {
        let sameTranslation = translationTargetLanguage == targetLanguage
            && translationModel == (routerModel ?? "")
            && translationGlossary == glossary
        let preserved = sameTranslation
            ? Dictionary(uniqueKeysWithValues: translatedCues.map { ($0.id, $0.text) })
            : [:]
        let pending = source.filter { preserved[$0.id] == nil }
        stage = .translating
        progress = 0.76
        let batches = Self.translationBatches(pending)
        translationTargetLanguage = targetLanguage
        translationModel = routerModel ?? ""
        translationGlossary = glossary
        translatedBatchCount = 0
        translationBatchCount = batches.count
        if !sameTranslation { translatedCues = [] }
        refreshVisibleCues()
        saveRecovery()
        guard !batches.isEmpty else {
            adoptTranslationInTranscriptMode()
            refreshVisibleCues()
            complete()
            return
        }
        translationTask = Task { [weak self] in
            guard let self else { return }
            do {
                var translated = preserved
                for (index, batch) in batches.enumerated() {
                    try Task.checkCancellation()
                    let context = Self.translationContext(for: batch, in: source, translated: translated)
                    let result = try await self.translateBatchResilient(
                        batch, targetLanguage: targetLanguage,
                        port: port, routerModel: routerModel,
                        glossary: glossary, context: context
                    )
                    translated.merge(result) { _, new in new }
                    self.translatedCues = source.compactMap { cue in
                        translated[cue.id].map {
                            SubtitleCue(id: cue.id, start: cue.start, end: cue.end, text: $0)
                        }
                    }
                    self.translatedBatchCount = index + 1
                    self.refreshVisibleCues()
                    self.saveRecovery()
                    self.progress = 0.76 + Double(index + 1) / Double(batches.count) * 0.22
                }
                self.translatedCues = source.map {
                    SubtitleCue(id: $0.id, start: $0.start, end: $0.end,
                                text: translated[$0.id] ?? $0.text)
                }
                self.adoptTranslationInTranscriptMode()
                self.refreshVisibleCues()
                self.saveRecovery()
                self.complete()
            } catch is CancellationError {
                return
            } catch {
                self.fail(error.localizedDescription)
            }
        }
    }

    private func translateBatch(_ cues: [SubtitleCue], targetLanguage: String,
                                port: Int, routerModel: String?, glossary: String,
                                context: String) async throws -> [Int: String] {
        let request = try Self.translationRequest(
            cues: cues, targetLanguage: targetLanguage, port: port,
            routerModel: routerModel, apiKey: ServerSettings.activeAPIKey(),
            glossary: glossary, context: context
        )
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        guard (200..<300).contains(http.statusCode) else {
            throw AudioTranslationError.server(http.statusCode, Self.serverError(from: data))
        }
        return try Self.translations(from: data, expectedCues: cues)
    }

    private func translateBatchResilient(_ cues: [SubtitleCue], targetLanguage: String,
                                         port: Int, routerModel: String?, glossary: String,
                                         context: String) async throws -> [Int: String] {
        do {
            return try await translateBatch(cues, targetLanguage: targetLanguage,
                                            port: port, routerModel: routerModel,
                                            glossary: glossary, context: context)
        } catch AudioTranslationError.invalidResponse where cues.count > 1 {
            let middle = cues.count / 2
            let first = try await translateBatchResilient(
                Array(cues[..<middle]), targetLanguage: targetLanguage,
                port: port, routerModel: routerModel,
                glossary: glossary, context: context
            )
            let second = try await translateBatchResilient(
                Array(cues[middle...]), targetLanguage: targetLanguage,
                port: port, routerModel: routerModel,
                glossary: glossary, context: context
            )
            return first.merging(second) { _, translated in translated }
        } catch AudioTranslationError.invalidResponse where !context.isEmpty {
            return try await translateBatch(
                cues, targetLanguage: targetLanguage, port: port,
                routerModel: routerModel, glossary: glossary, context: ""
            )
        }
    }

    nonisolated static func translationRequest(cues: [SubtitleCue], targetLanguage: String,
                                               port: Int, routerModel: String?,
                                               apiKey: String?, glossary: String = "",
                                               context: String = "") throws -> URLRequest {
        let rows = cues.map { ["id": $0.id, "text": $0.text] as [String: Any] }
        let payload: [String: Any] = [
            "target_language": targetLanguage,
            "glossary": glossary,
            "neighboring_context": context,
            "cues": rows
        ]
        let input = try JSONSerialization.data(withJSONObject: payload)
        let prompt = String(decoding: input, as: UTF8.self)
        let itemSchemas: [[String: Any]] = cues.map { cue in
            [
                "type": "object",
                "properties": [
                    "id": ["const": cue.id],
                    "text": ["type": "string", "minLength": 1]
                ],
                "required": ["id", "text"],
                "additionalProperties": false
            ]
        }
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "translations": [
                    "type": "array", "prefixItems": itemSchemas,
                    "minItems": cues.count, "maxItems": cues.count
                ]
            ],
            "required": ["translations"],
            "additionalProperties": false
        ]
        var body: [String: Any] = [
            "stream": false,
            "temperature": 0.1,
            "max_tokens": min(8192, max(2048, cues.reduce(0) { $0 + $1.text.count } * 2)),
            "chat_template_kwargs": ["enable_thinking": false],
            "response_format": [
                "type": "json_schema",
                "json_schema": [
                    "name": "subtitle_translation", "strict": true, "schema": schema
                ]
            ],
            "messages": [
                ["role": "system", "content": "Translate only the text inside the cues array into the requested target language. The source cue is authoritative: do not summarize, omit, invent, reorder, merge, split or expand its content. The glossary and neighboring_context are reference-only; never translate, copy, paraphrase or include their text in the answer. Keep terminology, names, tone and grammatical references consistent. Preserve each exact id and return one concise, non-empty translation for every cue. Return only the required JSON."],
                ["role": "user", "content": prompt]
            ]
        ]
        if let routerModel, !routerModel.isEmpty { body["model"] = routerModel }
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions") else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 600
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("off", forHTTPHeaderField: "X-Tosh-Agent")
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return request
    }

    nonisolated static func translations(from data: Data,
                                         expectedCues cues: [SubtitleCue]) throws -> [Int: String] {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = root["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw AudioTranslationError.invalidResponse
        }
        guard let first = content.firstIndex(of: "{"), let last = content.lastIndex(of: "}") else {
            throw AudioTranslationError.invalidResponse
        }
        let json = Data(content[first...last].utf8)
        guard let translatedRoot = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let translations = translatedRoot["translations"] as? [[String: Any]] else {
            throw AudioTranslationError.invalidResponse
        }
        var result: [Int: String] = [:]
        for row in translations {
            guard let id = row["id"] as? Int, let text = row["text"] as? String else { continue }
            result[id] = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard result.count == cues.count,
              cues.allSatisfy({ result[$0.id]?.isEmpty == false }) else {
            throw AudioTranslationError.invalidResponse
        }
        for cue in cues {
            guard let text = result[cue.id], text.count <= max(100, cue.text.count * 3) else {
                throw AudioTranslationError.invalidResponse
            }
        }
        return result
    }

    nonisolated private static func serverError(from data: Data) -> String {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = root["error"] as? [String: Any],
              let message = error["message"] as? String else { return "" }
        return String(message.prefix(240))
    }

    nonisolated static func translationBatches(_ cues: [SubtitleCue]) -> [[SubtitleCue]] {
        cues.map { [$0] }
    }

    nonisolated static func translationContext(for batch: [SubtitleCue], in source: [SubtitleCue],
                                               translated: [Int: String]) -> String {
        guard let first = batch.first, let last = batch.last,
              let start = source.firstIndex(where: { $0.id == first.id }),
              let end = source.firstIndex(where: { $0.id == last.id }) else { return "" }
        let lower = max(0, start - 3)
        let upper = min(source.count - 1, end + 3)
        let neighboring = source[lower...upper].map { cue in
            let prior = translated[cue.id].map { " => \($0)" } ?? ""
            return "[\(cue.id)] \(cue.text)\(prior)"
        }.joined(separator: "\n")
        let memoryIDs = Array(translated.keys.sorted().prefix(6))
            + Array(translated.keys.sorted().suffix(8))
        let memory = Array(Set(memoryIDs)).sorted().compactMap { id -> String? in
            guard let sourceCue = source.first(where: { $0.id == id }),
                  let translatedText = translated[id] else { return nil }
            return "[\(id)] \(sourceCue.text) => \(translatedText)"
        }.joined(separator: "\n")
        return memory.isEmpty ? neighboring : "CONSISTENCY MEMORY:\n\(memory)\n\nNEIGHBORING CONTEXT:\n\(neighboring)"
    }

    /// Finishing a translation must not undo the view the user chose; only a
    /// transcript still showing the original alone has anywhere to move to.
    private func adoptTranslationInTranscriptMode() {
        if transcriptMode == .original { transcriptMode = .translated }
    }

    func setTranscriptMode(_ mode: AudioTranscriptMode) {
        transcriptMode = mode
        refreshVisibleCues()
    }

    func cuePair(id: Int) -> (original: SubtitleCue?, translated: SubtitleCue?) {
        (originalCues.first { $0.id == id }, translatedCues.first { $0.id == id })
    }

    func updateCue(id: Int, text: String, translated: Bool) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        recordEdit()
        if translated {
            guard let index = translatedCues.firstIndex(where: { $0.id == id }) else { return }
            translatedCues[index].text = trimmed
        } else {
            guard let index = originalCues.firstIndex(where: { $0.id == id }) else { return }
            originalCues[index].text = trimmed
        }
        refreshVisibleCues()
        saveRecovery()
    }

    func deleteCue(id: Int) {
        recordEdit()
        originalCues.removeAll { $0.id == id }
        translatedCues.removeAll { $0.id == id }
        normalizeCueIDs()
    }

    func mergeCueWithNext(id: Int) {
        guard let index = originalCues.firstIndex(where: { $0.id == id }),
              originalCues.indices.contains(index + 1) else { return }
        recordEdit()
        originalCues[index].end = originalCues[index + 1].end
        originalCues[index].text += " " + originalCues[index + 1].text
        originalCues.remove(at: index + 1)
        if let translatedIndex = translatedCues.firstIndex(where: { $0.id == id }),
           translatedCues.indices.contains(translatedIndex + 1) {
            translatedCues[translatedIndex].end = translatedCues[translatedIndex + 1].end
            translatedCues[translatedIndex].text += " " + translatedCues[translatedIndex + 1].text
            translatedCues.remove(at: translatedIndex + 1)
        }
        normalizeCueIDs()
    }

    func splitCue(id: Int) {
        guard let index = originalCues.firstIndex(where: { $0.id == id }) else { return }
        let cue = originalCues[index]
        let words = cue.text.split(whereSeparator: \.isWhitespace)
        guard words.count > 1 else { return }
        recordEdit()
        let middleWord = words.count / 2
        let middleTime = cue.start + (cue.end - cue.start) / 2
        originalCues[index] = SubtitleCue(id: cue.id, start: cue.start, end: middleTime,
                                          text: words[..<middleWord].joined(separator: " "))
        originalCues.insert(SubtitleCue(id: cue.id + 1, start: middleTime, end: cue.end,
                                       text: words[middleWord...].joined(separator: " ")), at: index + 1)
        if let translatedIndex = translatedCues.firstIndex(where: { $0.id == id }) {
            let translatedCue = translatedCues[translatedIndex]
            let translatedWords = translatedCue.text.split(whereSeparator: \.isWhitespace)
            if translatedWords.count > 1 {
                let translatedMiddle = translatedWords.count / 2
                translatedCues[translatedIndex] = SubtitleCue(
                    id: translatedCue.id, start: translatedCue.start, end: middleTime,
                    text: translatedWords[..<translatedMiddle].joined(separator: " ")
                )
                translatedCues.insert(SubtitleCue(
                    id: translatedCue.id + 1, start: middleTime, end: translatedCue.end,
                    text: translatedWords[translatedMiddle...].joined(separator: " ")
                ), at: translatedIndex + 1)
            } else {
                translatedCues.remove(at: translatedIndex)
            }
        }
        normalizeCueIDs()
    }

    func replaceAll(_ search: String, with replacement: String) {
        guard !search.isEmpty else { return }
        recordEdit()
        originalCues = originalCues.map { cue in
            var updated = cue
            updated.text = cue.text.replacingOccurrences(of: search, with: replacement,
                                                         options: [.caseInsensitive, .diacriticInsensitive])
            return updated
        }
        translatedCues = translatedCues.map { cue in
            var updated = cue
            updated.text = cue.text.replacingOccurrences(of: search, with: replacement,
                                                         options: [.caseInsensitive, .diacriticInsensitive])
            return updated
        }
        refreshVisibleCues()
        saveRecovery()
    }

    func undoEdit() {
        guard let previous = undoHistory.popLast() else { return }
        redoHistory.append((originalCues, translatedCues))
        originalCues = previous.0
        translatedCues = previous.1
        refreshVisibleCues()
        saveRecovery()
    }

    func redoEdit() {
        guard let next = redoHistory.popLast() else { return }
        undoHistory.append((originalCues, translatedCues))
        originalCues = next.0
        translatedCues = next.1
        refreshVisibleCues()
        saveRecovery()
    }

    var canUndoEdit: Bool { !undoHistory.isEmpty }
    var canRedoEdit: Bool { !redoHistory.isEmpty }

    private func recordEdit() {
        undoHistory.append((originalCues, translatedCues))
        if undoHistory.count > 50 { undoHistory.removeFirst() }
        redoHistory.removeAll()
    }

    func saveProject(to url: URL, targetLanguage: String, glossary: String) throws {
        let document = projectDocument(targetLanguage: targetLanguage, glossary: glossary)
        let data = try JSONEncoder.audioProject.encode(document)
        try data.write(to: url, options: .atomic)
    }

    func loadProject(from url: URL) throws {
        let data = try Data(contentsOf: url)
        let document = try JSONDecoder.audioProject.decode(AudioProjectDocument.self, from: data)
        let mediaURL = URL(fileURLWithPath: document.sourcePath)
        guard FileManager.default.fileExists(atPath: mediaURL.path) else {
            throw CocoaError(.fileNoSuchFile,
                             userInfo: [NSLocalizedDescriptionKey: "No se encontró el audio o vídeo original / the original audio or video could not be found."])
        }
        select(mediaURL)
        originalCues = document.originalCues
        translatedCues = document.translatedCues
        detectedLanguage = document.detectedLanguage
        translationTargetLanguage = document.targetLanguage
        translationModel = document.translationModel
        translationGlossary = document.glossary
        transcriptMode = translatedCues.isEmpty ? .original : .bilingual
        refreshVisibleCues()
        saveRecovery()
    }

    func loadRecoveryProject() throws {
        try loadProject(from: Self.recoveryURL)
    }

    private func normalizeCueIDs() {
        let translationByID = Dictionary(uniqueKeysWithValues: translatedCues.map { ($0.id, $0.text) })
        let pairs = originalCues.enumerated().map { offset, cue in
            (SubtitleCue(id: offset + 1, start: cue.start, end: cue.end, text: cue.text),
             translationByID[cue.id])
        }
        originalCues = pairs.map(\.0)
        translatedCues = pairs.compactMap { cue, translation in
            translation.map {
                SubtitleCue(id: cue.id, start: cue.start, end: cue.end, text: $0)
            }
        }
        refreshVisibleCues()
        saveRecovery()
    }

    private func exportCues(track: AudioExportTrack) -> [SubtitleCue] {
        switch track {
        case .original:
            originalCues
        case .translated:
            translatedCues.isEmpty ? originalCues : translatedCues
        case .bilingual:
            originalCues.map { cue in
                let translated = translatedCues.first { $0.id == cue.id }?.text
                let text = translated.map { "\(cue.text)\n\($0)" } ?? cue.text
                return SubtitleCue(id: cue.id, start: cue.start, end: cue.end, text: text)
            }
        }
    }

    private func refreshVisibleCues() {
        switch transcriptMode {
        case .original, .bilingual:
            cues = originalCues
        case .translated:
            cues = translatedCues.isEmpty ? originalCues : translatedCues
        }
    }

    private func projectDocument(targetLanguage: String, glossary: String) -> AudioProjectDocument {
        AudioProjectDocument(
            version: 1, savedAt: .now, sourcePath: sourceURL?.path ?? "",
            detectedLanguage: detectedLanguage, targetLanguage: targetLanguage,
            translationModel: translationModel, glossary: glossary,
            originalCues: originalCues, translatedCues: translatedCues
        )
    }

    private func saveRecovery() {
        guard sourceURL != nil, !originalCues.isEmpty else { return }
        do {
            let url = Self.recoveryURL
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            let document = projectDocument(targetLanguage: translationTargetLanguage,
                                           glossary: translationGlossary)
            try JSONEncoder.audioProject.encode(document).write(to: url, options: .atomic)
        } catch {
            AppLog.speech.error("audio recovery save failed: \(error.localizedDescription)")
        }
    }

    private func complete() {
        runID = nil
        translationTask = nil
        stage = .completed
        progress = 1
        ticker?.cancel()
        ticker = nil
        cleanupWorkDirectory()
        restorePersistentModelIfNeeded()
    }

    private func fail(_ message: String) {
        runID = nil
        translationTask = nil
        stage = .failed(message)
        error = message
        process = nil
        ticker?.cancel()
        ticker = nil
        cleanupWorkDirectory()
        restorePersistentModelIfNeeded()
    }

    private func startTicker() {
        elapsed = 0
        ticker?.cancel()
        ticker = Task { [weak self] in
            let started = Date.now
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(250))
                guard let self else { return }
                self.elapsed = Date.now.timeIntervalSince(started)
            }
        }
    }

    private func restorePersistentModelIfNeeded() {
        guard shouldRestorePersistentModel, let restoreModel, let restoreModelURL else { return }
        shouldRestorePersistentModel = false
        SpeechDictationController.shared.preload(model: restoreModel, modelURL: restoreModelURL,
                                                 gpuIndex: restoreGPUIndex)
    }

    private func cleanupWorkDirectory() {
        if let workDirectory { try? FileManager.default.removeItem(at: workDirectory) }
        workDirectory = nil
        stderrBuffer = ""
        stdoutBuffer = ""
        segmentTotal = 0
        finishedSegments = 0
    }

}

/// Everything a transcription carries from the sidebar to the translation step.
private struct AudioTranscriptionJob {
    let model: SpeechModel
    let modelURL: URL
    let vadConfiguration: AudioVADConfiguration
    let gpuIndex: Int
    let language: String
    let operation: AudioStudioOperation
    let targetLanguage: String
    let chatPort: Int
    let routerModel: String?
    let glossary: String
}

struct AudioStudioFailure: Error {
    let message: String

    init(_ message: String) { self.message = message }
}
