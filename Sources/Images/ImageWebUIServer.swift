// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import AppKit
import Combine

/// One warm `sd-server`. Images generate through its HTTP API so the text
/// conditioning cache survives from one run to the next.
@MainActor
final class ImageWebUIServer: ObservableObject {
    static let shared = ImageWebUIServer()

    enum State: Equatable {
        case stopped
        case starting
        case running
        case failed(String)
    }

    struct Launch: Equatable {
        var contextArgs: [String]
        var env: [String: String]
        var port: Int
    }

    struct JobSnapshot {
        var status: String
        var stage: String
        var completed: Int?
        var total: Int?
        var previewB64: String?
        var imagesB64: [String]
        var error: String?
    }

    struct ConditioningCacheSnapshot {
        struct Entry {
            var id: String
            var prompt: String
            var width: Int
            var height: Int
            var clipSkip: Int?
            var zeroOutMasked: Bool?
            var ageSeconds: Int
        }
        struct Lora {
            var path: String
            var strength: Double
        }
        var capacity: Int
        var size: Int
        var storageScopeID: String
        var modelLoaded: Bool
        var isSnapshot: Bool
        var textEncoder: String
        var diffusionModel: String
        var tokenizer: String
        var loras: [Lora]
        var entries: [Entry]
    }

    enum ServerError: LocalizedError {
        case missingBinary
        case busy
        case cacheSnapshotUnavailable
        case exited(String)
        case timeout
        case http(Int, String)

        var errorDescription: String? {
            switch self {
            case .missingBinary: return "MISSING_SD_SERVER"
            case .busy: return "IMAGE_SERVER_BUSY"
            case .cacheSnapshotUnavailable: return "CACHE_SNAPSHOT_UNAVAILABLE"
            case .exited(let tail): return tail.isEmpty ? "EXIT" : tail
            case .timeout: return "TIMEOUT"
            case .http(_, let body): return body
            }
        }
    }

    @Published private(set) var state: State = .stopped
    @Published private(set) var port: Int = 1234
    @Published private(set) var logText: String = ""
    @Published private(set) var loadedSummary: String = ""

    private var process: Process?
    private var stderrPipe: Pipe?
    private var launchKey = ""
    @Published private(set) var jobsInFlight = 0
    /// Lines the image log should see.
    var onLog: ((String) -> Void)?

    private init() {}

    var isRunning: Bool {
        if case .running = state { return true }
        return false
    }

    /// Process is up or still coming up — Settings should offer Stop, not a dead Start.
    var isActive: Bool {
        switch state {
        case .starting, .running: return true
        default: return false
        }
    }

    var url: URL? {
        guard isRunning else { return nil }
        return URL(string: "http://127.0.0.1:\(port)/")
    }

    /// sd-server reports `empty` with no model loaded and `ready` once weights are in.
    /// Both mean the HTTP listener is up; waiting only for `ready` leaves Settings stuck
    /// on "Starting" forever for a background WebUI with no model yet.
    nonisolated static func isListenReady(status: String) -> Bool {
        switch status {
        case "ready", "empty": return true
        default: return false
        }
    }

    static func bundledBinary() -> String? {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("bin-image/sd-server").path,
           FileManager.default.isExecutableFile(atPath: bundled) {
            return bundled
        }
        let dev = NSString(string: "~/Projects/mps/toshllm/vendor/stable-diffusion.cpp/build-static/bin/sd-server")
            .expandingTildeInPath
        if FileManager.default.isExecutableFile(atPath: dev) { return dev }
        return nil
    }

    /// Default model root for the web UI: `vision/` when present (MOSS layout), else models root.
    static func modelDirectory() -> URL {
        let root = ServerSettings.modelsDirectory
        let vision = root.appendingPathComponent("vision", isDirectory: true)
        if FileManager.default.fileExists(atPath: vision.path) { return vision }
        return root
    }

    func syncWithPreference() {
        let enabled = UserDefaults.standard.bool(forKey: SettingsKeys.imagenWebUI)
        if enabled { start() } else { stop() }
    }

    func setEnabled(_ enabled: Bool) {
        UserDefaults.standard.set(enabled, forKey: SettingsKeys.imagenWebUI)
        if enabled { start() } else { stop() }
    }

    /// Empty WebUI, until Generate starts the same process with a model loaded.
    func start() {
        if isRunning { return }
        if case .starting = state { return }
        // Orphan: process still alive after a failed readiness wait — reclaim it.
        if process?.isRunning == true {
            Task { await reclaimRunningProcess() }
            return
        }
        let port = UserDefaults.standard.object(forKey: SettingsKeys.imagenWebUIPort) as? Int ?? 1234
        let launch = Launch(contextArgs: [], env: ProcessInfo.processInfo.environment, port: max(1, min(port, 65535)))
        Task {
            do {
                try await ensure(launch)
            } catch {
                if case .starting = state {
                    stopProcess()
                    state = .failed((error as? LocalizedError)?.errorDescription ?? error.localizedDescription)
                }
            }
        }
    }

    func beginJob() { jobsInFlight += 1 }
    func endJob() { jobsInFlight = max(0, jobsInFlight - 1) }

    /// Start or keep the server. A different model, GPU, or TE mode restarts it.
    /// Steps, seed, and prompt do not.
    func ensure(_ launch: Launch) async throws {
        let key = Self.fingerprint(launch)
        if isRunning, process?.isRunning == true, key == launchKey { return }
        if jobsInFlight > 0, key != launchKey { throw ServerError.busy }
        stopProcess()
        guard let bin = Self.bundledBinary() else {
            state = .failed("MISSING_SD_SERVER")
            throw ServerError.missingBinary
        }
        port = launch.port
        var args = [
            "--model-dir", Self.modelDirectory().path,
            "-l", "127.0.0.1",
            "--listen-port", String(port),
            "--conditioning-cache-size", "16",
        ]
        if !launch.contextArgs.contains("--auto-fit") {
            args += ["--auto-fit", "off"]
        }
        if !launch.contextArgs.contains("--conditioning-cache-dir") {
            args += ["--conditioning-cache-dir",
                     AppSupport.directory.appendingPathComponent("imagen-conditioning-cache", isDirectory: true).path]
        }
        args += launch.contextArgs
        if !args.contains("--release-te-after-encode") && !args.contains("--keep-te") {
            args.append("--keep-te")
        }
        if !args.contains("--lora-model-dir"), let lora = ImageGenPool.loraDirectory() {
            args += ["--lora-model-dir", lora.path]
        }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: bin)
        p.arguments = args
        p.environment = launch.env
        let err = Pipe()
        p.standardOutput = err
        p.standardError = err
        stderrPipe = err
        err.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                self?.appendLog(text)
            }
        }
        p.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            Task { @MainActor in
                guard let self, self.process == proc else { return }
                EngineLock.remove(pid: proc.processIdentifier)
                self.process = nil
                self.launchKey = ""
                if case .starting = self.state, status != 0 {
                    self.state = .failed("EXIT_\(status)")
                } else if case .running = self.state {
                    self.state = .stopped
                }
            }
        }
        state = .starting
        loadedSummary = ""
        logText = ""
        appendLog("$ sd-server " + args.joined(separator: " ") + "\n")
        process = p
        do {
            try p.run()
            EngineLock.add(pid: p.processIdentifier)
        } catch {
            process = nil
            state = .failed(error.localizedDescription)
            throw error
        }
        launchKey = key
        do {
            try await waitUntilReady(process: p)
        } catch {
            if process == p {
                stopProcess()
                state = .failed((error as? LocalizedError)?.errorDescription ?? error.localizedDescription)
            }
            throw error
        }
        loadedSummary = launch.contextArgs.isEmpty
            ? ""
            : contextSummary(launch.contextArgs)
    }

    func submit(_ body: [String: Any]) async throws -> String {
        let data = try JSONSerialization.data(withJSONObject: body)
        let (payload, code) = try await request(path: "/sdcpp/v1/img_gen", method: "POST", body: data)
        guard code == 202 || code == 200 else {
            throw ServerError.http(code, String(data: payload, encoding: .utf8) ?? "HTTP \(code)")
        }
        let json = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any]
        guard let id = json?["id"] as? String else {
            throw ServerError.http(code, "missing job id")
        }
        return id
    }

    func poll(_ id: String) async throws -> JobSnapshot {
        let (payload, code) = try await request(path: "/sdcpp/v1/jobs/\(id)", method: "GET", body: nil)
        guard code == 200, let json = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any] else {
            throw ServerError.http(code, String(data: payload, encoding: .utf8) ?? "HTTP \(code)")
        }
        let progress = json["progress"] as? [String: Any]
        let preview = json["preview"] as? [String: Any]
        let result = json["result"] as? [String: Any]
        let images = (result?["images"] as? [[String: Any]])?.compactMap { $0["b64_json"] as? String } ?? []
        let error = json["error"] as? [String: Any]
        let message = error?["message"] as? String
        func intField(_ any: Any?) -> Int? {
            switch any {
            case let n as Int: return n
            case let n as NSNumber: return n.intValue
            default: return nil
            }
        }
        return JobSnapshot(
            status: json["status"] as? String ?? "",
            stage: progress?["stage"] as? String ?? "",
            completed: intField(progress?["completed"]),
            total: intField(progress?["total"]),
            previewB64: preview?["b64_json"] as? String,
            imagesB64: images,
            error: message)
    }

    func cancelJob(_ id: String) async throws {
        let (payload, code) = try await request(path: "/sdcpp/v1/jobs/\(id)/cancel", method: "POST", body: Data("{}".utf8))
        guard code == 200 else {
            throw ServerError.http(code, String(data: payload, encoding: .utf8) ?? "HTTP \(code)")
        }
    }

    func fetchConditioningCache() async throws -> ConditioningCacheSnapshot {
        let (payload, code) = try await request(path: "/sdcpp/v1/conditioning-cache", method: "GET", body: nil)
        if code == 503,
           let json = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any],
           json["code"] as? String == "cache_snapshot_unavailable" {
            throw ServerError.cacheSnapshotUnavailable
        }
        guard code == 200,
              let json = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any] else {
            throw ServerError.http(code, String(data: payload, encoding: .utf8) ?? "HTTP \(code)")
        }
        func intField(_ any: Any?) -> Int {
            switch any {
            case let n as Int: return n
            case let n as NSNumber: return n.intValue
            default: return 0
            }
        }
        let raw = json["entries"] as? [[String: Any]] ?? []
        let entries = raw.compactMap { row -> ConditioningCacheSnapshot.Entry? in
            guard let id = row["id"] as? String else { return nil }
            return .init(id: id,
                         prompt: row["prompt"] as? String ?? "",
                         width: intField(row["width"]),
                         height: intField(row["height"]),
                         clipSkip: (row["clip_skip"] as? NSNumber)?.intValue,
                         zeroOutMasked: row["zero_out_masked"] as? Bool,
                         ageSeconds: intField(row["age_seconds"]))
        }
        let loras = (json["loras"] as? [[String: Any]] ?? []).compactMap { row -> ConditioningCacheSnapshot.Lora? in
            guard let path = row["path"] as? String, let strength = row["strength"] as? NSNumber else { return nil }
            return .init(path: path, strength: strength.doubleValue)
        }
        return ConditioningCacheSnapshot(
            capacity: intField(json["capacity"]),
            size: intField(json["size"]),
            storageScopeID: json["storage_scope_id"] as? String ?? "",
            modelLoaded: (json["model_loaded"] as? Bool) ?? true,
            isSnapshot: (json["snapshot"] as? Bool) ?? false,
            textEncoder: json["text_encoder"] as? String ?? "",
            diffusionModel: json["diffusion_model"] as? String ?? "",
            tokenizer: json["tokenizer"] as? String ?? "",
            loras: loras,
            entries: entries)
    }

    func clearConditioningCache() async throws {
        let (payload, code) = try await request(path: "/sdcpp/v1/conditioning-cache", method: "DELETE", body: nil)
        guard code == 200 else {
            throw ServerError.http(code, String(data: payload, encoding: .utf8) ?? "HTTP \(code)")
        }
    }

    func deleteConditioningCacheEntry(id: String) async throws {
        let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? id
        let (payload, code) = try await request(path: "/sdcpp/v1/conditioning-cache?id=\(encoded)",
                                                method: "DELETE", body: nil)
        guard code == 200 else {
            throw ServerError.http(code, String(data: payload, encoding: .utf8) ?? "HTTP \(code)")
        }
    }

    func deleteDiskConditioningCache(scope: String? = nil, id: String? = nil) async throws {
        let query: String
        if let scope, let id {
            guard scope.utf8.count == 16, id.utf8.count == 16,
                  (scope + id).utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
                throw ServerError.http(400, "Invalid cache scope or entry id")
            }
            query = "scope=\(scope)&id=\(id)"
        } else if scope == nil && id == nil {
            query = "all=true"
        } else {
            throw ServerError.http(400, "Invalid cache scope or entry id")
        }
        let (payload, code) = try await request(path: "/sdcpp/v1/conditioning-cache/disk?\(query)", method: "DELETE", body: nil)
        guard code == 200 else {
            throw ServerError.http(code, String(data: payload, encoding: .utf8) ?? "HTTP \(code)")
        }
    }

    /// Returns how many entries were removed.
    func purgeConditioningCache(maxAgeSeconds: Int) async throws -> Int {
        let body = try JSONSerialization.data(withJSONObject: ["max_age_seconds": maxAgeSeconds])
        let (payload, code) = try await request(path: "/sdcpp/v1/conditioning-cache/purge",
                                                method: "POST", body: body)
        guard code == 200,
              let json = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any] else {
            throw ServerError.http(code, String(data: payload, encoding: .utf8) ?? "HTTP \(code)")
        }
        if let n = json["removed"] as? Int { return n }
        if let n = json["removed"] as? NSNumber { return n.intValue }
        return 0
    }

    func stop() {
        stopProcess()
        state = .stopped
        launchKey = ""
        loadedSummary = ""
    }

    func openInBrowser() {
        if !isRunning { start() }
        if let url { NSWorkspace.shared.open(url) }
    }

    nonisolated func shutdown() {
        Task { @MainActor in self.stop() }
    }

    private func waitUntilReady(process p: Process) async throws {
        let deadline = Date().addingTimeInterval(15 * 60)
        while Date() < deadline {
            if Task.isCancelled { throw CancellationError() }
            if process != p || !p.isRunning {
                throw ServerError.exited(String(logText.suffix(500)))
            }
            if let status = try? await fetchStatus(), Self.isListenReady(status: status) {
                state = .running
                return
            }
            try await Task.sleep(nanoseconds: 400_000_000)
        }
        throw ServerError.timeout
    }

    /// Attach UI state to an already-running process whose readiness wait never completed.
    private func reclaimRunningProcess() async {
        guard let p = process, p.isRunning else {
            process = nil
            state = .stopped
            return
        }
        state = .starting
        do {
            try await waitUntilReady(process: p)
        } catch {
            stopProcess()
            state = .failed((error as? LocalizedError)?.errorDescription ?? error.localizedDescription)
        }
    }

    private func fetchStatus() async throws -> String {
        let (payload, code) = try await request(path: "/sdcpp/v1/models/status", method: "GET", body: nil)
        guard code == 200,
              let json = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any],
              let state = json["state"] as? String else {
            throw ServerError.http(code, "status")
        }
        return state
    }

    private func request(path: String, method: String, body: Data?) async throws -> (Data, Int) {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        req.httpMethod = method
        req.httpBody = body
        if body != nil { req.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        req.timeoutInterval = 60 * 30
        let (data, response) = try await URLSession.shared.data(for: req)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (data, code)
    }

    private func stopProcess() {
        guard let p = process else { return }
        let pid = p.processIdentifier
        process = nil
        launchKey = ""
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        if p.isRunning {
            p.terminate()
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                if p.isRunning { kill(pid, SIGKILL) }
            }
        }
        EngineLock.remove(pid: pid)
    }

    private func appendLog(_ text: String) {
        logText = String((logText + text).suffix(80_000))
        onLog?(text)
    }

    private func contextSummary(_ args: [String]) -> String {
        var names: [String] = []
        var i = 0
        let wanted: Set<String> = ["--diffusion-model", "--model", "--llm", "--vae", "--t5xxl", "--clip_l"]
        while i < args.count {
            if wanted.contains(args[i]), i + 1 < args.count {
                names.append(URL(fileURLWithPath: args[i + 1]).lastPathComponent)
                i += 2
            } else {
                i += 1
            }
        }
        return names.joined(separator: " · ")
    }

    private static func fingerprint(_ launch: Launch) -> String {
        let keys = [
            "TOSH_FA_AMD", "TOSH_MM_WIDE_DISABLE", "TOSH_MM_ACC_HALF",
            "GGML_METAL_NCB", "GGML_METAL_DEVICES", "GGML_METAL_DEVICE_LIST",
            "GGML_METAL_DEVICE_INDEX", "GGML_METAL_SHARED_BUFFERS_DISABLE",
        ]
        let env = keys.map { "\($0)=\(launch.env[$0] ?? "")" }.joined(separator: "\n")
        return "\(launch.port)\n\(launch.contextArgs.joined(separator: "\n"))\n\(env)"
    }
}
