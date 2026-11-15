// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import AppKit
import Combine

/// Background `sd-server` (sd.cpp web UI + OpenAI / SDAPI). Shares GPU/VRAM with
/// in-app generation, so it stays off until the user enables it.
@MainActor
final class ImageWebUIServer: ObservableObject {
    static let shared = ImageWebUIServer()

    enum State: Equatable {
        case stopped
        case starting
        case running
        case failed(String)
    }

    @Published private(set) var state: State = .stopped
    @Published private(set) var port: Int = 1234

    private var process: Process?
    private var stderrPipe: Pipe?

    private init() {}

    var isRunning: Bool {
        if case .running = state { return true }
        return false
    }

    var url: URL? {
        guard isRunning else { return nil }
        return URL(string: "http://127.0.0.1:\(port)/")
    }

    static func bundledBinary() -> String? {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("bin-image/sd-server").path,
           FileManager.default.isExecutableFile(atPath: bundled) {
            return bundled
        }
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

    func start() {
        if case .running = state { return }
        if case .starting = state { return }
        guard let bin = Self.bundledBinary() else {
            state = .failed("MISSING_SD_SERVER")
            return
        }
        stopProcess()
        let port = UserDefaults.standard.object(forKey: SettingsKeys.imagenWebUIPort) as? Int ?? 1234
        self.port = max(1, min(port, 65535))
        let modelDir = Self.modelDirectory()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: bin)
        p.arguments = [
            "--model-dir", modelDir.path,
            "-l", "127.0.0.1",
            "--listen-port", String(self.port),
            "--auto-fit", "off",
        ]
        let err = Pipe()
        p.standardOutput = FileHandle.nullDevice
        p.standardError = err
        stderrPipe = err
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                guard let self else { return }
                if self.process == proc {
                    self.process = nil
                    if case .starting = self.state, proc.terminationStatus != 0 {
                        self.state = .failed("EXIT_\(proc.terminationStatus)")
                    } else if case .running = self.state {
                        self.state = .stopped
                    }
                }
            }
        }
        do {
            state = .starting
            process = p
            try p.run()
            // Give the HTTP listener a moment, then mark running (sd-server loads models lazily).
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 800_000_000)
                guard self.process == p, p.isRunning else { return }
                self.state = .running
            }
        } catch {
            process = nil
            state = .failed(error.localizedDescription)
        }
    }

    func stop() {
        stopProcess()
        state = .stopped
    }

    func openInBrowser() {
        if !isRunning { start() }
        if let url { NSWorkspace.shared.open(url) }
    }

    /// Called on app quit so the child is not orphaned with VRAM.
    nonisolated func shutdown() {
        Task { @MainActor in self.stop() }
    }

    private func stopProcess() {
        guard let p = process else { return }
        process = nil
        if p.isRunning {
            p.terminate()
            // Hard kill if it ignores SIGTERM (Metal shutdown can hang).
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                if p.isRunning { p.interrupt() }
            }
        }
    }
}
