// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import AppKit

extension ImageGenerator {
    func runOnServer(args: [String], env: [String: String], output: URL, steps: Int) async {
        let server = ImageWebUIServer.shared
        let port = max(1, min(UserDefaults.standard.object(forKey: SettingsKeys.imagenWebUIPort) as? Int ?? 1234, 65535))
        let split = ImageServerHTTP.partition(args)
        server.onLog = { [weak self] text in
            Task { @MainActor in self?.noteServerLog(text, steps: steps) }
        }
        var jobBegan = false
        func cleanupRun() {
            if jobBegan { server.endJob() }
            server.onLog = nil
            activeJobID = nil
            runTask = nil
            cancelRequested = false
        }
        do {
            try await server.ensure(.init(contextArgs: split.context, env: env, port: port))
            server.beginJob()
            jobBegan = true
            if cancelRequested {
                cleanupRun()
                finish(status: 15, output: output)
                return
            }
            let body = try ImageServerHTTP.requestBody(from: split.values, bools: split.bools, output: output)
            let id = try await server.submit(body)
            activeJobID = id
            fileLog.append("POST /sdcpp/v1/img_gen \(id)\n")
            var lastPreview = ""
            var cancellationSent = false
            while true {
                if cancelRequested && !cancellationSent {
                    do {
                        try await server.cancelJob(id)
                        cancellationSent = true
                    } catch {
                        cancelError = error.localizedDescription
                        fileLog.append("Cancellation failed: \(error.localizedDescription)\n")
                        cancelRequested = false
                    }
                }
                let job = try await server.poll(id)
                apply(job, steps: steps, lastPreview: &lastPreview)
                switch job.status {
                case "completed":
                    try writeImages(job.imagesB64, output: output)
                    cleanupRun()
                    finish(status: 0, output: output)
                    return
                case "failed":
                    logTail += job.error ?? "failed"
                    cleanupRun()
                    finish(status: 1, output: output)
                    return
                case "cancelled":
                    cleanupRun()
                    finish(status: 15, output: output)
                    return
                default:
                    try await Task.sleep(nanoseconds: 350_000_000)
                }
            }
        } catch {
            logTail += error.localizedDescription
            fileLog.append(error.localizedDescription + "\n")
            cleanupRun()
            state = .failed(error.localizedDescription)
            onFinish?()
        }
    }

    private func noteServerLog(_ text: String, steps: Int) {
        fileLog.append(text)
        logTail = String((logTail + text).suffix(4000))
        if text.contains("conditioning cache hit") || text.contains("reloading text encoder") {
            consume(text, steps: steps)
        }
    }

    private func apply(_ job: ImageWebUIServer.JobSnapshot, steps: Int, lastPreview: inout String) {
        if job.stage == "decoding" || job.stage == "encoding" { stage = .decoding }
        if let done = job.completed, let total = job.total, total > 0, job.stage == "sampling" {
            if firstStepAt == nil { firstStepAt = Date() }
            stage = .sampling
            progress = Double(done) / Double(total)
            stepText = "\(done)/\(total)"
        } else if stage == .loading, job.status == "generating" {
            stage = .sampling
        }
        if let b64 = job.previewB64, b64 != lastPreview, let data = Data(base64Encoded: b64),
           let img = NSImage(data: data) {
            lastPreview = b64
            previewImage = img
        }
        _ = steps
    }

    private func writeImages(_ images: [String], output: URL) throws {
        if images.isEmpty { throw ImageWebUIServer.ServerError.http(500, "no images") }
        for (i, b64) in images.enumerated() {
            guard let data = Data(base64Encoded: b64) else { continue }
            let path = pendingOutputPattern.map { String(format: $0, i) } ?? output.path
            try data.write(to: URL(fileURLWithPath: path))
        }
    }
}
