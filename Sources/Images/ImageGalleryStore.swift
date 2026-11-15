// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AppKit
import Foundation

/// Disk-backed generation params written next to each `toshllm_*.png`.
struct ImageGalleryRecord: Codable, Equatable, Identifiable {
    var id: UUID
    var imagePath: String
    var createdAt: Date
    var prompt: String
    var negativePrompt: String
    var seed: Int
    var steps: Int
    var width: Int
    var height: Int
    var duration: Int
    var modelID: String
    var cfgScale: Double
    var samplingMethod: String
    var scheduler: String
    var flowShift: Double
    var strength: Double
    var aspect: String
    var baseSize: Int
    var format: String
    var initImagePath: String
    var instanceLabel: String?
    var latentPath: String?
    var fullSteps: Int?
    var endedAtStep: Int?

    var imageURL: URL { URL(fileURLWithPath: imagePath) }
    var sidecarURL: URL { imageURL.appendingPathExtension("json") }
    var latentURL: URL? {
        guard let latentPath, !latentPath.isEmpty else { return nil }
        return URL(fileURLWithPath: latentPath)
    }

    static func sidecar(for imageURL: URL) -> URL {
        URL(fileURLWithPath: imageURL.path + ".json")
    }

    func apply(to cfg: inout ImageInstanceConfig) {
        cfg.prompt = prompt
        cfg.negativePrompt = negativePrompt
        cfg.seed = seed
        if steps > 0 { cfg.steps = steps }
        if width > 0 { cfg.customWidth = width }
        if height > 0 { cfg.customHeight = height }
        if !modelID.isEmpty { cfg.modelID = modelID }
        if cfgScale > 0 { cfg.cfgScale = cfgScale }
        if !samplingMethod.isEmpty { cfg.samplingMethod = samplingMethod }
        if !scheduler.isEmpty { cfg.scheduler = scheduler }
        cfg.flowShift = flowShift
        if strength > 0 { cfg.strength = strength }
        if !aspect.isEmpty { cfg.aspect = aspect }
        if baseSize > 0 { cfg.baseSize = baseSize }
        if !format.isEmpty { cfg.format = format }
        if !initImagePath.isEmpty { cfg.initImagePath = initImagePath }
        if width > 0 && height > 0 {
            cfg.aspect = ImageAspect.custom.rawValue
            cfg.customUsesPixels = true
            cfg.customWidth = width
            cfg.customHeight = height
        }
    }
}

enum ImageGalleryStore {
    static let maxRecords = 200

    static func indexURL(in imagenDirectory: URL) -> URL {
        imagenDirectory.appendingPathComponent("gallery-index.json")
    }

    static func writeSidecar(_ record: ImageGalleryRecord) {
        let url = ImageGalleryRecord.sidecar(for: record.imageURL)
        guard let data = try? JSONEncoder().encode(record) else { return }
        try? data.write(to: url, options: .atomic)
    }

    static func loadSidecar(for imageURL: URL) -> ImageGalleryRecord? {
        let url = ImageGalleryRecord.sidecar(for: imageURL)
        guard let data = try? Data(contentsOf: url),
              let record = try? JSONDecoder().decode(ImageGalleryRecord.self, from: data)
        else { return nil }
        return record
    }

    static func saveIndex(_ records: [ImageGalleryRecord], in imagenDirectory: URL) {
        try? FileManager.default.createDirectory(at: imagenDirectory, withIntermediateDirectories: true)
        let trimmed = Array(records.prefix(maxRecords))
        guard let data = try? JSONEncoder().encode(trimmed) else { return }
        try? data.write(to: indexURL(in: imagenDirectory), options: .atomic)
    }

    static func loadIndex(in imagenDirectory: URL) -> [ImageGalleryRecord] {
        let url = indexURL(in: imagenDirectory)
        guard let data = try? Data(contentsOf: url),
              let records = try? JSONDecoder().decode([ImageGalleryRecord].self, from: data)
        else { return [] }
        return records.filter { FileManager.default.fileExists(atPath: $0.imagePath) }
    }

    /// Rebuild gallery from index + orphaned `toshllm_*` files with sidecars.
    static func scan(imagenDirectory: URL) -> [ImageGalleryRecord] {
        var byPath: [String: ImageGalleryRecord] = [:]
        for record in loadIndex(in: imagenDirectory) {
            byPath[record.imagePath] = record
        }
        let files = (try? FileManager.default.contentsOfDirectory(
            at: imagenDirectory, includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles])) ?? []
        for file in files {
            let name = file.deletingPathExtension().lastPathComponent
            let ext = file.pathExtension.lowercased()
            guard name.hasPrefix("toshllm_"),
                  ["png", "jpg", "jpeg", "webp"].contains(ext),
                  !name.contains("-x2"), !name.contains("-x4") else { continue }
            if byPath[file.path] != nil { continue }
            if let sidecar = loadSidecar(for: file) {
                byPath[file.path] = sidecar
                continue
            }
            // Minimal record when only the image remains.
            let values = try? file.resourceValues(forKeys: [.contentModificationDateKey])
            byPath[file.path] = ImageGalleryRecord(
                id: UUID(), imagePath: file.path,
                createdAt: values?.contentModificationDate ?? Date(),
                prompt: "", negativePrompt: "", seed: -1, steps: 0,
                width: 0, height: 0, duration: 0, modelID: "",
                cfgScale: 0, samplingMethod: "", scheduler: "", flowShift: -1,
                strength: 0.6, aspect: "", baseSize: 1024, format: ext,
                initImagePath: "", instanceLabel: nil, latentPath: nil,
                fullSteps: nil, endedAtStep: nil)
        }
        return byPath.values.sorted { $0.createdAt > $1.createdAt }
    }

    static func delete(_ record: ImageGalleryRecord) {
        try? FileManager.default.removeItem(at: record.imageURL)
        try? FileManager.default.removeItem(at: ImageGalleryRecord.sidecar(for: record.imageURL))
        if let latent = record.latentURL {
            try? FileManager.default.removeItem(at: latent)
            try? FileManager.default.removeItem(at: URL(fileURLWithPath: latent.path + ".json"))
        }
    }
}
