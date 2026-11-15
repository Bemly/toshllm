// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AppKit
import Foundation
import ImageIO

/// Disk-backed generation params written next to each `toshllm_*.{png,jpg}`.
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
    var mossTextEncoder: String
    var instanceLabel: String?
    var latentPath: String?
    var fullSteps: Int?
    var endedAtStep: Int?

    var imageURL: URL { URL(fileURLWithPath: imagePath) }

    var latentURL: URL? {
        guard let latentPath, !latentPath.isEmpty else { return nil }
        return URL(fileURLWithPath: latentPath)
    }

    /// True when this record can drive "Import to instance".
    var hasUsableParams: Bool {
        !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || (width > 0 && height > 0 && seed >= 0)
    }

    init(id: UUID = UUID(), imagePath: String, createdAt: Date = Date(),
         prompt: String = "", negativePrompt: String = "", seed: Int = -1, steps: Int = 0,
         width: Int = 0, height: Int = 0, duration: Int = 0, modelID: String = "",
         cfgScale: Double = 0, samplingMethod: String = "", scheduler: String = "",
         flowShift: Double = -1, strength: Double = 0.6, aspect: String = "",
         baseSize: Int = 1024, format: String = "png", initImagePath: String = "",
         mossTextEncoder: String = "", instanceLabel: String? = nil, latentPath: String? = nil,
         fullSteps: Int? = nil, endedAtStep: Int? = nil) {
        self.id = id
        self.imagePath = imagePath
        self.createdAt = createdAt
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.steps = steps
        self.width = width
        self.height = height
        self.duration = duration
        self.modelID = modelID
        self.cfgScale = cfgScale
        self.samplingMethod = samplingMethod
        self.scheduler = scheduler
        self.flowShift = flowShift
        self.strength = strength
        self.aspect = aspect
        self.baseSize = baseSize
        self.format = format
        self.initImagePath = initImagePath
        self.mossTextEncoder = mossTextEncoder
        self.instanceLabel = instanceLabel
        self.latentPath = latentPath
        self.fullSteps = fullSteps
        self.endedAtStep = endedAtStep
    }

    static func sidecar(for imageURL: URL) -> URL {
        URL(fileURLWithPath: imageURL.path + ".json")
    }

    func apply(to cfg: inout ImageInstanceConfig) {
        if !prompt.isEmpty { cfg.prompt = prompt }
        cfg.negativePrompt = negativePrompt
        if seed >= 0 { cfg.seed = seed }
        if steps > 0 { cfg.steps = steps }
        if !modelID.isEmpty { cfg.modelID = modelID }
        if cfgScale > 0 { cfg.cfgScale = cfgScale }
        if !samplingMethod.isEmpty { cfg.samplingMethod = samplingMethod }
        if !scheduler.isEmpty { cfg.scheduler = scheduler }
        cfg.flowShift = flowShift
        if strength > 0 { cfg.strength = strength }
        if baseSize > 0 { cfg.baseSize = baseSize }
        if !format.isEmpty { cfg.format = format }
        if !initImagePath.isEmpty { cfg.initImagePath = initImagePath }
        if !mossTextEncoder.isEmpty { cfg.mossTextEncoder = mossTextEncoder }
        if width > 0 && height > 0 {
            cfg.aspect = ImageAspect.custom.rawValue
            cfg.customUsesPixels = true
            cfg.customWidth = width
            cfg.customHeight = height
        } else if !aspect.isEmpty {
            cfg.aspect = aspect
        }
    }

    /// Prefer richer fields when merging a recovered record over a sparse index entry.
    func merging(better: ImageGalleryRecord) -> ImageGalleryRecord {
        var out = self
        if better.hasUsableParams && (!hasUsableParams || better.prompt.count > prompt.count) {
            out.prompt = better.prompt.isEmpty ? out.prompt : better.prompt
            out.negativePrompt = better.negativePrompt.isEmpty ? out.negativePrompt : better.negativePrompt
            if better.seed >= 0 { out.seed = better.seed }
            if better.steps > 0 { out.steps = better.steps }
            if better.cfgScale > 0 { out.cfgScale = better.cfgScale }
            if !better.modelID.isEmpty { out.modelID = better.modelID }
            if !better.samplingMethod.isEmpty { out.samplingMethod = better.samplingMethod }
            if !better.scheduler.isEmpty { out.scheduler = better.scheduler }
            if better.flowShift != -1 { out.flowShift = better.flowShift }
            if !better.mossTextEncoder.isEmpty { out.mossTextEncoder = better.mossTextEncoder }
            if better.fullSteps != nil { out.fullSteps = better.fullSteps }
            if better.endedAtStep != nil { out.endedAtStep = better.endedAtStep }
        }
        if better.width > 0 { out.width = better.width }
        if better.height > 0 { out.height = better.height }
        if let path = better.latentPath, !path.isEmpty { out.latentPath = path }
        if let label = better.instanceLabel, !label.isEmpty { out.instanceLabel = label }
        return out
    }
}

enum ImageGalleryStore {
    static let maxRecords = 200

    /// Resolve `/var` ↔ `/private/var` (and similar) so index lookups match file scans.
    /// `resolvingSymlinksInPath` only rewrites `/var` when the target exists; missing
    /// paths still need the macOS prefix normalized so upsert/scan keys match.
    static func canonicalPath(_ path: String) -> String {
        let url = URL(fileURLWithPath: path)
        if FileManager.default.fileExists(atPath: url.path) {
            return url.resolvingSymlinksInPath().path
        }
        var p = url.standardizedFileURL.path
        if p.hasPrefix("/var/") { p = "/private" + p }
        return p
    }

    static func canonicalURL(_ url: URL) -> URL {
        URL(fileURLWithPath: canonicalPath(url.path))
    }

    static func indexURL(in imagenDirectory: URL) -> URL {
        imagenDirectory.appendingPathComponent("gallery-index.json")
    }

    static func writeSidecar(_ record: ImageGalleryRecord) {
        let url = ImageGalleryRecord.sidecar(for: record.imageURL)
        let enc = JSONEncoder()
        enc.dateEncodingStrategy = .iso8601
        enc.outputFormatting = [.sortedKeys]
        guard let data = try? enc.encode(record) else { return }
        try? data.write(to: url, options: .atomic)
    }

    static func loadSidecar(for imageURL: URL) -> ImageGalleryRecord? {
        let url = ImageGalleryRecord.sidecar(for: imageURL)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return decodeRecord(data)
    }

    static func saveIndex(_ records: [ImageGalleryRecord], in imagenDirectory: URL) {
        try? FileManager.default.createDirectory(at: imagenDirectory, withIntermediateDirectories: true)
        let trimmed = Array(records.prefix(maxRecords))
        let enc = JSONEncoder()
        enc.dateEncodingStrategy = .iso8601
        enc.outputFormatting = [.sortedKeys]
        guard let data = try? enc.encode(trimmed) else { return }
        try? data.write(to: indexURL(in: imagenDirectory), options: .atomic)
    }

    static func loadIndex(in imagenDirectory: URL) -> [ImageGalleryRecord] {
        let url = indexURL(in: imagenDirectory)
        guard let data = try? Data(contentsOf: url),
              let records = decodeRecords(data)
        else { return [] }
        return records.compactMap { raw -> ImageGalleryRecord? in
            var r = raw
            r.imagePath = canonicalPath(r.imagePath)
            guard FileManager.default.fileExists(atPath: r.imagePath) else { return nil }
            return r
        }
    }

    /// Upsert one record into the on-disk library index without wiping others.
    static func upsert(_ record: ImageGalleryRecord, in imagenDirectory: URL) {
        var normalized = record
        normalized.imagePath = canonicalPath(record.imagePath)
        if let latent = record.latentPath, !latent.isEmpty {
            normalized.latentPath = canonicalPath(latent)
        }
        var byPath: [String: ImageGalleryRecord] = [:]
        for existing in loadIndex(in: imagenDirectory) {
            byPath[canonicalPath(existing.imagePath)] = existing
        }
        if let old = byPath[normalized.imagePath] {
            byPath[normalized.imagePath] = old.merging(better: normalized)
        } else {
            byPath[normalized.imagePath] = normalized
        }
        let sorted = byPath.values.sorted { $0.createdAt > $1.createdAt }
        saveIndex(sorted, in: imagenDirectory)
        writeSidecar(byPath[normalized.imagePath] ?? normalized)
    }

    /// Rebuild gallery from index + image files, recovering params from sidecars / latent JSON / pixels.
    static func scan(imagenDirectory: URL) -> [ImageGalleryRecord] {
        var byPath: [String: ImageGalleryRecord] = [:]
        for record in loadIndex(in: imagenDirectory) {
            let key = canonicalPath(record.imagePath)
            var enriched = enrich(record)
            enriched.imagePath = key
            byPath[key] = enriched
        }
        let root = canonicalURL(imagenDirectory)
        let files = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles])) ?? []
        for file in files {
            let resolved = canonicalURL(file)
            let name = resolved.deletingPathExtension().lastPathComponent
            let ext = resolved.pathExtension.lowercased()
            guard name.hasPrefix("toshllm_"),
                  ["png", "jpg", "jpeg", "webp"].contains(ext),
                  !name.contains("-x2"), !name.contains("-x4") else { continue }
            let key = resolved.path
            let recovered = record(forImage: resolved) ?? ImageGalleryRecord(
                id: UUID(), imagePath: key,
                createdAt: (try? resolved.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? Date(),
                prompt: "", negativePrompt: "", seed: -1, steps: 0,
                width: 0, height: 0, duration: 0, modelID: "",
                cfgScale: 0, samplingMethod: "", scheduler: "", flowShift: -1,
                strength: 0.6, aspect: "", baseSize: 1024, format: ext,
                initImagePath: "", mossTextEncoder: "",
                instanceLabel: nil, latentPath: nil, fullSteps: nil, endedAtStep: nil)
            var enriched = enrich(recovered)
            enriched.imagePath = key
            if let old = byPath[key] {
                byPath[key] = old.merging(better: enriched)
            } else {
                byPath[key] = enriched
            }
        }
        let sorted = byPath.values.sorted { $0.createdAt > $1.createdAt }
        // Persist repaired records so the next launch is not empty again.
        saveIndex(sorted, in: imagenDirectory)
        for record in sorted where record.hasUsableParams {
            writeSidecar(record)
        }
        return sorted
    }

    static func delete(_ record: ImageGalleryRecord) {
        try? FileManager.default.removeItem(at: record.imageURL)
        try? FileManager.default.removeItem(at: ImageGalleryRecord.sidecar(for: record.imageURL))
        if let latent = record.latentURL {
            try? FileManager.default.removeItem(at: latent)
            try? FileManager.default.removeItem(at: URL(fileURLWithPath: latent.path + ".json"))
        }
    }

    static func pixelSize(of imageURL: URL) -> (Int, Int)? {
        guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int,
              w > 0, h > 0
        else { return nil }
        return (w, h)
    }

    static func pixelSize(of image: NSImage) -> (Int, Int)? {
        guard let rep = image.representations.first,
              rep.pixelsWide > 0, rep.pixelsHigh > 0 else {
            return nil
        }
        return (rep.pixelsWide, rep.pixelsHigh)
    }

    // MARK: - recovery

    static func record(forImage file: URL) -> ImageGalleryRecord? {
        if let sidecar = loadSidecar(for: file), sidecar.hasUsableParams || sidecar.width > 0 {
            return enrich(sidecar)
        }
        if let fromLatent = recordFromLatentSidecar(for: file) {
            return enrich(fromLatent)
        }
        return nil
    }

    static func recordFromLatentSidecar(for imageURL: URL) -> ImageGalleryRecord? {
        let latent = imageURL.deletingPathExtension().appendingPathExtension("sdlt")
        let meta = URL(fileURLWithPath: latent.path + ".json")
        guard let data = try? Data(contentsOf: meta),
              let handoff = try? JSONDecoder().decode(ImageLatentHandoff.self, from: data)
        else { return nil }
        let values = try? imageURL.resourceValues(forKeys: [.contentModificationDateKey])
        return ImageGalleryRecord(
            id: UUID(), imagePath: imageURL.path,
            createdAt: values?.contentModificationDate ?? Date(),
            prompt: handoff.prompt, negativePrompt: handoff.negativePrompt,
            seed: handoff.seed, steps: handoff.fullSteps,
            width: handoff.width, height: handoff.height, duration: 0,
            modelID: handoff.modelID, cfgScale: handoff.cfgScale,
            samplingMethod: handoff.sampler, scheduler: handoff.scheduler,
            flowShift: handoff.flowShift, strength: 0.6, aspect: ImageAspect.custom.rawValue,
            baseSize: max(handoff.width, handoff.height), format: imageURL.pathExtension.lowercased(),
            initImagePath: "", mossTextEncoder: handoff.textEncoder,
            instanceLabel: nil,
            latentPath: FileManager.default.fileExists(atPath: latent.path) ? latent.path : nil,
            fullSteps: handoff.fullSteps, endedAtStep: handoff.endedAtStep)
    }

    static func enrich(_ record: ImageGalleryRecord) -> ImageGalleryRecord {
        var out = record
        if out.width <= 0 || out.height <= 0,
           let (w, h) = pixelSize(of: out.imageURL) {
            out.width = w
            out.height = h
        }
        if !out.hasUsableParams, let recovered = recordFromLatentSidecar(for: out.imageURL) {
            out = out.merging(better: recovered)
        }
        if (out.width <= 0 || out.height <= 0),
           let (w, h) = pixelSize(of: out.imageURL) {
            out.width = w
            out.height = h
        }
        if out.latentPath == nil {
            let latent = out.imageURL.deletingPathExtension().appendingPathExtension("sdlt")
            if FileManager.default.fileExists(atPath: latent.path) {
                out.latentPath = latent.path
            }
        }
        return out
    }

    // MARK: - decode (tolerant of missing mossTextEncoder on old files)

    private static func decodeRecord(_ data: Data) -> ImageGalleryRecord? {
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .custom { decoder in
            let c = try decoder.singleValueContainer()
            if let d = try? c.decode(Double.self) { return Date(timeIntervalSinceReferenceDate: d) }
            if let s = try? c.decode(String.self) {
                let iso = ISO8601DateFormatter()
                if let d = iso.date(from: s) { return d }
            }
            throw DecodingError.dataCorruptedError(in: c, debugDescription: "bad date")
        }
        if let r = try? dec.decode(ImageGalleryRecord.self, from: data) { return r }
        // Fallback without custom date strategy (reference-date Doubles from default encoder).
        return try? JSONDecoder().decode(ImageGalleryRecord.self, from: data)
    }

    private static func decodeRecords(_ data: Data) -> [ImageGalleryRecord]? {
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .custom { decoder in
            let c = try decoder.singleValueContainer()
            if let d = try? c.decode(Double.self) { return Date(timeIntervalSinceReferenceDate: d) }
            if let s = try? c.decode(String.self) {
                let iso = ISO8601DateFormatter()
                if let d = iso.date(from: s) { return d }
            }
            throw DecodingError.dataCorruptedError(in: c, debugDescription: "bad date")
        }
        if let r = try? dec.decode([ImageGalleryRecord].self, from: data) { return r }
        return try? JSONDecoder().decode([ImageGalleryRecord].self, from: data)
    }
}

// Make mossTextEncoder optional for older sidecars.
extension ImageGalleryRecord {
    enum CodingKeys: String, CodingKey {
        case id, imagePath, createdAt, prompt, negativePrompt, seed, steps
        case width, height, duration, modelID, cfgScale, samplingMethod, scheduler
        case flowShift, strength, aspect, baseSize, format, initImagePath
        case mossTextEncoder, instanceLabel, latentPath, fullSteps, endedAtStep
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decode(UUID.self, forKey: .id)) ?? UUID()
        imagePath = (try? c.decode(String.self, forKey: .imagePath)) ?? ""
        createdAt = (try? c.decode(Date.self, forKey: .createdAt)) ?? Date()
        prompt = (try? c.decode(String.self, forKey: .prompt)) ?? ""
        negativePrompt = (try? c.decode(String.self, forKey: .negativePrompt)) ?? ""
        seed = (try? c.decode(Int.self, forKey: .seed)) ?? -1
        steps = (try? c.decode(Int.self, forKey: .steps)) ?? 0
        width = (try? c.decode(Int.self, forKey: .width)) ?? 0
        height = (try? c.decode(Int.self, forKey: .height)) ?? 0
        duration = (try? c.decode(Int.self, forKey: .duration)) ?? 0
        modelID = (try? c.decode(String.self, forKey: .modelID)) ?? ""
        cfgScale = (try? c.decode(Double.self, forKey: .cfgScale)) ?? 0
        samplingMethod = (try? c.decode(String.self, forKey: .samplingMethod)) ?? ""
        scheduler = (try? c.decode(String.self, forKey: .scheduler)) ?? ""
        flowShift = (try? c.decode(Double.self, forKey: .flowShift)) ?? -1
        strength = (try? c.decode(Double.self, forKey: .strength)) ?? 0.6
        aspect = (try? c.decode(String.self, forKey: .aspect)) ?? ""
        baseSize = (try? c.decode(Int.self, forKey: .baseSize)) ?? 1024
        format = (try? c.decode(String.self, forKey: .format)) ?? "png"
        initImagePath = (try? c.decode(String.self, forKey: .initImagePath)) ?? ""
        mossTextEncoder = (try? c.decode(String.self, forKey: .mossTextEncoder)) ?? ""
        instanceLabel = try? c.decode(String.self, forKey: .instanceLabel)
        latentPath = try? c.decode(String.self, forKey: .latentPath)
        fullSteps = try? c.decode(Int.self, forKey: .fullSteps)
        endedAtStep = try? c.decode(Int.self, forKey: .endedAtStep)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(imagePath, forKey: .imagePath)
        try c.encode(createdAt, forKey: .createdAt)
        try c.encode(prompt, forKey: .prompt)
        try c.encode(negativePrompt, forKey: .negativePrompt)
        try c.encode(seed, forKey: .seed)
        try c.encode(steps, forKey: .steps)
        try c.encode(width, forKey: .width)
        try c.encode(height, forKey: .height)
        try c.encode(duration, forKey: .duration)
        try c.encode(modelID, forKey: .modelID)
        try c.encode(cfgScale, forKey: .cfgScale)
        try c.encode(samplingMethod, forKey: .samplingMethod)
        try c.encode(scheduler, forKey: .scheduler)
        try c.encode(flowShift, forKey: .flowShift)
        try c.encode(strength, forKey: .strength)
        try c.encode(aspect, forKey: .aspect)
        try c.encode(baseSize, forKey: .baseSize)
        try c.encode(format, forKey: .format)
        try c.encode(initImagePath, forKey: .initImagePath)
        try c.encode(mossTextEncoder, forKey: .mossTextEncoder)
        try c.encodeIfPresent(instanceLabel, forKey: .instanceLabel)
        try c.encodeIfPresent(latentPath, forKey: .latentPath)
        try c.encodeIfPresent(fullSteps, forKey: .fullSteps)
        try c.encodeIfPresent(endedAtStep, forKey: .endedAtStep)
    }
}
