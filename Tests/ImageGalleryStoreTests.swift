// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AppKit
import XCTest
@testable import ToshLLM

final class ImageGalleryStoreTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-gallery-\(UUID().uuidString)", isDirectory: true)
            .resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    func testUpsertWritesSidecarAndIndexWithRealParams() throws {
        let image = try writeTinyJPEG(named: "toshllm_test_1.jpg")
        let record = ImageGalleryRecord(
            imagePath: image.path, prompt: "a fox in snow", negativePrompt: "blurry",
            seed: 42, steps: 8, width: 1024, height: 1536, modelID: "MOSS-Image (Q8)",
            cfgScale: 3, samplingMethod: "euler", scheduler: "sgm_uniform",
            mossTextEncoder: "q8")
        ImageGalleryStore.upsert(record, in: dir)

        let sidecar = try XCTUnwrap(ImageGalleryStore.loadSidecar(for: image))
        XCTAssertEqual(sidecar.prompt, "a fox in snow")
        XCTAssertEqual(sidecar.width, 1024)
        XCTAssertEqual(sidecar.height, 1536)
        XCTAssertEqual(sidecar.seed, 42)
        XCTAssertTrue(sidecar.hasUsableParams)

        let index = ImageGalleryStore.loadIndex(in: dir)
        XCTAssertEqual(index.count, 1)
        XCTAssertEqual(index[0].prompt, "a fox in snow")
        XCTAssertEqual(index[0].imagePath, ImageGalleryStore.canonicalPath(image.path))
    }

    func testScanRecoversParamsFromLatentJSONWhenGallerySidecarMissing() throws {
        let image = try writeTinyJPEG(named: "toshllm_recover.jpg")
        let latent = image.deletingPathExtension().appendingPathExtension("sdlt")
        // Touch empty latent so the path exists.
        try Data().write(to: latent)
        let handoff = ImageLatentHandoff(
            startAtStep: 0, endAtStep: 8, addNoise: true, latentInPath: "",
            writeLatent: true, fullSteps: 20, endedAtStep: 8, seed: 99,
            sampler: "euler", scheduler: "simple", flowShift: 5, cfgScale: 3,
            modelID: "MOSS-Image (Q8)", textEncoder: "animus",
            width: 768, height: 1152,
            prompt: "green hair girl", negativePrompt: "lowres")
        let meta = URL(fileURLWithPath: latent.path + ".json")
        try JSONEncoder().encode(handoff).write(to: meta)

        // Poison the index with an empty 0×0 entry (the bug users hit).
        let empty = ImageGalleryRecord(imagePath: image.path, prompt: "", width: 0, height: 0)
        ImageGalleryStore.saveIndex([empty], in: dir)

        let scanned = ImageGalleryStore.scan(imagenDirectory: dir)
        let key = ImageGalleryStore.canonicalPath(image.path)
        let hit = try XCTUnwrap(scanned.first(where: { $0.imagePath == key }))
        XCTAssertEqual(hit.prompt, "green hair girl")
        XCTAssertEqual(hit.seed, 99)
        XCTAssertEqual(hit.width, 768)
        XCTAssertEqual(hit.height, 1152)
        XCTAssertEqual(hit.mossTextEncoder, "animus")
        XCTAssertTrue(hit.hasUsableParams)

        var cfg = ImageInstanceConfig()
        hit.apply(to: &cfg)
        XCTAssertEqual(cfg.prompt, "green hair girl")
        XCTAssertEqual(cfg.seed, 99)
        XCTAssertEqual(cfg.customWidth, 768)
        XCTAssertEqual(cfg.customHeight, 1152)
        XCTAssertEqual(cfg.mossTextEncoder, "animus")
        XCTAssertEqual(cfg.aspect, ImageAspect.custom.rawValue)
    }

    func testScanFillsPixelSizeWhenNoMetadata() throws {
        let image = try writeTinyJPEG(named: "toshllm_pixels_only.jpg")
        let scanned = ImageGalleryStore.scan(imagenDirectory: dir)
        let key = ImageGalleryStore.canonicalPath(image.path)
        let hit = try XCTUnwrap(scanned.first(where: { $0.imagePath == key }))
        XCTAssertGreaterThan(hit.width, 0)
        XCTAssertGreaterThan(hit.height, 0)
    }

    func testUpsertDoesNotWipeSiblingLibraryEntries() throws {
        let a = try writeTinyJPEG(named: "toshllm_a.jpg")
        let b = try writeTinyJPEG(named: "toshllm_b.jpg")
        ImageGalleryStore.upsert(
            ImageGalleryRecord(imagePath: a.path, prompt: "one", seed: 1, width: 64, height: 64),
            in: dir)
        ImageGalleryStore.upsert(
            ImageGalleryRecord(imagePath: b.path, prompt: "two", seed: 2, width: 64, height: 64),
            in: dir)
        XCTAssertEqual(ImageGalleryStore.loadIndex(in: dir).count, 2)
    }

    func testMossTextEncoderCasesIncludeAnimusAndHeretic() {
        let labels = ImageMossTextEncoder.allCases.map(\.shortLabel)
        XCTAssertEqual(labels, ["Q8", "Animus", "Heretic"])
        XCTAssertEqual(ImageMossTextEncoder.q8.component.fileName,
                       "Qwen3.5-9B-Q8_0.gguf")
        XCTAssertEqual(ImageMossTextEncoder.animus.component.fileName,
                       "Qwen3.5-9B-Animus-V13.0-Q8_0.gguf")
        XCTAssertEqual(ImageMossTextEncoder.heretic.component.fileName,
                       "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q8_0.gguf")
        for encoder in ImageMossTextEncoder.allCases {
            XCTAssertEqual(URL(string: encoder.component.urlString)?.lastPathComponent,
                           encoder.component.fileName)
        }
    }

    func testMossBaseF16SavedIDResolvesToQ8() {
        XCTAssertTrue(ImageGenCatalog.isMossBase("MOSS-Image (F16)"))
        XCTAssertEqual(ImageGenCatalog.model(id: "MOSS-Image (F16)")?.id,
                       ImageGenCatalog.mossImageBaseQ8.id)
        XCTAssertFalse(ImageGenCatalog.models.contains { $0.id == "MOSS-Image (F16)" })
    }

    func testCanonicalPathUnifiesVarPrivateSymlink() {
        let a = ImageGalleryStore.canonicalPath("/var/folders/test/file.jpg")
        let b = ImageGalleryStore.canonicalPath("/private/var/folders/test/file.jpg")
        XCTAssertTrue(a.hasSuffix("/folders/test/file.jpg"))
        XCTAssertEqual(a, b)
    }

    func testGeneratedImageResolvedSizeFallsBackToPixels() throws {
        let image = try writeTinyJPEG(named: "toshllm_size.jpg")
        let record = ImageGalleryRecord(imagePath: image.path, prompt: "x", width: 0, height: 0)
        let generated = try XCTUnwrap(GeneratedImage.from(record: record))
        let (w, h) = generated.resolvedSize
        XCTAssertGreaterThan(w, 0)
        XCTAssertGreaterThan(h, 0)
        XCTAssertNotEqual(generated.sizeLabel, "—")
        XCTAssertFalse(generated.sizeLabel.contains("0×0"))
    }

    private func writeTinyJPEG(named name: String) throws -> URL {
        let url = dir.appendingPathComponent(name)
        let image = NSImage(size: NSSize(width: 32, height: 48))
        image.lockFocus()
        NSColor.red.setFill()
        NSBezierPath(rect: NSRect(x: 0, y: 0, width: 32, height: 48)).fill()
        image.unlockFocus()
        let tiff = try XCTUnwrap(image.tiffRepresentation)
        let rep = try XCTUnwrap(NSBitmapImageRep(data: tiff))
        let data = try XCTUnwrap(rep.representation(using: .jpeg, properties: [:]))
        try data.write(to: url)
        return url.resolvingSymlinksInPath()
    }
}
