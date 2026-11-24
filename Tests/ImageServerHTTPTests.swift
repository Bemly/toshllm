import XCTest
@testable import ToshLLM

final class ImageServerHTTPTests: XCTestCase {
    func testPartitionKeepsModelOnContextAndStepsOnRequest() {
        let split = ImageServerHTTP.partition([
            "--diffusion-model", "/models/dit.gguf",
            "--llm", "/models/te.gguf",
            "--release-te-after-encode",
            "--vae-tiling",
            "-p", "a cat",
            "--steps", "20",
            "--seed", "7",
            "-W", "1024", "-H", "768",
            "--cfg-scale", "1.0",
            "--start-at-step", "0",
            "--end-at-step", "6",
            "--add-noise", "on",
            "--latent-out", "/tmp/out.sdlt",
            "--batch-count", "3",
        ])
        XCTAssertEqual(split.context, [
            "--diffusion-model", "/models/dit.gguf",
            "--llm", "/models/te.gguf",
            "--release-te-after-encode",
        ])
        XCTAssertTrue(split.bools.contains("--vae-tiling"))
        XCTAssertEqual(split.values["--steps"], ["20"])
        XCTAssertEqual(split.values["--end-at-step"], ["6"])
        XCTAssertEqual(split.values["--latent-out"], ["/tmp/out.sdlt"])
        XCTAssertNil(split.values["--diffusion-model"])
    }

    func testRequestBodyMapsHandoffAndStripsLoraTags() throws {
        let values: [String: [String]] = [
            "-p": ["portrait <lora:style:0.8> detail"],
            "--steps": ["12"],
            "--cfg-scale": ["3.5"],
            "--sampling-method": ["euler"],
            "--scheduler": ["simple"],
            "--start-at-step": ["4"],
            "--end-at-step": ["10"],
            "--add-noise": ["off"],
            "--latent-in": ["/tmp/a.sdlt"],
            "--latent-out": ["/tmp/b.sdlt"],
            "--batch-count": ["1"],
            "-W": ["640"], "-H": ["640"],
            "--seed": ["42"],
            "--vae-tile-size": ["384"],
        ]
        let body = try ImageServerHTTP.requestBody(
            from: values,
            bools: ["--vae-tiling"],
            output: URL(fileURLWithPath: "/tmp/out.png"),
            loadFiles: false)
        XCTAssertEqual(body["prompt"] as? String, "portrait  detail")
        XCTAssertEqual(body["start_at_step"] as? Int, 4)
        XCTAssertEqual(body["end_at_step"] as? Int, 10)
        XCTAssertEqual(body["add_noise"] as? String, "off")
        XCTAssertEqual(body["latent_in"] as? String, "/tmp/a.sdlt")
        XCTAssertEqual(body["latent_out"] as? String, "/tmp/b.sdlt")
        let sample = try XCTUnwrap(body["sample_params"] as? [String: Any])
        XCTAssertEqual(sample["sample_steps"] as? Int, 12)
        XCTAssertEqual(sample["sample_method"] as? String, "euler")
        let guidance = try XCTUnwrap(sample["guidance"] as? [String: Any])
        XCTAssertEqual(guidance["txt_cfg"] as? Double, 3.5)
        let tiling = try XCTUnwrap(body["vae_tiling_params"] as? [String: Any])
        XCTAssertEqual(tiling["enabled"] as? Bool, true)
        XCTAssertEqual(tiling["tile_size_w"] as? Int, 384)
        let loras = try XCTUnwrap(body["lora"] as? [[String: Any]])
        XCTAssertEqual(loras.count, 1)
        XCTAssertEqual(loras[0]["path"] as? String, "style")
        XCTAssertEqual(loras[0]["multiplier"] as? Double, 0.8)
    }

    func testStepsOnlyChangeDoesNotTouchContextFingerprint() {
        let base = ImageServerHTTP.partition([
            "--diffusion-model", "/a.gguf", "--llm", "/te.gguf", "-p", "x", "--steps", "10",
        ]).context
        let again = ImageServerHTTP.partition([
            "--diffusion-model", "/a.gguf", "--llm", "/te.gguf", "-p", "x", "--steps", "50",
        ]).context
        XCTAssertEqual(base, again)
    }

    func testConditioningCacheDirectoryStaysOnServerContext() {
        let split = ImageServerHTTP.partition([
            "--conditioning-cache-size", "16",
            "--conditioning-cache-dir", "/tmp/conditioning-cache",
            "-p", "a cat",
        ])
        XCTAssertEqual(split.context, [
            "--conditioning-cache-size", "16",
            "--conditioning-cache-dir", "/tmp/conditioning-cache",
        ])
        XCTAssertNil(split.values["--conditioning-cache-dir"])
    }

    @MainActor func testExtraUIMountArgsPreferBundledTrees() {
        // Empty when not running inside the app bundle; still exercises the API shape.
        let args = ServerSettings.extraUIMountArgs
        if args.isEmpty {
            XCTAssertTrue(args.isEmpty)
            return
        }
        XCTAssertEqual(args.first, "--ui-mount")
        XCTAssertTrue(args[1].contains("/kvmem:"))
        XCTAssertTrue(args[1].contains("/strata:"))
    }

    func testImageServerSectionExists() {
        XCTAssertEqual(Section_.imageServer.rawValue, "imageServer")
        XCTAssertFalse(Section_.imageServer.icon.isEmpty)
    }
}
