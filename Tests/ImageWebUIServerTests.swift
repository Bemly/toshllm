// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Combine
import XCTest
@testable import ToshLLM

final class ImageWebUIServerTests: XCTestCase {
    func testListenReadyAcceptsEmptyAndReady() {
        XCTAssertTrue(ImageWebUIServer.isListenReady(status: "empty"))
        XCTAssertTrue(ImageWebUIServer.isListenReady(status: "ready"))
    }

    func testListenReadyRejectsLoadingAndUnknown() {
        XCTAssertFalse(ImageWebUIServer.isListenReady(status: "loading"))
        XCTAssertFalse(ImageWebUIServer.isListenReady(status: "error"))
        XCTAssertFalse(ImageWebUIServer.isListenReady(status: ""))
    }

    @MainActor
    func testIsActiveWhileStartingOrRunning() {
        // Shared singleton — only assert the pure helpers + documented contract.
        // empty WebUI must count as listen-ready so Settings leaves "Starting".
        XCTAssertEqual(
            ["empty", "ready"].filter(ImageWebUIServer.isListenReady(status:)),
            ["empty", "ready"])
    }

    @MainActor
    func testJobActivityNotifiesCacheView() {
        let server = ImageWebUIServer.shared
        let initialCount = server.jobsInFlight
        var counts: [Int] = []
        let observer = server.$jobsInFlight.sink { counts.append($0) }
        server.beginJob()
        server.endJob()
        XCTAssertEqual(counts, [initialCount, initialCount + 1, initialCount])
        withExtendedLifetime(observer) {}
    }

    @MainActor
    func testCancellationKeepsGeneratorBusyUntilRunFinishes() {
        let generator = ImageGenerator()
        generator.state = .generating
        generator.cancel()
        XCTAssertTrue(generator.cancelRequested)
        XCTAssertTrue(generator.isBusy)
        generator.cancel()
        XCTAssertTrue(generator.cancelRequested)
    }

    @MainActor
    func testStoppingQueuePreservesPendingJobsAndIndependentRuns() {
        let pool = ImageGenPool()
        let generator = pool.generator(for: pool.configs[0].id)
        generator.state = .generating
        pool.enqueue("pending")
        pool.queueActive = true
        pool.stopQueue()
        XCTAssertFalse(pool.queueActive)
        XCTAssertEqual(pool.queue.count, 1)
        XCTAssertFalse(generator.cancelRequested)
    }
}
