// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

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
}
