import Foundation
import XCTest
@testable import ToshLLM

final class ImageConditioningDiskCacheTests: XCTestCase {
    func testInventoryReadsMultipleScopesWithoutModel() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        func write(_ folder: String, encoder: String, prompt: String, prefix: String = "a32089c;NONE;;;0;0;") throws {
            let dir = root.appendingPathComponent(folder)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let paths = ["", "", "", "", "", encoder, "", "/tokenizer.json", "/moss.gguf", "", "", "", ""]
            let scope = prefix + paths.map {
                "\($0.utf8.count):\($0);" + ($0.isEmpty ? "" : "100:123;")
            }.joined() + ";lora="
            var data = Data("SDCACHE1".utf8)
            func integer<T: FixedWidthInteger>(_ value: T) {
                var little = value.littleEndian
                withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
            }
            func string(_ value: String) {
                let bytes = Data(value.utf8)
                integer(UInt64(bytes.count))
                data.append(bytes)
            }
            integer(UInt32(1)); integer(UInt32(0x01020304)); integer(Int64(1000))
            string(scope); string("same-id"); string(prompt)
            integer(Int32(-1)); integer(Int32(1024)); integer(Int32(1536)); integer(UInt8(0))
            try data.write(to: dir.appendingPathComponent("same-id.cndcache"))
        }

        try write("scope-a", encoder: "/模型;one.gguf", prompt: String(repeating: "prompt ", count: 100))
        try write("scope-b", encoder: "/other.gguf", prompt: "second")
        try write("scope-c", encoder: "/ambiguous.gguf", prompt: "third", prefix: "a32089c;NONE;rule;extra;;0;0;")
        try Data("3".utf8).write(to: root.appendingPathComponent("scope-b/same-id.roles"))
        try Data("invalid".utf8).write(to: root.appendingPathComponent("scope-a/broken.cndcache"))
        let result = try ImageConditioningDiskCache.scan(directory: root)
        XCTAssertEqual(result.entries.count, 3)
        XCTAssertEqual(result.unreadable, 1)
        XCTAssertEqual(Set(result.entries.map(\.id)), ["scope-a/same-id.cndcache", "scope-b/same-id.cndcache", "scope-c/same-id.cndcache"])
        let first = try XCTUnwrap(result.entries.first { $0.textEncoder == "/模型;one.gguf" })
        XCTAssertEqual(first.prompt, String(repeating: "prompt ", count: 100))
        XCTAssertEqual(first.diffusionModel, "/moss.gguf")
        XCTAssertEqual(first.tokenizer, "/tokenizer.json")
        XCTAssertEqual(first.width, 1024)
        XCTAssertEqual(first.height, 1536)
        XCTAssertEqual(first.clipSkip, -1)
        XCTAssertEqual(first.roles, 0)
        XCTAssertEqual(result.entries.first { $0.prompt == "second" }?.roles, 3)
        XCTAssertEqual(result.entries.first { $0.prompt == "third" }?.textEncoder, "")
    }

    func testMissingDirectoryIsEmpty() throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let result = try ImageConditioningDiskCache.scan(directory: missing)
        XCTAssertTrue(result.entries.isEmpty)
        XCTAssertEqual(result.unreadable, 0)
    }

    func testDeleteDiskEntriesIsScopedAndRejectsTraversal() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let folder = root.appendingPathComponent("0123456789abcdef")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = folder.appendingPathComponent("fedcba9876543210.cndcache")
        let other = folder.appendingPathComponent("other.txt")
        let roles = folder.appendingPathComponent("fedcba9876543210.roles")
        try Data().write(to: file)
        try Data("1".utf8).write(to: roles)
        try Data().write(to: other)
        XCTAssertThrowsError(try ImageConditioningDiskCache.remove(directory: root, scope: "../elsewhere", id: "fedcba9876543210"))
        try ImageConditioningDiskCache.remove(directory: root, scope: "0123456789abcdef", id: "fedcba9876543210")
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: roles.path))
        try Data().write(to: file)
        try Data("2".utf8).write(to: roles)
        try ImageConditioningDiskCache.removeAll(directory: root)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: roles.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: other.path))
    }
}
