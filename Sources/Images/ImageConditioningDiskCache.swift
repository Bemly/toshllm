// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

enum ImageConditioningDiskCache {
    struct Entry: Identifiable {
        let id: String
        let scopeID: String
        let cacheID: String
        let prompt: String
        let width: Int
        let height: Int
        let clipSkip: Int
        let zeroOutMasked: Bool
        let created: Date
        let textEncoder: String
        let diffusionModel: String
        let tokenizer: String
        let scope: String
        let roles: Int
    }

    struct Inventory {
        let entries: [Entry]
        let unreadable: Int
    }

    private enum ReadError: Error { case invalid }

    private struct Reader {
        let file: FileHandle

        func bytes(_ count: Int) throws -> Data {
            guard count >= 0, let data = try file.read(upToCount: count), data.count == count else {
                throw ReadError.invalid
            }
            return data
        }

        func number<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
            let data = try bytes(MemoryLayout<T>.size)
            return data.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(as: T.self)) }
        }

        func string() throws -> String {
            let length = try number(UInt64.self)
            guard length <= 16 * 1024 * 1024,
                  let result = String(data: try bytes(Int(length)), encoding: .utf8) else {
                throw ReadError.invalid
            }
            return result
        }
    }

    // The scope uses byte-counted paths, so paths containing semicolons stay unambiguous.
    private static func modelPaths(_ scope: String) -> [String]? {
        let bytes = Array(scope.utf8)
        var index = 0
        func skipField() -> Bool {
            guard let end = bytes[index...].firstIndex(of: 59) else { return false }
            index = end + 1
            return true
        }
        var prefix: [String] = []
        for _ in 0..<6 {
            let start = index
            guard skipField() else { return nil }
            prefix.append(String(decoding: bytes[start..<(index - 1)], as: UTF8.self))
        }
        guard let linear = Double(prefix[4]), linear.isFinite,
              let attention = Double(prefix[5]), attention.isFinite else { return nil }
        var paths: [String] = []
        for _ in 0..<13 {
            guard index < bytes.count, let colon = bytes[index...].firstIndex(of: 58),
                  let length = Int(String(decoding: bytes[index..<colon], as: UTF8.self)),
                  length >= 0, length <= bytes.count - colon - 1 else { return nil }
            index = colon + 1
            guard let path = String(bytes: bytes[index..<(index + length)], encoding: .utf8) else { return nil }
            index += length
            guard index < bytes.count, bytes[index] == 59 else { return nil }
            index += 1
            if !path.isEmpty {
                let start = index
                guard skipField() else { return nil }
                let fingerprint = String(decoding: bytes[start..<(index - 1)], as: UTF8.self)
                if fingerprint != "missing" {
                    let parts = fingerprint.split(separator: ":", omittingEmptySubsequences: false)
                    guard parts.count == 2, UInt64(parts[0]) != nil,
                          parts[1] == "unknown" || Int64(parts[1]) != nil else { return nil }
                }
            }
            paths.append(path)
        }
        // Embedding names are not escaped by the engine; leave paths unlabeled in that case.
        guard String(decoding: bytes[index...], as: UTF8.self).hasPrefix(";lora=") else { return nil }
        return paths
    }

    static func read(_ url: URL) throws -> Entry {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let reader = Reader(file: handle)
        guard try reader.bytes(8) == Data("SDCACHE1".utf8),
              try reader.number(UInt32.self) == 1,
              try reader.number(UInt32.self) == 0x01020304 else { throw ReadError.invalid }
        let created = try reader.number(Int64.self)
        let scope = try reader.string()
        _ = try reader.string()
        let prompt = try reader.string()
        let clipSkip = try reader.number(Int32.self)
        let width = try reader.number(Int32.self)
        let height = try reader.number(Int32.self)
        let masked = try reader.number(UInt8.self)
        guard masked <= 1 else { throw ReadError.invalid }
        let paths = modelPaths(scope)
        let relativeID = url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        let roleURL = url.deletingPathExtension().appendingPathExtension("roles")
        let roles: Int
        do {
            let value = try String(contentsOf: roleURL, encoding: .utf8)
            guard let parsed = Int(value), (1...3).contains(parsed) else { throw ReadError.invalid }
            roles = parsed
        } catch let error as NSError where error.domain == NSCocoaErrorDomain && error.code == NSFileReadNoSuchFileError {
            roles = 0
        }
        return Entry(id: relativeID, scopeID: url.deletingLastPathComponent().lastPathComponent,
                     cacheID: url.deletingPathExtension().lastPathComponent,
                     prompt: prompt, width: Int(width), height: Int(height),
                     clipSkip: Int(clipSkip), zeroOutMasked: masked == 1,
                     created: Date(timeIntervalSince1970: TimeInterval(created)),
                     textEncoder: paths?[5] ?? "", diffusionModel: paths?[8] ?? "",
                     tokenizer: paths?[7] ?? "", scope: scope, roles: roles)
    }

    static func remove(directory: URL, scope: String, id: String) throws {
        guard validHash(scope), validHash(id) else { throw ReadError.invalid }
        let folder = directory.appendingPathComponent(scope, isDirectory: true)
        let values = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else { throw ReadError.invalid }
        try FileManager.default.removeItem(at: folder.appendingPathComponent(id + ".cndcache"))
        let roleFile = folder.appendingPathComponent(id + ".roles")
        do {
            try FileManager.default.removeItem(at: roleFile)
        } catch let error as NSError where error.domain == NSCocoaErrorDomain && error.code == NSFileNoSuchFileError {
        }
    }

    static func removeAll(directory: URL) throws {
        let manager = FileManager.default
        let folders: [URL]
        do {
            folders = try manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        } catch let error as NSError where error.domain == NSCocoaErrorDomain && error.code == NSFileReadNoSuchFileError {
            return
        }
        for folder in folders where validHash(folder.lastPathComponent) {
            let values = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true else { continue }
            for file in try manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
                where file.pathExtension == "cndcache" || file.pathExtension == "roles" {
                try manager.removeItem(at: file)
            }
        }
    }

    private static func validHash(_ value: String) -> Bool {
        value.utf8.count == 16 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }

    static func scan(directory: URL) throws -> Inventory {
        let manager = FileManager.default
        let folders: [URL]
        do {
            folders = try manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey],
                                                      options: [.skipsHiddenFiles])
        } catch let error as NSError where error.domain == NSCocoaErrorDomain && error.code == NSFileReadNoSuchFileError {
            return Inventory(entries: [], unreadable: 0)
        }
        var entries: [Entry] = []
        var unreadable = 0
        for folder in folders {
            guard try folder.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else { continue }
            let files = try manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil,
                                                        options: [.skipsHiddenFiles])
            for file in files where file.pathExtension == "cndcache" {
                do {
                    entries.append(try read(file))
                } catch {
                    unreadable += 1
                }
            }
        }
        entries.sort { $0.created > $1.created }
        return Inventory(entries: entries, unreadable: unreadable)
    }
}
