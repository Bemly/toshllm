// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// The 16-bit PCM WAV that afconvert writes for the speech engine, cut into speech spans.
struct PCM16Wave: Sendable {
    let sampleRate: Int
    let channels: Int
    /// Interleaved little-endian samples.
    let samples: Data

    init(sampleRate: Int, channels: Int, samples: Data) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.samples = samples
    }

    /// Reads the `fmt ` and `data` chunks, skipping any other chunk afconvert adds.
    init?(data: Data) {
        let base = data.startIndex
        func uint16(_ offset: Int) -> Int { Int(data[base + offset]) | Int(data[base + offset + 1]) << 8 }
        func uint32(_ offset: Int) -> Int { uint16(offset) | uint16(offset + 2) << 16 }
        func tag(_ offset: Int) -> String {
            String(decoding: data[base + offset..<base + offset + 4], as: UTF8.self)
        }
        guard data.count >= 12, tag(0) == "RIFF", tag(8) == "WAVE" else { return nil }
        var offset = 12
        var format: (rate: Int, channels: Int, bits: Int, tag: Int)?
        while offset + 8 <= data.count {
            let id = tag(offset)
            let size = uint32(offset + 4)
            let body = offset + 8
            if id == "fmt ", body + 16 <= data.count {
                format = (uint32(body + 4), uint16(body + 2), uint16(body + 14), uint16(body))
            } else if id == "data", let format {
                guard format.tag == 1 || format.tag == 0xFFFE, format.bits == 16, format.channels > 0 else {
                    return nil
                }
                let end = min(data.count, body + size)
                self.init(sampleRate: format.rate, channels: format.channels,
                          samples: data[base + body..<base + end])
                return
            }
            offset = body + size + (size & 1)
        }
        return nil
    }

    private var frameBytes: Int { channels * 2 }

    var frameCount: Int { samples.count / max(1, frameBytes) }

    var duration: TimeInterval {
        sampleRate > 0 ? Double(frameCount) / Double(sampleRate) : 0
    }

    /// A complete WAV file holding the frames between two times, clamped to the recording.
    func wavFile(from start: TimeInterval, to end: TimeInterval) -> Data {
        let first = min(frameCount, max(0, Int((start * Double(sampleRate)).rounded(.down))))
        let last = min(frameCount, max(first, Int((end * Double(sampleRate)).rounded(.up))))
        let range = samples.startIndex + first * frameBytes..<samples.startIndex + last * frameBytes
        return Self.wavFile(pcm: samples.subdata(in: range), sampleRate: sampleRate, channels: channels)
    }

    static func wavFile(pcm: Data, sampleRate: Int, channels: Int) -> Data {
        var header = Data()
        func append(_ text: String) { header.append(contentsOf: Array(text.utf8)) }
        func append32(_ value: Int) { withUnsafeBytes(of: UInt32(value).littleEndian) { header.append(contentsOf: $0) } }
        func append16(_ value: Int) { withUnsafeBytes(of: UInt16(value).littleEndian) { header.append(contentsOf: $0) } }
        append("RIFF"); append32(36 + pcm.count); append("WAVE")
        append("fmt "); append32(16); append16(1); append16(channels); append32(sampleRate)
        append32(sampleRate * channels * 2); append16(channels * 2); append16(16)
        append("data"); append32(pcm.count)
        return header + pcm
    }
}
