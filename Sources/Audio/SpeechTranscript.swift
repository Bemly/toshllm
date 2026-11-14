// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

struct SpeechWord: Equatable, Sendable {
    let start: TimeInterval
    let end: TimeInterval
    let text: String
}

/// Turns audio.cpp transcripts, word timestamps and VAD spans into chat text and subtitle cues.
enum SpeechTranscript {
    /// Cue limits: two subtitle lines of about 42 characters, a comfortable reading time, and
    /// the pause that reads as a new sentence.
    static let maxCueCharacters = 84
    static let maxCueSeconds = 7.0
    static let cueBreakPause = 0.8

    static func normalized(_ raw: String) -> String {
        raw.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// `[{"start_sample":…,"end_sample":…,"word":…}]`, as `--words-out` writes it and inside
    /// the `words` field of the server's details response.
    static func words(jsonData: Data, sampleRate: Double = SpeechEngine.sampleRate,
                      offset: TimeInterval = 0) -> [SpeechWord] {
        guard let rows = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]] else {
            return []
        }
        return words(rows: rows, sampleRate: sampleRate, offset: offset)
    }

    /// Text and word timestamps from a `POST /v1/audio/transcriptions/details` response.
    static func details(jsonData: Data) -> (text: String, words: [SpeechWord])? {
        guard let root = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            return nil
        }
        let rate = (root["sample_rate"] as? NSNumber)?.doubleValue ?? SpeechEngine.sampleRate
        let rows = root["words"] as? [[String: Any]] ?? []
        return (root["text"] as? String ?? "", words(rows: rows, sampleRate: rate, offset: 0))
    }

    /// Speech spans in seconds from the Silero VAD `--segments-out` file.
    static func speechSpans(jsonData: Data,
                            sampleRate: Double = SpeechEngine.sampleRate) -> [ClosedRange<TimeInterval>] {
        guard let rows = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]] else {
            return []
        }
        return rows.compactMap { row in
            guard let start = (row["start_sample"] as? NSNumber)?.doubleValue,
                  let end = (row["end_sample"] as? NSNumber)?.doubleValue, end > start else { return nil }
            return (start / sampleRate)...(end / sampleRate)
        }
    }

    /// Back-to-back windows covering the recording when voice detection is off.
    static func fixedWindows(duration: TimeInterval,
                             window: TimeInterval = AudioVADConfiguration.standardMaxSpeechSeconds)
        -> [ClosedRange<TimeInterval>] {
        guard duration > 0, window > 0 else { return [] }
        return stride(from: 0.0, to: duration, by: window).map { $0...min(duration, $0 + window) }
    }

    /// Joins recognized words, leaving no space between CJK or Thai characters (which the
    /// models emit one per entry) and none before closing punctuation.
    static func join(_ words: [String]) -> String {
        var result = ""
        for raw in words {
            let word = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !word.isEmpty else { continue }
            if let last = result.unicodeScalars.last, let first = word.unicodeScalars.first,
               !(isUnspaced(last) || isUnspaced(first) || isClosingPunctuation(first)) {
                result += " "
            }
            result += word
        }
        return result
    }

    /// Subtitle cues for one transcribed span. Word timestamps place each cue exactly; without
    /// them the text is split at sentences and timed in proportion to its length.
    static func cues(text: String, words: [SpeechWord],
                     span: ClosedRange<TimeInterval>) -> [SubtitleCue] {
        let grouped = groups(of: words)
        if !grouped.isEmpty {
            return grouped.enumerated().compactMap { index, group in
                let text = join(group.map(\.text))
                guard !text.isEmpty, let first = group.first, let last = group.last else { return nil }
                let nextStart = index + 1 < grouped.count ? grouped[index + 1].first?.start ?? span.upperBound
                    : span.upperBound
                let start = max(span.lowerBound, first.start)
                let end = min(max(last.end, start + 0.5), max(nextStart, last.end), span.upperBound)
                return SubtitleCue(id: 0, start: start, end: max(end, start), text: text)
            }
        }
        let pieces = textPieces(normalized(text))
        let total = pieces.reduce(0) { $0 + $1.count }
        guard total > 0 else { return [] }
        let duration = span.upperBound - span.lowerBound
        var cursor = span.lowerBound
        return pieces.map { piece in
            let length = duration * Double(piece.count) / Double(total)
            defer { cursor += length }
            return SubtitleCue(id: 0, start: cursor, end: cursor + length, text: piece)
        }
    }

    /// Cues numbered from 1 in time order.
    static func numbered(_ cues: [SubtitleCue]) -> [SubtitleCue] {
        cues.sorted { $0.start < $1.start }.enumerated().map { index, cue in
            SubtitleCue(id: index + 1, start: cue.start, end: cue.end, text: cue.text)
        }
    }

    /// One `[hh:mm:ss] text` line per cue, the format attached recordings reach the chat in.
    static func timestamped(_ cues: [SubtitleCue]) -> String {
        cues.map { "[\(timestamp(seconds: $0.start))] \($0.text)" }.joined(separator: "\n")
    }

    private static func words(rows: [[String: Any]], sampleRate: Double,
                              offset: TimeInterval) -> [SpeechWord] {
        guard sampleRate > 0 else { return [] }
        return rows.compactMap { row in
            guard let text = row["word"] as? String,
                  let start = (row["start_sample"] as? NSNumber)?.doubleValue,
                  let end = (row["end_sample"] as? NSNumber)?.doubleValue else { return nil }
            return SpeechWord(start: offset + start / sampleRate,
                              end: offset + max(start, end) / sampleRate, text: text)
        }
    }

    private static func groups(of words: [SpeechWord]) -> [[SpeechWord]] {
        var groups: [[SpeechWord]] = []
        var current: [SpeechWord] = []
        for word in words where !word.text.trimmingCharacters(in: .whitespaces).isEmpty {
            if let first = current.first, let previous = current.last {
                let characters = join(current.map(\.text)).count + 1 + word.text.count
                let sentenceEnded = previous.text.last.map(isSentenceEnd) ?? false
                if word.start - previous.end >= cueBreakPause
                    || characters > maxCueCharacters
                    || word.end - first.start > maxCueSeconds
                    || sentenceEnded && join(current.map(\.text)).count >= maxCueCharacters / 3 {
                    groups.append(current)
                    current = []
                }
            }
            current.append(word)
        }
        if !current.isEmpty { groups.append(current) }
        return groups
    }

    /// Splits text at sentence ends and keeps each piece under the cue length, cutting between
    /// words, or between characters for scripts written without spaces.
    private static func textPieces(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        let spaced = text.contains(" ")
        let units = spaced ? text.split(separator: " ").map(String.init) : text.map(String.init)
        var pieces: [String] = []
        var current: [String] = []
        func render(_ units: [String]) -> String { spaced ? units.joined(separator: " ") : units.joined() }
        for unit in units {
            if !current.isEmpty, render(current + [unit]).count > maxCueCharacters {
                pieces.append(render(current))
                current = []
            }
            current.append(unit)
            if let last = unit.last, isSentenceEnd(last), render(current).count >= maxCueCharacters / 3 {
                pieces.append(render(current))
                current = []
            }
        }
        if !current.isEmpty { pieces.append(render(current)) }
        return pieces
    }

    private static func isSentenceEnd(_ character: Character) -> Bool {
        ".!?。！？…".contains(character)
    }

    private static func isClosingPunctuation(_ scalar: Unicode.Scalar) -> Bool {
        ".,;:!?)]}%…»”’、。，！？；：）".unicodeScalars.contains(scalar)
    }

    private static func isUnspaced(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x0E00...0x0E7F,   // Thai
             0x3000...0x30FF,   // CJK punctuation, Hiragana, Katakana
             0x3400...0x4DBF,   // CJK extension A
             0x4E00...0x9FFF,   // CJK unified ideographs
             0xF900...0xFAFF,   // CJK compatibility ideographs
             0xFF00...0xFFEF:   // full-width forms
            return true
        default:
            return false
        }
    }

    private static func timestamp(seconds: Double) -> String {
        let total = max(0, Int(seconds))
        func twoDigits(_ value: Int) -> String {
            value < 10 ? "0\(value)" : String(value)
        }
        return "\(twoDigits(total / 3_600)):\(twoDigits((total / 60) % 60)):\(twoDigits(total % 60))"
    }
}
