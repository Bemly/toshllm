// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

struct SpeechLanguage: Identifiable, Hashable {
    let id: String
    let es: String
    let en: String
    /// The label Qwen3-ASR expects in its `language` option.
    let qwenName: String

    static let common: [SpeechLanguage] = [
        .init(id: "auto", es: "Detectar automáticamente", en: "Detect automatically", qwenName: ""),
        .init(id: "es", es: "Español", en: "Spanish", qwenName: "Spanish"),
        .init(id: "en", es: "Inglés", en: "English", qwenName: "English"),
        .init(id: "fr", es: "Francés", en: "French", qwenName: "French"),
        .init(id: "de", es: "Alemán", en: "German", qwenName: "German"),
        .init(id: "it", es: "Italiano", en: "Italian", qwenName: "Italian"),
        .init(id: "pt", es: "Portugués", en: "Portuguese", qwenName: "Portuguese"),
        .init(id: "ja", es: "Japonés", en: "Japanese", qwenName: "Japanese"),
        .init(id: "ko", es: "Coreano", en: "Korean", qwenName: "Korean"),
        .init(id: "zh", es: "Chino", en: "Chinese", qwenName: "Chinese"),
        .init(id: "ru", es: "Ruso", en: "Russian", qwenName: "Russian")
    ]
}
