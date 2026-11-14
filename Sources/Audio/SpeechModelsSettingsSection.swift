// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

struct SpeechModelsSettingsSection: View {
    @EnvironmentObject private var loc: Localizer
    @EnvironmentObject private var models: ModelStore
    @AppStorage(SettingsKeys.speechModel) private var selectedID = SpeechModel.recommendedID
    @AppStorage(SettingsKeys.speechInputMethod) private var methodRaw = ""
    @AppStorage(SettingsKeys.speechLoadPolicy) private var loadPolicyRaw = SpeechLoadPolicy.onDemand.rawValue

    private var selected: SpeechModel { SpeechModel.model(id: selectedID) }

    var body: some View {
        // The panel header above already names this category.
        Section {
            LabeledContent(loc.t("Método del micrófono", "Microphone method")) {
                ToshDropdown(selection: $methodRaw, options: [
                    .init(value: "", title: loc.t("Elegir…", "Choose…")),
                    .init(value: SpeechInputMethod.apple.rawValue, title: loc.t("Dictado de Apple", "Apple Dictation")),
                    .init(value: SpeechInputMethod.audioCpp.rawValue, title: "audio.cpp · GPU"),
                ])
            }
            .settingsGlyph("mic")
            .help(loc.t("Elige el dictado de macOS o audio.cpp acelerado por la GPU.",
                        "Choose macOS dictation or GPU-accelerated audio.cpp."))

            if methodRaw == SpeechInputMethod.apple.rawValue {
                Text(loc.t("Dictado en vivo con el servicio integrado de macOS. No reserva VRAM.",
                           "Live dictation with macOS's built-in service. It reserves no VRAM."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if methodRaw != SpeechInputMethod.apple.rawValue {
                LabeledContent(loc.t("Carga del modelo de voz", "Speech model loading")) {
                    ToshDropdown(selection: $loadPolicyRaw, options: [
                        .init(value: SpeechLoadPolicy.onDemand.rawValue, title: loc.t("Bajo demanda", "On demand")),
                        .init(value: SpeechLoadPolicy.alwaysLoaded.rawValue, title: loc.t("Siempre cargado", "Always loaded")),
                    ])
                }
                .settingsGlyph("memorychip")
                .help(loc.t("Carga bajo demanda libera VRAM; Siempre cargado reduce la espera del próximo dictado.",
                            "On demand releases VRAM; Always loaded reduces the wait for the next dictation."))

                Text(loadPolicyRaw == SpeechLoadPolicy.alwaysLoaded.rawValue
                     ? loc.t("Conserva el modelo en la VRAM para que el siguiente dictado empiece antes.",
                             "Keeps the model in VRAM so the next dictation starts sooner.")
                     : loc.t("Libera el modelo y su VRAM al terminar cada transcripción.",
                             "Releases the model and its VRAM after each transcription."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

            LabeledContent(loc.t("Modelo de voz", "Speech model")) {
                ToshDropdown(selection: $selectedID, options: SpeechModel.catalog.map {
                    .init(value: $0.id, title: label(for: $0))
                }, width: 320)
            }
            .settingsGlyph("waveform")
            .help(loc.t("Elige el equilibrio entre velocidad, uso de VRAM y precisión.",
                        "Choose the balance between speed, VRAM use, and accuracy."))

            LabeledContent(loc.t("Perfil", "Profile")) {
                Text(loc.t(selected.detailES, selected.detailEN))
                    .foregroundStyle(.secondary)
            }
            .settingsGlyph("info.circle")

            if models.speechModelInstalled(selected) {
                Label(loc.t("Instalado y listo para micrófono, audio y video",
                            "Installed and ready for microphone, audio, and video"),
                      systemImage: "checkmark.seal.fill")
                    .foregroundStyle(.green)
            } else if let download = models.speechDownload(selected) {
                DownloadRow(item: download)
            } else {
                Button {
                    models.downloadSpeechModel(selected)
                } label: {
                    Label(loc.t("Descargar %@ (%@ MB)", "Download %@ (%@ MB)", "\(selected.name)", "\(selected.sizeMB)"),
                          systemImage: "arrow.down.circle")
                }
                .help(loc.t("Descarga el modelo seleccionado para usarlo con micrófono, audio y vídeo.",
                            "Downloads the selected model for microphone, audio, and video."))
            }

            if selected.needsAligner {
                LabeledContent(loc.t("Tiempos por palabra", "Word timings")) {
                    if models.speechAlignerInstalled(selected) {
                        Label(loc.t("Instalados", "Installed"), systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else if let download = models.speechAlignerDownload(selected) {
                        DownloadRow(item: download)
                    } else if let aligner = selected.aligner {
                        Button(loc.t("Descargar alineador (%@ MB)", "Download aligner (%@ MB)",
                                     "\(aligner.sizeMB)")) {
                            models.downloadSpeechAligner(selected)
                        }
                        .help(loc.t("Qwen3 Forced Aligner marca el inicio y el fin de cada palabra; sin él los subtítulos se reparten dentro de cada tramo de voz.",
                                    "Qwen3 Forced Aligner marks each word's start and end; without it subtitles are spread across each speech segment."))
                    }
                }
                .settingsGlyph("text.line.first.and.arrowtriangle.forward")
            }

            Text(loc.t("El modelo se descarga bajo demanda. El audio y los videos se transcriben localmente en la GPU y se entregan al chat como texto.",
                       "The model downloads on demand. Audio and video are transcribed locally on the GPU and passed to chat as text."))
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private func label(for model: SpeechModel) -> String {
        let badge: String
        switch model.tier {
        case .fast: badge = loc.t("Rápido", "Fast")
        case .balanced: badge = loc.t("Equilibrado", "Balanced")
        case .recommended: badge = loc.t("Recomendado", "Recommended")
        case .maximum: badge = loc.t("Máxima precisión", "Maximum accuracy")
        case .advanced: badge = loc.t("Avanzado", "Advanced")
        }
        return "\(model.name) · \(model.sizeMB) MB · \(badge)"
    }
}
