// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

struct VoiceInputOptionsView: View {
    @EnvironmentObject private var loc: Localizer
    @EnvironmentObject private var models: ModelStore
    @Binding var methodRaw: String
    @Binding var loadPolicyRaw: String
    @Binding var speechModelID: String
    let dismiss: () -> Void

    private var selectedModel: SpeechModel { SpeechModel.model(id: speechModelID) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(loc.t("Entrada de voz", "Voice input"))
                .font(.headline)

            Button {
                methodRaw = SpeechInputMethod.apple.rawValue
            } label: {
                methodLabel(
                    title: loc.t("Dictado de Apple", "Apple Dictation"),
                    detail: loc.t("En vivo · sin reservar VRAM", "Live · no VRAM reserved"),
                    systemImage: "apple.logo",
                    selected: methodRaw == SpeechInputMethod.apple.rawValue)
            }
            .buttonStyle(.plain)
            .help(loc.t("Usa el dictado integrado de macOS sin cargar un modelo en la GPU.",
                        "Uses macOS built-in dictation without loading a model on the GPU."))

            Button {
                methodRaw = SpeechInputMethod.audioCpp.rawValue
            } label: {
                methodLabel(
                    title: "audio.cpp · GPU",
                    detail: loc.t("Privado · multilingüe · Radeon", "Private · multilingual · Radeon"),
                    systemImage: "waveform.badge.mic",
                    selected: methodRaw == SpeechInputMethod.audioCpp.rawValue)
            }
            .buttonStyle(.plain)
            .help(loc.t("Transcribe localmente con audio.cpp en la GPU AMD.",
                        "Transcribes locally with audio.cpp on the AMD GPU."))

            if methodRaw == SpeechInputMethod.audioCpp.rawValue {
                Divider()
                Picker(loc.t("Modelo", "Model"), selection: $speechModelID) {
                    ForEach(SpeechModel.catalog) { model in
                        Text("\(model.name) · \(model.sizeMB) MB").tag(model.id)
                    }
                }
                .help(loc.t("Elige el equilibrio entre velocidad, memoria y precisión.",
                            "Choose the balance between speed, memory, and accuracy."))
                Picker(loc.t("Carga", "Loading"), selection: $loadPolicyRaw) {
                    Text(loc.t("Bajo demanda", "On demand")).tag(SpeechLoadPolicy.onDemand.rawValue)
                    Text(loc.t("Siempre cargado", "Always loaded")).tag(SpeechLoadPolicy.alwaysLoaded.rawValue)
                }
                .help(loc.t("Carga bajo demanda libera VRAM; Siempre cargado inicia el dictado más rápido.",
                            "On demand releases VRAM; Always loaded starts dictation faster."))

                if models.speechModelInstalled(selectedModel) {
                    Label(loc.t("Modelo instalado", "Model installed"), systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else if let download = models.speechDownload(selectedModel) {
                    DownloadRow(item: download)
                } else {
                    Button {
                        models.downloadSpeechModel(selectedModel)
                    } label: {
                        Label(loc.t("Descargar %@", "Download %@", "\(selectedModel.name)"),
                              systemImage: "arrow.down.circle")
                    }
                    .help(loc.t("Descarga este modelo de voz en el Mac.",
                                "Downloads this speech model to the Mac."))
                }
            }

            HStack {
                Spacer()
                Button(loc.t("Listo", "Done"), action: dismiss)
                    .keyboardShortcut(.defaultAction)
                    .disabled(methodRaw.isEmpty)
                    .help(loc.t("Guarda el método de entrada de voz seleccionado.",
                                "Saves the selected voice input method."))
            }
        }
        .padding(16)
        .frame(width: 330)
    }

    private func methodLabel(title: String, detail: String,
                             systemImage: String, selected: Bool) -> some View {
        HStack(spacing: 10) {
            Image(systemName: systemImage)
                .frame(width: 24)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(selected ? Color.accentColor : .secondary)
                .accessibilityHidden(true)
        }
        .contentShape(.rect)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(selected ? [.isSelected] : [])
    }
}
