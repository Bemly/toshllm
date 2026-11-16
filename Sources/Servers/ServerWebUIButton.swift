// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

struct ServerWebUIButton: View {
    enum Presentation {
        case labeled
        case icon
    }

    @ObservedObject var server: ServerController
    var presentation: Presentation = .labeled
    @EnvironmentObject private var loc: Localizer
    @Environment(\.openWindow) private var openWindow
    @ObservedObject private var imageWebUI = ImageWebUIServer.shared

    private var canOpen: Bool {
        server.state == .running || imageWebUI.isRunning
    }

    var body: some View {
        Button {
            openWindow(id: "webui", value: WebUITarget(
                llamaURL: server.webChatURL,
                title: loc.t("WebUI", "WebUI")))
        } label: {
            if presentation == .icon {
                Image(systemName: "macwindow")
            } else {
                Label("WebUI", systemImage: "macwindow")
            }
        }
        .disabled(!canOpen)
        .opacity(canOpen ? 1 : 0.48)
        .help(canOpen
              ? loc.t("Abrir WebUI en ToshLLM (llama.cpp, KVMem, Strata, sd.cpp)",
                      "Open WebUI inside ToshLLM (llama.cpp, KVMem, Strata, sd.cpp)")
              : loc.t("Inicia el servidor de texto o el WebUI de imágenes",
                      "Start the text server or the image WebUI"))
        .accessibilityLabel(loc.t("Abrir WebUI", "Open WebUI"))
    }
}
