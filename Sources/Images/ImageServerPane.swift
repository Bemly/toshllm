// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

struct ImageServerSidebarRow: View {
    @ObservedObject private var server = ImageWebUIServer.shared
    @EnvironmentObject private var control: ControlPanelState
    @EnvironmentObject private var loc: Localizer
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let selected = control.section == .imageServer
        Button {
            let select = {
                control.serverAnchor = nil
                control.section = .imageServer
            }
            if reduceMotion { select() } else { withAnimation(.snappy(duration: 0.2), select) }
        } label: {
            HStack(spacing: 12) {
                Image(systemName: server.isRunning ? "circle.fill" : "circle")
                    .font(.system(size: 9))
                    .foregroundStyle(server.isRunning ? .green : .secondary)
                Text(loc.t("Servidor de imagen", "Image server"))
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            .padding(.leading, 14).padding(.vertical, 11)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(selected ? WorkspaceStyle.inset : .clear, in: RoundedRectangle(cornerRadius: 8))
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

struct ImageServerPane: View {
    @ObservedObject private var server = ImageWebUIServer.shared
    @EnvironmentObject private var loc: Localizer
    @AppStorage(SettingsKeys.imagenWebUIPort) private var port = 1234

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 10) {
                    Circle()
                        .fill(server.isRunning ? Color.green : Color.secondary)
                        .frame(width: 8, height: 8)
                    Text(statusText)
                        .font(.headline)
                    Spacer()
                    Button(server.isRunning
                           ? loc.t("Detener", "Stop")
                           : loc.t("Iniciar", "Start")) {
                        if server.isRunning { server.stop() } else { server.start() }
                    }
                    Button(loc.t("Abrir WebUI", "Open WebUI")) { server.openInBrowser() }
                        .disabled(ImageWebUIServer.bundledBinary() == nil)
                }
                HStack(spacing: 8) {
                    Text(loc.t("Puerto", "Port"))
                    TextField("1234", value: $port, format: .number.grouping(.never))
                        .frame(width: 80)
                        .onChange(of: port) { _, _ in
                            if server.isRunning {
                                server.stop()
                                server.start()
                            }
                        }
                }
                if !server.loadedSummary.isEmpty {
                    Text(server.loadedSummary)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Text(loc.t(
                    "Generar usa este proceso. Si el texto ya está en caché, no vuelve a cargar el codificador.",
                    "Generate uses this process. A cached prompt does not load the text encoder again."))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if case .failed(let msg) = server.state {
                    Text(msg).font(.caption).foregroundStyle(.orange)
                }
                if !server.logText.isEmpty {
                    Text(server.logText)
                        .font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(24)
            .frame(maxWidth: 720, alignment: .leading)
        }
    }

    private var statusText: String {
        switch server.state {
        case .stopped: return loc.t("Detenido", "Stopped")
        case .starting: return loc.t("Arrancando", "Starting")
        case .running: return loc.t("En marcha · %@", "Running · %@", "\(server.port)")
        case .failed: return loc.t("Error", "Failed")
        }
    }
}
