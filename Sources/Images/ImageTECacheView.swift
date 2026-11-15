// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// Lists / clears warm sd-server text-encoder conditioning cache entries.
struct ImageTECacheView: View {
    @EnvironmentObject var loc: Localizer
    @ObservedObject private var server = ImageWebUIServer.shared
    @AppStorage(SettingsKeys.imagenTECacheRetentionDays) private var retentionDays = 7
    @AppStorage(SettingsKeys.imagenWebUI) private var imagenWebUI = false

    @State private var entries: [TECacheEntry] = []
    @State private var capacity = 0
    @State private var size = 0
    @State private var modelLoaded = false
    @State private var status = ""
    @State private var busy = false

    struct TECacheEntry: Identifiable, Equatable {
        let id: String
        let prompt: String
        let width: Int
        let height: Int
        let ageSeconds: Int
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            if !server.isRunning {
                offlineCard
            } else {
                list
            }
        }
        .onAppear { Task { await refresh(purgeFirst: true) } }
        .onChange(of: server.state) { _, _ in
            Task { await refresh(purgeFirst: false) }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(loc.t("Caché del codificador de texto", "Text-encoder cache"))
                    .font(.title3.weight(.semibold))
                Spacer()
                if server.isRunning {
                    Text("\(size)/\(capacity)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                    Button {
                        Task { await refresh(purgeFirst: false) }
                    } label: {
                        Label(loc.t("Actualizar", "Refresh"), systemImage: "arrow.clockwise")
                    }
                    .disabled(busy)
                    Button(role: .destructive) {
                        Task { await clearAll() }
                    } label: {
                        Label(loc.t("Vaciar", "Clear all"), systemImage: "trash")
                    }
                    .disabled(busy || size == 0)
                }
            }
            Text(loc.t("Las incrustaciones de prompt se reutilizan mientras sd-server está caliente. Borra una entrada o fija cuántos días conservarlas.",
                        "Prompt embeddings are reused while sd-server stays warm. Delete an entry or set how many days to keep them."))
                .font(.callout)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Text(loc.t("Conservar", "Keep for"))
                Stepper(value: $retentionDays, in: 0...365) {
                    Text(retentionDays == 0
                         ? loc.t("sin caducidad", "no expiry")
                         : loc.t("%@ días", "%@ days", "\(retentionDays)"))
                        .frame(minWidth: 100, alignment: .leading)
                }
                .onChange(of: retentionDays) { _, _ in
                    Task { await refresh(purgeFirst: true) }
                }
                Spacer()
                if !status.isEmpty {
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
            }
            .controlSize(.small)
        }
    }

    private var offlineCard: some View {
        VStack(spacing: 12) {
            Image(systemName: "externaldrive.badge.xmark")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(.tertiary)
            Text(loc.t("sd-server no está en marcha", "sd-server is not running"))
                .font(.headline)
            Text(loc.t("Arranca el servidor de imagen en caliente para ver y borrar la caché del codificador.",
                        "Start the warm image server to inspect and clear the text-encoder cache."))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
            Button {
                imagenWebUI = true
                server.start()
            } label: {
                Label(loc.t("Iniciar servidor de imagen", "Start image server"), systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var list: some View {
        Group {
            if entries.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "internaldrive")
                        .font(.system(size: 28, weight: .light))
                        .foregroundStyle(.tertiary)
                    Text(modelLoaded
                         ? loc.t("Ninguna entrada en caché", "No cached entries")
                         : loc.t("Modelo no cargado — la caché aparece tras la primera generación",
                                 "No model loaded — the cache appears after the first generation"))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(entries) { entry in
                        HStack(alignment: .top, spacing: 12) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(entry.prompt.isEmpty ? "—" : entry.prompt)
                                    .lineLimit(3)
                                HStack(spacing: 8) {
                                    if entry.width > 0 && entry.height > 0 {
                                        Text("\(entry.width)×\(entry.height)")
                                            .font(.caption.monospaced())
                                    }
                                    Text(ageLabel(entry.ageSeconds))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                            Button(role: .destructive) {
                                Task { await delete(entry.id) }
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.borderless)
                            .disabled(busy)
                        }
                        .padding(.vertical, 4)
                    }
                }
                .listStyle(.inset)
            }
        }
    }

    private func ageLabel(_ seconds: Int) -> String {
        if seconds < 60 { return loc.t("hace %@ s", "%@s ago", "\(seconds)") }
        if seconds < 3600 { return loc.t("hace %@ min", "%@ min ago", "\(seconds / 60)") }
        if seconds < 86400 { return loc.t("hace %@ h", "%@h ago", "\(seconds / 3600)") }
        return loc.t("hace %@ d", "%@d ago", "\(seconds / 86400)")
    }

    @MainActor
    private func refresh(purgeFirst: Bool) async {
        guard server.isRunning else {
            entries = []; size = 0; capacity = 0; modelLoaded = false
            return
        }
        busy = true
        defer { busy = false }
        if purgeFirst, retentionDays > 0 {
            let removed = (try? await server.purgeConditioningCache(maxAgeSeconds: retentionDays * 86_400)) ?? 0
            if removed > 0 {
                status = loc.t("Eliminadas %@ caducadas", "Purged %@ expired", "\(removed)")
            }
        }
        do {
            let snap = try await server.fetchConditioningCache()
            capacity = snap.capacity
            size = snap.size
            modelLoaded = snap.modelLoaded
            entries = snap.entries.map {
                TECacheEntry(id: $0.id, prompt: $0.prompt, width: $0.width,
                             height: $0.height, ageSeconds: $0.ageSeconds)
            }
            if status.isEmpty {
                status = loc.t("Actualizado", "Updated")
            }
        } catch {
            status = error.localizedDescription
        }
    }

    @MainActor
    private func delete(_ id: String) async {
        busy = true
        defer { busy = false }
        do {
            try await server.deleteConditioningCacheEntry(id: id)
            status = loc.t("Entrada eliminada", "Entry deleted")
            await refresh(purgeFirst: false)
        } catch {
            status = error.localizedDescription
        }
    }

    @MainActor
    private func clearAll() async {
        busy = true
        defer { busy = false }
        do {
            try await server.clearConditioningCache()
            status = loc.t("Caché vaciada", "Cache cleared")
            await refresh(purgeFirst: false)
        } catch {
            status = error.localizedDescription
        }
    }
}
