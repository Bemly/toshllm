// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// Lists / clears warm sd-server text-encoder conditioning cache entries.
struct ImageTECacheView: View {
    @EnvironmentObject var loc: Localizer
    @ObservedObject private var server = ImageWebUIServer.shared
    @AppStorage(SettingsKeys.imagenTECacheRetentionDays) private var retentionDays = 7

    @State private var entries: [TECacheEntry] = []
    @State private var diskEntries: [ImageConditioningDiskCache.Entry] = []
    @State private var diskUnreadable = 0
    @State private var diskLoaded = false
    @State private var capacity = 0
    @State private var size = 0
    @State private var storageScopeID = ""
    @State private var modelLoaded = false
    @State private var hasSnapshot = false
    @State private var textEncoder = ""
    @State private var diffusionModel = ""
    @State private var tokenizer = ""
    @State private var loras: [ImageWebUIServer.ConditioningCacheSnapshot.Lora] = []
    @State private var status = ""
    @State private var busy = false
    @State private var refreshID = UUID()
    @State private var confirmingClearDisk = false

    private struct RefreshTrigger: Equatable {
        let isRunning: Bool
        let jobsInFlight: Int
    }

    struct TECacheEntry: Identifiable, Equatable {
        let id: String
        let prompt: String
        let width: Int
        let height: Int
        let clipSkip: Int?
        let zeroOutMasked: Bool?
        let ageSeconds: Int
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            list
        }
        .task(id: RefreshTrigger(isRunning: server.isRunning, jobsInFlight: server.jobsInFlight)) {
            repeat {
                await refresh(purgeFirst: server.jobsInFlight == 0)
                guard server.isRunning, server.jobsInFlight > 0, !Task.isCancelled else { return }
                do {
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                } catch {
                    return
                }
            } while !Task.isCancelled
        }
        .confirmationDialog(loc.t("¿Eliminar toda la caché de disco?", "Delete all disk cache?"),
                            isPresented: $confirmingClearDisk) {
            Button(loc.t("Eliminar toda la caché", "Delete all cache"), role: .destructive) {
                Task { await deleteDisk() }
            }
        } message: {
            Text(loc.t("También se borrará la caché activa. Esta acción no se puede deshacer.",
                       "The active cache will also be cleared. This cannot be undone."))
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(loc.t("Caché del codificador de texto", "Text-encoder cache"))
                    .font(.title3.weight(.semibold))
                Spacer()
                Button {
                    Task { await refresh(purgeFirst: false) }
                } label: {
                    Label(loc.t("Actualizar", "Refresh"), systemImage: "arrow.clockwise")
                }
                .disabled(busy)
                Button(role: .destructive) {
                    confirmingClearDisk = true
                } label: {
                    Label(loc.t("Eliminar toda la caché", "Delete all cache"), systemImage: "trash")
                }
                .disabled(busy || server.jobsInFlight > 0 || (diskEntries.isEmpty && diskUnreadable == 0 && size == 0) || (server.isActive && !server.isRunning))
            }
            HStack(spacing: 14) {
                Label(loc.t("Disco: %@", "Disk: %@", "\(diskEntries.count)"), systemImage: "internaldrive")
                Text(modelLoaded
                     ? loc.t("Caché activa: %@/%@", "Active cache: %@/%@", "\(size)", "\(capacity)")
                     : loc.t("Ningún modelo cargado", "No model loaded"))
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
            Text(loc.t("Las entradas de disco no están validadas. La caducidad se aplica al modelo activo. Las entradas antiguas no indican si el texto es positivo o negativo.",
                       "Disk entries are unvalidated. Retention applies to the active model. Older entries may not record whether text is positive or negative."))
                .font(.caption).foregroundStyle(.secondary)
            if modelLoaded {
                VStack(alignment: .leading, spacing: 3) {
                    scopeLine(loc.t("Codificador de texto", "Text encoder"), textEncoder)
                    scopeLine(loc.t("Modelo de difusión", "Diffusion model"), diffusionModel)
                    scopeLine(loc.t("Tokenizador", "Tokenizer"), tokenizer)
                    scopeLine("LoRA", loras.isEmpty ? "—" : loras.map { "\(name($0.path)) (\($0.strength))" }.joined(separator: "\n"))
                }
                .font(.caption)
                .textSelection(.enabled)
            }

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

    private var list: some View {
        Group {
            if entries.isEmpty && diskEntries.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "internaldrive")
                        .font(.system(size: 28, weight: .light))
                        .foregroundStyle(.tertiary)
                    Text(busy
                         ? loc.t("Consultando la caché…", "Checking the cache…")
                         : !diskLoaded
                         ? loc.t("No se pudo consultar la caché", "Cache status unavailable")
                         : loc.t("Ninguna entrada en caché", "No cached entries"))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                    if !entries.isEmpty {
                        Text(loc.t("Caché activa", "Active cache")).font(.headline)
                            ForEach(entries) { entry in
                                HStack(alignment: .top, spacing: 12) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(roleLabel(diskEntries.first { $0.scopeID == storageScopeID && $0.cacheID == entry.id }?.roles ?? 0))
                                            .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                                        Text(entry.prompt.isEmpty ? "—" : entry.prompt)
                                            .fixedSize(horizontal: false, vertical: true)
                                            .textSelection(.enabled)
                                        HStack(spacing: 8) {
                                            if entry.width > 0 && entry.height > 0 {
                                                Text("\(entry.width)×\(entry.height)")
                                                    .font(.caption.monospaced())
                                            }
                                            Text(ageLabel(entry.ageSeconds))
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                        if let clipSkip = entry.clipSkip {
                                            Text("CLIP skip: \(clipSkip)").font(.caption.monospaced())
                                        }
                                        if let masked = entry.zeroOutMasked {
                                            Text("zero_out_masked: \(masked ? "true" : "false")").font(.caption.monospaced())
                                        }
                                    }
                                    Spacer()
                                    Button(role: .destructive) {
                                        Task { await delete(entry.id) }
                                    } label: {
                                        Image(systemName: "trash")
                                    }
                                    .buttonStyle(.borderless)
                                    .disabled(busy || server.jobsInFlight > 0)
                                }
                                .padding(14)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
                            }
                    }
                    if !diskEntries.isEmpty {
                        Text(loc.t("Entradas de disco · %@", "Disk entries · %@", "\(diskEntries.count)"))
                            .font(.headline).padding(.top, 4)
                        ForEach(diskEntries) { entry in
                            VStack(alignment: .leading, spacing: 10) {
                                HStack(alignment: .top) {
                                    Label(roleLabel(entry.roles), systemImage: "text.bubble")
                                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                                    Spacer()
                                    Text(entry.created.formatted(date: .abbreviated, time: .shortened))
                                        .font(.caption).foregroundStyle(.secondary)
                                    Button(role: .destructive) {
                                        Task { await deleteDisk(entry) }
                                    } label: {
                                        Image(systemName: "trash")
                                    }
                                    .buttonStyle(.borderless)
                                    .help(loc.t("Eliminar entrada de disco", "Delete disk entry"))
                                    .disabled(busy || server.jobsInFlight > 0 || (server.isActive && !server.isRunning))
                                }
                                Text(entry.prompt.isEmpty ? "—" : entry.prompt)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .textSelection(.enabled)
                                Text("\(entry.width)×\(entry.height) · CLIP skip: \(entry.clipSkip) · zero_out_masked: \(entry.zeroOutMasked ? "true" : "false")")
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                                VStack(alignment: .leading, spacing: 4) {
                                    scopeLine(loc.t("Codificador de texto", "Text encoder"), entry.textEncoder)
                                    scopeLine(loc.t("Modelo de difusión", "Diffusion model"), entry.diffusionModel)
                                    scopeLine(loc.t("Tokenizador", "Tokenizer"), entry.tokenizer)
                                }.font(.caption)
                                DisclosureGroup(loc.t("Ámbito completo", "Full scope")) {
                                    Text(entry.scope).font(.caption.monospaced())
                                        .fixedSize(horizontal: false, vertical: true)
                                        .textSelection(.enabled)
                                }
                                .font(.caption)
                            }
                            .padding(14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
                        }
                    }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }

    private func scopeLine(_ label: String, _ value: String) -> some View {
        Text("\(label): \(name(value))")
            .fixedSize(horizontal: false, vertical: true)
    }

    private func name(_ path: String) -> String {
        path.isEmpty ? "—" : URL(fileURLWithPath: path).lastPathComponent
    }

    private func roleLabel(_ roles: Int) -> String {
        switch roles {
        case 1: return loc.t("Texto positivo", "Positive prompt")
        case 2: return loc.t("Texto negativo", "Negative prompt")
        case 3: return loc.t("Texto positivo y negativo", "Positive and negative prompt")
        default: return loc.t("Texto · dirección no registrada", "Prompt · direction not recorded")
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
        let currentRefresh = UUID()
        refreshID = currentRefresh
        busy = true
        defer { if refreshID == currentRefresh { busy = false } }
        do {
            let directory = AppSupport.directory.appendingPathComponent("imagen-conditioning-cache", isDirectory: true)
            let disk = try await Task.detached(priority: .utility) {
                try ImageConditioningDiskCache.scan(directory: directory)
            }.value
            guard !Task.isCancelled, refreshID == currentRefresh else { return }
            diskEntries = disk.entries
            diskUnreadable = disk.unreadable
            diskLoaded = true
            if !server.isRunning {
                entries = []; size = 0; capacity = 0; modelLoaded = false; hasSnapshot = false
                status = diskUnreadable > 0
                    ? loc.t("%@ archivos de caché ilegibles", "%@ unreadable cache files", "\(diskUnreadable)")
                    : loc.t("Actualizado", "Updated")
                return
            }
            var snap = try await server.fetchConditioningCache()
            guard !Task.isCancelled, refreshID == currentRefresh else { return }
            if purgeFirst, server.jobsInFlight == 0, retentionDays > 0, snap.modelLoaded {
                let removed = try await server.purgeConditioningCache(maxAgeSeconds: retentionDays * 86_400)
                guard !Task.isCancelled, refreshID == currentRefresh else { return }
                if removed > 0 {
                    snap = try await server.fetchConditioningCache()
                    guard !Task.isCancelled, refreshID == currentRefresh else { return }
                    let updatedDisk = try await Task.detached(priority: .utility) {
                        try ImageConditioningDiskCache.scan(directory: directory)
                    }.value
                    guard !Task.isCancelled, refreshID == currentRefresh else { return }
                    diskEntries = updatedDisk.entries
                    diskUnreadable = updatedDisk.unreadable
                }
            }
            capacity = snap.capacity
            size = snap.size
            storageScopeID = snap.storageScopeID
            modelLoaded = snap.modelLoaded
            hasSnapshot = true
            entries = snap.entries.map {
                TECacheEntry(id: $0.id, prompt: $0.prompt, width: $0.width,
                             height: $0.height, clipSkip: $0.clipSkip,
                             zeroOutMasked: $0.zeroOutMasked, ageSeconds: $0.ageSeconds)
            }
            textEncoder = snap.textEncoder
            diffusionModel = snap.diffusionModel
            tokenizer = snap.tokenizer
            loras = snap.loras
            status = diskUnreadable > 0
                ? loc.t("%@ archivos de caché ilegibles", "%@ unreadable cache files", "\(diskUnreadable)")
                : snap.isSnapshot
                ? loc.t("Instantánea durante la generación", "Snapshot during generation")
                : loc.t("Actualizado", "Updated")
        } catch {
            guard !Task.isCancelled, refreshID == currentRefresh else { return }
            if let serverError = error as? ImageWebUIServer.ServerError,
               case .cacheSnapshotUnavailable = serverError {
                status = loc.t("Esperando la instantánea de caché", "Waiting for cache snapshot")
            } else {
                status = error.localizedDescription
            }
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
    private func deleteDisk(_ entry: ImageConditioningDiskCache.Entry? = nil) async {
        busy = true
        defer { busy = false }
        do {
            guard server.jobsInFlight == 0, !server.isActive || server.isRunning else {
                throw ImageWebUIServer.ServerError.busy
            }
            if server.isRunning {
                try await server.deleteDiskConditioningCache(scope: entry?.scopeID, id: entry?.cacheID)
            } else {
                let directory = AppSupport.directory.appendingPathComponent("imagen-conditioning-cache", isDirectory: true)
                if let entry {
                    try ImageConditioningDiskCache.remove(directory: directory, scope: entry.scopeID, id: entry.cacheID)
                } else {
                    try ImageConditioningDiskCache.removeAll(directory: directory)
                }
            }
            await refresh(purgeFirst: false)
        } catch {
            status = error.localizedDescription
        }
    }
}
