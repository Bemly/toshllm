// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// Persistent gallery of generated images (survives relaunch via sidecars / latent JSON).
struct ImageGalleryBrowser: View {
    @ObservedObject var pool: ImageGenPool
    @EnvironmentObject var loc: Localizer
    @EnvironmentObject var models: ModelStore
    var onImport: ((ImageGalleryRecord) -> Void)?

    @State private var selectedID: GeneratedImage.ID?
    @State private var order: ImageGalleryOrder = .recent
    @AppStorage(SettingsKeys.imagenQueueGrid) private var grid = true

    private var ordered: [GeneratedImage] {
        order == .recent ? pool.libraryGallery : pool.libraryGallery.reversed()
    }

    private var selected: GeneratedImage? {
        ordered.first(where: { $0.id == selectedID }) ?? ordered.first
    }

    var body: some View {
        VStack(spacing: 14) {
            toolbar
            if pool.libraryGallery.isEmpty {
                emptyState
            } else {
                GeometryReader { geo in
                    HStack(alignment: .top, spacing: 16) {
                        preview
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        if geo.size.width >= 640 {
                            thumbStrip
                                .frame(width: min(280, max(200, geo.size.width * 0.28)))
                        }
                    }
                }
            }
        }
        .onAppear {
            pool.reloadPersistedGallery(from: models)
            if selectedID == nil { selectedID = ordered.first?.id }
        }
        .onChange(of: pool.libraryGallery.map(\.id)) {
            if selectedID == nil || !pool.libraryGallery.contains(where: { $0.id == selectedID }) {
                selectedID = ordered.first?.id
            }
        }
        .onChange(of: selectedID) {
            pool.continueNotice = ""
        }
    }

    private var toolbar: some View {
        HStack(spacing: 10) {
            Text(loc.t("%@ imágenes", "%@ images", "\(pool.libraryGallery.count)"))
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)
            Spacer()
            Picker("", selection: $order) {
                Text(loc.t("Recientes", "Recent")).tag(ImageGalleryOrder.recent)
                Text(loc.t("Más antiguas", "Oldest")).tag(ImageGalleryOrder.oldest)
            }
            .labelsHidden()
            .frame(width: 140)
            FeedLayoutPicker(grid: $grid)
            Button {
                pool.reloadPersistedGallery(from: models)
            } label: {
                Label(loc.t("Actualizar", "Refresh"), systemImage: "arrow.clockwise")
            }
            .controlSize(.small)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 36, weight: .light))
                .foregroundStyle(.tertiary)
            Text(loc.t("La galería está vacía", "Gallery is empty"))
                .font(.title3.weight(.semibold))
            Text(loc.t("Las imágenes generadas se guardan aquí y sobreviven al reinicio. Importa sus parámetros a una instancia con un clic.",
                        "Generated images stay here across relaunches. Import their parameters into an instance with one click."))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var preview: some View {
        VStack(spacing: 0) {
            if let item = selected {
                Image(nsImage: item.image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 12))
                metaBlock(for: item)
                actions(for: item)
            } else {
                Spacer()
            }
        }
    }

    private func metaBlock(for item: GeneratedImage) -> some View {
        let record = ImageGalleryStore.enrich(
            item.record
                ?? ImageGalleryStore.loadSidecar(for: item.url)
                ?? ImageGalleryStore.record(forImage: item.url)
                ?? ImageGalleryRecord(imagePath: item.url.path))
        let (rw, rh) = item.resolvedSize
        let w = rw > 0 ? rw : record.width
        let h = rh > 0 ? rh : record.height
        let prompt = item.prompt.nilIfEmpty ?? record.prompt.nilIfEmpty
        let seed = item.seed >= 0 ? item.seed : record.seed
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Text(w > 0 && h > 0 ? "\(w) × \(h)" : loc.t("Tamaño desconocido", "Unknown size"))
                    .font(.caption.monospaced())
                if seed >= 0 {
                    Text("#\(seed)").font(.caption.monospaced()).foregroundStyle(.secondary)
                }
                if let model = record.modelID.nilIfEmpty {
                    Text(model).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                if item.canContinue, let ended = item.endedAtStep, let full = item.fullSteps {
                    Text("\(ended)/\(full)").font(.caption.monospaced()).foregroundStyle(.secondary)
                }
                Spacer()
            }
            if let prompt {
                Text(prompt)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(4)
            } else {
                Text(loc.t("Sin parámetros recuperables", "No recoverable parameters"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            if !pool.continueNotice.isEmpty {
                Text(imageFailureText(pool.continueNotice, loc))
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
    }

    private func actions(for item: GeneratedImage) -> some View {
        let record = ImageGalleryStore.enrich(
            item.record
                ?? ImageGalleryStore.record(forImage: item.url)
                ?? ImageGalleryStore.loadSidecar(for: item.url)
                ?? ImageGalleryRecord(imagePath: item.url.path))
        return HStack(spacing: 10) {
            Spacer(minLength: 8)
            if item.canContinue {
                Button {
                    if let config = pool.configs.first {
                        pool.continueFull(from: item, on: config)
                    }
                } label: {
                    Label(loc.t("Continuar", "Continue"), systemImage: "forward.end")
                }
                .disabled(pool.configs.isEmpty)
                .help(loc.t("Continúa el borrador hasta los pasos de borrador actuales de la instancia.",
                            "Resume the draft to the instance's current draft steps."))
            }
            if record.hasUsableParams || record.width > 0 {
                Button {
                    pool.importRecord(record)
                    onImport?(record)
                } label: {
                    Label(loc.t("Importar a instancia", "Import to instance"),
                          systemImage: "square.and.arrow.down.on.square")
                }
                .help(loc.t("Copia prompt, semilla, tamaño y modelo a la instancia activa.",
                            "Copy prompt, seed, size, and model into the active instance."))
            }
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([item.url])
            } label: {
                Label(loc.t("Mostrar en Finder", "Reveal in Finder"), systemImage: "folder")
            }
            Button(role: .destructive) {
                pool.removeFromLibrary(item.id)
            } label: {
                Label(loc.t("Eliminar", "Delete"), systemImage: "trash")
            }
        }
        .controlSize(.small)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .overlay(alignment: .top) { Divider() }
    }

    private var thumbStrip: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: grid ? 88 : 200), spacing: 8)], spacing: 8) {
                ForEach(ordered) { item in
                    Button {
                        selectedID = item.id
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Image(nsImage: item.image)
                                .resizable()
                                .scaledToFill()
                                .frame(maxWidth: .infinity)
                                .frame(height: grid ? 88 : 120)
                                .clipped()
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .strokeBorder(item.id == selected?.id ? Color.appAccent : .clear, lineWidth: 2))
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                            Text(item.sizeLabel)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                            if !grid {
                                Text(item.prompt.isEmpty ? item.url.lastPathComponent : item.prompt)
                                    .font(.caption2)
                                    .lineLimit(2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(4)
        }
    }
}

private extension String {
    var nilIfEmpty: String? {
        let t = trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }
}
