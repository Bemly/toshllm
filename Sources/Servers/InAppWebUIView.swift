// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AppKit
import SwiftUI
import WebKit

/// Target for the main-window WebUI mode (llama / KVMem / Strata / sd.cpp).
struct WebUITarget: Codable, Hashable {
    /// llama-server chat URL (may include lang/gpu/backend query).
    var llamaURL: URL
    var title: String
}

enum InAppWebUISurface: String, CaseIterable, Identifiable {
    case llama
    case kvmem
    case strata
    case sd

    var id: String { rawValue }

    func title(_ loc: Localizer) -> String {
        switch self {
        case .llama: return "llama.cpp"
        case .kvmem: return "KVMem"
        case .strata: return "Strata"
        case .sd: return "sd.cpp"
        }
    }
}

struct InAppWebUIView: View {
    let llamaURL: URL
    let title: String
    @EnvironmentObject private var loc: Localizer
    @EnvironmentObject private var manager: ServerManager
    @ObservedObject private var imageWebUI = ImageWebUIServer.shared
    @State private var surface: InAppWebUISurface = .llama
    @State private var reloadTick = 0
    /// When false, the representable is removed so WKWebView is dismantled and memory can drop.
    @State private var webMounted = true

    init(target: WebUITarget) {
        self.llamaURL = target.llamaURL
        self.title = target.title
    }

    private var llamaLive: Bool {
        let port = URLComponents(url: llamaURL, resolvingAgainstBaseURL: false)?.port
            ?? (llamaURL.port ?? 8080)
        return manager.servers.contains { $0.runningPort == port }
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            if webMounted, let page = pageURL(for: surface) {
                InAppWebUIRepresentable(url: page, reloadTick: reloadTick)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .id(surface) // tear down the previous page when switching engines/UIs
            } else {
                unavailablePlaceholder
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle(title)
        .frame(minWidth: 720, minHeight: 480)
        .onAppear {
            webMounted = true
            if !llamaLive, imageWebUI.isRunning {
                surface = .sd
            }
        }
        .onDisappear {
            // Drop the WKWebView as soon as the window goes away.
            webMounted = false
        }
    }

    private var toolbar: some View {
        HStack(spacing: 8) {
            Button {
                reloadTick &+= 1
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .help(loc.t("Recargar", "Reload"))
            .disabled(pageURL(for: surface) == nil)

            Picker(loc.t("Interfaz", "Interface"), selection: $surface) {
                ForEach(InAppWebUISurface.allCases) { item in
                    Text(item.title(loc)).tag(item)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 420)
            .onChange(of: surface) { _, next in
                if next == .sd, !imageWebUI.isRunning {
                    imageWebUI.start()
                }
                reloadTick &+= 1
            }

            if surface == .sd, !imageWebUI.isRunning {
                Text(sdStatusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 8)

            Button {
                if let page = pageURL(for: surface) {
                    NSWorkspace.shared.open(page)
                }
            } label: {
                Label(loc.t("Abrir en el navegador", "Open in Browser"), systemImage: "safari")
            }
            .disabled(pageURL(for: surface) == nil)
            .help(loc.t("Abrir esta página en el navegador del sistema",
                        "Open this page in the system browser"))
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var unavailablePlaceholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "macwindow")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text(placeholderMessage)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if surface == .sd, ImageWebUIServer.bundledBinary() != nil, !imageWebUI.isRunning {
                Button(loc.t("Iniciar sd.cpp WebUI", "Start sd.cpp WebUI")) {
                    imageWebUI.start()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        }
        .padding(24)
    }

    private var placeholderMessage: String {
        switch surface {
        case .llama, .kvmem, .strata:
            return loc.t("Inicia el servidor de texto para ver esta interfaz.",
                         "Start the text server to view this interface.")
        case .sd:
            if ImageWebUIServer.bundledBinary() == nil {
                return loc.t("No hay sd-server en este build.", "sd-server is missing from this build.")
            }
            return loc.t("El WebUI de sd.cpp no está en marcha.", "The sd.cpp WebUI is not running.")
        }
    }

    private var sdStatusText: String {
        switch imageWebUI.state {
        case .starting: return loc.t("Arrancando…", "Starting…")
        case .failed(let msg): return msg
        case .stopped: return loc.t("Detenido", "Stopped")
        case .running: return ""
        }
    }

    private func pageURL(for surface: InAppWebUISurface) -> URL? {
        switch surface {
        case .llama:
            return llamaLive ? urlReplacingPath(llamaURL, path: "/") : nil
        case .kvmem:
            return llamaLive ? urlReplacingPath(llamaURL, path: "/kvmem/") : nil
        case .strata:
            return llamaLive ? urlReplacingPath(llamaURL, path: "/strata/") : nil
        case .sd:
            return imageWebUI.url
        }
    }

    private func urlReplacingPath(_ base: URL, path: String) -> URL {
        var comps = URLComponents(url: base, resolvingAgainstBaseURL: false) ?? URLComponents()
        comps.path = path
        return comps.url ?? base
    }
}

private struct InAppWebUIRepresentable: NSViewRepresentable {
    let url: URL
    let reloadTick: Int

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        // Non-persistent store so closing the window can drop page data with the view.
        configuration.websiteDataStore = .nonPersistent()
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.navigationDelegate = context.coordinator
        view.uiDelegate = context.coordinator
        context.coordinator.attach(view)
        context.coordinator.load(url, in: view, force: true)
        return view
    }

    static func dismantleNSView(_ view: WKWebView, coordinator: Coordinator) {
        coordinator.tearDown(view)
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        let force = context.coordinator.lastTick != reloadTick
        context.coordinator.lastTick = reloadTick
        context.coordinator.load(url, in: view, force: force)
    }

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        private var loaded: URL?
        private weak var webView: WKWebView?
        var lastTick = -1

        func attach(_ view: WKWebView) {
            webView = view
        }

        func load(_ url: URL, in view: WKWebView, force: Bool = false) {
            if !force, loaded == url { return }
            loaded = url
            view.load(URLRequest(url: url))
        }

        func tearDown(_ view: WKWebView) {
            view.stopLoading()
            view.navigationDelegate = nil
            view.uiDelegate = nil
            loaded = nil
            webView = nil
            // Drop document + JS heap for this view.
            view.loadHTMLString("", baseURL: nil)
            let store = view.configuration.websiteDataStore
            store.removeData(ofTypes: WKWebsiteDataStore.allWebsiteDataTypes(),
                             modifiedSince: .distantPast) { }
        }

        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let url = action.request.url else {
                decisionHandler(.cancel)
                return
            }
            if Self.isLocal(url) {
                decisionHandler(.allow)
                return
            }
            if action.navigationType == .linkActivated, Self.isHTTP(url) {
                NSWorkspace.shared.open(url)
            }
            decisionHandler(.cancel)
        }

        func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                     for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
            if let url = action.request.url {
                if Self.isLocal(url) {
                    webView.load(URLRequest(url: url))
                } else if Self.isHTTP(url) {
                    NSWorkspace.shared.open(url)
                }
            }
            return nil
        }

        private static func isHTTP(_ url: URL) -> Bool {
            ["http", "https"].contains(url.scheme?.lowercased() ?? "")
        }

        private static func isLocal(_ url: URL) -> Bool {
            guard isHTTP(url) else { return false }
            let host = (url.host ?? "").lowercased()
            return host == "127.0.0.1" || host == "localhost" || host == "::1"
        }
    }
}
