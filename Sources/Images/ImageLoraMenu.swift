import SwiftUI

/// Multi-select LoRAs: each toggle adds or removes `<lora:file:weight>` in the prompt.
/// The engine loads tags from `--lora-model-dir`, so this menu only edits prompt text.
struct ImageLoraMenu: View {
    @Binding var prompt: String
    @EnvironmentObject var loc: Localizer

    @State private var files: [URL] = []

    var body: some View {
        Menu {
            if files.isEmpty {
                Text(loc.t("Pon los archivos en vision/loras o imagen/lora",
                           "Put files in vision/loras or imagen/lora"))
            } else {
                ForEach(files, id: \.path) { file in
                    let name = file.deletingPathExtension().lastPathComponent
                    let on = isActive(name)
                    Button {
                        toggle(name)
                    } label: {
                        HStack {
                            Text(name)
                            Spacer(minLength: 12)
                            if on { Image(systemName: "checkmark") }
                        }
                    }
                }
                Divider()
                if !activeNames().isEmpty {
                    Button(loc.t("Quitar todos los LoRA", "Clear all LoRAs"), role: .destructive) {
                        clearAll()
                    }
                }
                Text(loc.t("LCM y Turbo: pocos pasos y guía cerca de 1.5",
                           "LCM and Turbo: few steps and guidance near 1.5"))
            }
            Divider()
            Button(loc.t("Abrir la carpeta", "Open the folder"), systemImage: "folder") {
                NSWorkspace.shared.open(ImageGenPool.loraDirectoryForOpen())
            }
            Button(loc.t("Actualizar lista", "Refresh list"), systemImage: "arrow.clockwise") {
                refresh()
            }
        } label: {
            Label(menuTitle, systemImage: "slider.horizontal.below.square.filled.and.square")
        }
        .help(loc.t("Activa varios LoRA a la vez. El peso se edita en la etiqueta (<lora:nombre:0.8>). LCM/Turbo: pocos pasos y guía ~1.5.",
                    "Enable several LoRAs at once. Edit the weight in the tag (<lora:name:0.8>). LCM/Turbo: few steps and guidance ~1.5."))
        .task { refresh() }
    }

    private var menuTitle: String {
        let n = activeNames().count
        return n == 0 ? "LoRA" : "LoRA (\(n))"
    }

    private func refresh() {
        files = ImageGenPool.loraDirectory().map { ImageGenPool.loraFiles(in: $0) } ?? []
    }

    private func activeNames() -> [String] {
        Self.loraTags(in: prompt).map(\.name)
    }

    private func isActive(_ name: String) -> Bool {
        activeNames().contains { $0.caseInsensitiveCompare(name) == .orderedSame }
    }

    private func toggle(_ name: String) {
        if isActive(name) { remove(name) } else { insert(name) }
    }

    private func insert(_ name: String) {
        let tag = "<lora:\(name):0.8>"
        if prompt.isEmpty { prompt = tag }
        else if prompt.hasSuffix(" ") { prompt += tag }
        else { prompt += " " + tag }
    }

    private func remove(_ name: String) {
        for tag in Self.loraTags(in: prompt).reversed() where tag.name.caseInsensitiveCompare(name) == .orderedSame {
            let start = prompt.index(prompt.startIndex, offsetBy: tag.range.location)
            let end = prompt.index(start, offsetBy: tag.range.length)
            prompt.removeSubrange(start..<end)
        }
        prompt = prompt
            .replacingOccurrences(of: #"  +"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func clearAll() {
        for tag in Self.loraTags(in: prompt).reversed() {
            let start = prompt.index(prompt.startIndex, offsetBy: tag.range.location)
            let end = prompt.index(start, offsetBy: tag.range.length)
            prompt.removeSubrange(start..<end)
        }
        prompt = prompt
            .replacingOccurrences(of: #"  +"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private struct LoraTag {
        let name: String
        let range: NSRange
    }

    private static func loraTags(in text: String) -> [LoraTag] {
        guard let regex = try? NSRegularExpression(pattern: #"<lora:([^:>]+)(?::[^>]*)?>"#) else { return [] }
        let ns = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).compactMap { match in
            guard match.numberOfRanges >= 2 else { return nil }
            return LoraTag(name: ns.substring(with: match.range(at: 1)), range: match.range)
        }
    }
}
