// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Pure CLI → warm sd-server request mapping (no process I/O).
enum ImageServerHTTP {
    static let contextValueFlags: Set<String> = [
        "--model", "--diffusion-model", "--vae", "--audio-vae", "--embeddings-connectors",
        "--llm", "--llm_vision", "--llm-vision", "--t5xxl", "--clip_l", "--clip-l",
        "--tokenizer", "--lora-model-dir", "--backend", "--params-backend", "--control-net",
        "--taesd", "--high-noise-diffusion-model", "--clip-vision", "--photo-maker",
        "--model-dir", "--auto-fit", "--conditioning-cache-size",
    ]
    static let contextBoolFlags: Set<String> = [
        "--release-te-after-encode", "--keep-te", "--offload-to-cpu", "--diffusion-fa",
        "--taesd-preview-only", "--vae-on-cpu", "--clip-on-cpu", "--mmap",
    ]
    static let genBoolFlags: Set<String> = ["--vae-tiling"]
    static let valueFlags: Set<String> = contextValueFlags.union([
        "-p", "--prompt", "-n", "--negative-prompt", "--cfg-scale", "--img-cfg-scale",
        "--steps", "--sigmas", "-W", "--width", "-H", "--height", "--seed",
        "--preview", "--preview-interval", "--preview-path", "-o", "--output",
        "-i", "--init-img", "--strength", "--mask", "-r", "--ref-image",
        "--ref-image-args", "--sampling-method", "--scheduler", "--flow-shift",
        "--slg-scale", "--skip-layer-start", "--skip-layer-end", "--skip-layers",
        "--control-image", "--control-strength", "--control-type",
        "--batch-count", "--output-begin-idx", "--start-at-step", "--end-at-step",
        "--add-noise", "--latent-in", "--latent-out", "--cache-mode", "--cache-option",
        "--vae-tile-size", "--guidance",
    ])

    struct Partition {
        var context: [String]
        var values: [String: [String]]
        var bools: Set<String>
    }

    static func partition(_ args: [String]) -> Partition {
        var context: [String] = []
        var values: [String: [String]] = [:]
        var bools = Set<String>()
        var i = 0
        while i < args.count {
            let flag = args[i]
            if contextBoolFlags.contains(flag) {
                context.append(flag)
                i += 1
                continue
            }
            if genBoolFlags.contains(flag) {
                bools.insert(flag)
                i += 1
                continue
            }
            let takesValue = valueFlags.contains(flag)
                || (flag.hasPrefix("-") && i + 1 < args.count && !args[i + 1].hasPrefix("-"))
            if !takesValue {
                if flag.hasPrefix("-") { context.append(flag) }
                i += 1
                continue
            }
            guard i + 1 < args.count else { i += 1; continue }
            let value = args[i + 1]
            if contextValueFlags.contains(flag) {
                context += [flag, value]
            } else {
                values[flag, default: []].append(value)
            }
            i += 2
        }
        return Partition(context: context, values: values, bools: bools)
    }

    static func requestBody(from values: [String: [String]],
                            bools: Set<String>,
                            output: URL,
                            loadFiles: Bool = true) throws -> [String: Any] {
        func first(_ keys: String...) -> String? {
            for key in keys {
                if let v = values[key]?.first, !v.isEmpty { return v }
            }
            return nil
        }
        func number(_ keys: String...) -> Double? {
            for key in keys {
                if let raw = first(key) { return Double(raw) }
            }
            return nil
        }
        var prompt = first("-p", "--prompt") ?? ""
        let loras = extractLoras(&prompt)
        var body: [String: Any] = [
            "prompt": prompt,
            "negative_prompt": first("-n", "--negative-prompt") ?? "",
            "width": Int(number("-W", "--width") ?? 512),
            "height": Int(number("-H", "--height") ?? 512),
            "seed": Int(number("--seed") ?? -1),
            "batch_count": Int(number("--batch-count") ?? 1),
            "strength": number("--strength") ?? 0.75,
            "output_format": output.pathExtension.lowercased() == "jpg" ? "jpeg" : output.pathExtension.lowercased(),
            "output_compression": 100,
            "preview": first("--preview") ?? "none",
            "preview_interval": Int(number("--preview-interval") ?? 3),
        ]
        if let refs = first("--ref-image-args") { body["ref_image_args"] = refs }
        if let mode = first("--cache-mode") { body["cache_mode"] = mode }
        if let option = first("--cache-option") { body["cache_option"] = option }
        if let start = number("--start-at-step") { body["start_at_step"] = Int(start) }
        if let end = number("--end-at-step") { body["end_at_step"] = Int(end) }
        if let noise = first("--add-noise") { body["add_noise"] = noise }
        if let latent = first("--latent-in") { body["latent_in"] = latent }
        if let latent = first("--latent-out") { body["latent_out"] = latent }
        if let control = number("--control-strength") { body["control_strength"] = control }
        if let type = first("--control-type") { body["control_type"] = type }
        if !loras.isEmpty { body["lora"] = loras }

        var guidance: [String: Any] = [:]
        if let cfg = number("--cfg-scale") {
            guidance["txt_cfg"] = cfg
            guidance["img_cfg"] = number("--img-cfg-scale") ?? cfg
        }
        var slg: [String: Any] = [:]
        if let scale = number("--slg-scale") { slg["scale"] = scale }
        if let start = number("--skip-layer-start") { slg["layer_start"] = start }
        if let end = number("--skip-layer-end") { slg["layer_end"] = end }
        if let layers = first("--skip-layers") {
            slg["layers"] = layers.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        }
        if !slg.isEmpty { guidance["slg"] = slg }
        var sample: [String: Any] = [:]
        if let steps = number("--steps") { sample["sample_steps"] = Int(steps) }
        if let method = first("--sampling-method") { sample["sample_method"] = method }
        if let scheduler = first("--scheduler") { sample["scheduler"] = scheduler }
        if let shift = number("--flow-shift") { sample["flow_shift"] = shift }
        if let sigmas = first("--sigmas") {
            sample["custom_sigmas"] = sigmas.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        }
        if !guidance.isEmpty { sample["guidance"] = guidance }
        if !sample.isEmpty { body["sample_params"] = sample }

        var tiling: [String: Any] = ["enabled": bools.contains("--vae-tiling")]
        if let tile = first("--vae-tile-size") {
            if let n = Int(tile) {
                tiling["tile_size_w"] = n
                tiling["tile_size_h"] = n
            } else {
                let parts = tile.split(separator: "x")
                if parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]) {
                    tiling["tile_size_w"] = w
                    tiling["tile_size_h"] = h
                }
            }
        }
        body["vae_tiling_params"] = tiling

        if loadFiles {
            if let path = first("-i", "--init-img") { body["init_image"] = try base64File(path) }
            if let path = first("--mask") { body["mask_image"] = try base64File(path) }
            if let path = first("--control-image") { body["control_image"] = try base64File(path) }
            if let refs = values["-r"] ?? values["--ref-image"], !refs.isEmpty {
                body["ref_images"] = try refs.map { try base64File($0) }
            }
        }
        if (body["output_format"] as? String)?.isEmpty == true { body["output_format"] = "png" }
        return body
    }

    static func base64File(_ path: String) throws -> String {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return data.base64EncodedString()
    }

    /// Server APIs ignore `<lora:name:weight>` inside the prompt.
    static func extractLoras(_ prompt: inout String, loraDirectory: URL? = ImageGenPool.loraDirectory()) -> [[String: Any]] {
        guard let regex = try? NSRegularExpression(pattern: #"<lora:([^:>]+):([^>]+)>"#) else { return [] }
        let ns = prompt as NSString
        let matches = regex.matches(in: prompt, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return [] }
        var items: [[String: Any]] = []
        for match in matches.reversed() {
            let name = ns.substring(with: match.range(at: 1))
            let weight = Double(ns.substring(with: match.range(at: 2))) ?? 1
            if let path = resolveLora(name, in: loraDirectory) {
                items.append(["path": path, "multiplier": weight])
            }
            prompt.removeSubrange(Range(match.range, in: prompt)!)
        }
        return items.reversed()
    }

    static func resolveLora(_ name: String, in dir: URL?) -> String? {
        let file = URL(fileURLWithPath: name).lastPathComponent
        if file.contains(".") { return file }
        guard let dir else { return file }
        let exts = ["safetensors", "gguf", "pt", "ckpt"]
        for ext in exts {
            let candidate = dir.appendingPathComponent(file + "." + ext)
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate.lastPathComponent }
        }
        return file
    }
}
