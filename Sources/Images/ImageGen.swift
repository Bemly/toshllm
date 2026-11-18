// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import Metal
import Combine
import ImageIO
import UniformTypeIdentifiers

// Local text-to-image via a bundled stable-diffusion.cpp engine. A model is one
// or more files (a full checkpoint, or a diffusion transformer plus its VAE and
// text encoders).
//
// Storage matches sd-server / sd.cpp WebUI (`--model-dir models/vision`):
//   vision/checkpoints|diffusion_models|text_encoders|tokenizers|vae|loras/
// Generated PNGs still go under `imagen/` (outputs only, not weights).

/// One downloadable piece of an image model, tagged with the sd-cli flag it maps to.
struct ImageGenComponent: Identifiable {
    enum Kind { case checkpoint, diffusion, vae, audioVAE, connectors, textEncoder, llmVision, t5, clipL, tokenizer }
    let kind: Kind
    let urlString: String
    let fileName: String
    let sizeGB: Double
    /// Names the catalog downloaded this file under before; a copy already on disk still counts.
    var formerFileNames: [String] = []
    var id: String { fileName }

    /// sd-server category folder under `models/vision/` (see examples/server/model_catalog.cpp).
    var sdCategory: String {
        switch kind {
        case .checkpoint:  return "checkpoints"
        case .diffusion:   return "diffusion_models"
        case .vae, .audioVAE: return "vae"
        case .tokenizer:   return "tokenizers"
        case .textEncoder, .llmVision, .t5, .clipL, .connectors: return "text_encoders"
        }
    }

    /// The sd-cli argument this file is passed as.
    var flag: String {
        switch kind {
        case .checkpoint:  return "--model"
        case .diffusion:   return "--diffusion-model"
        case .vae:         return "--vae"
        case .audioVAE:    return "--audio-vae"
        case .connectors:  return "--embeddings-connectors"
        case .textEncoder: return "--llm"
        case .llmVision:   return "--llm_vision"
        case .t5:          return "--t5xxl"
        case .clipL:       return "--clip_l"
        case .tokenizer:   return "--tokenizer"
        }
    }

    func label(_ spanish: Bool) -> String {
        switch kind {
        case .checkpoint:  return spanish ? "Modelo" : "Model"
        case .diffusion:   return spanish ? "Modelo de difusión" : "Diffusion model"
        case .vae:         return "VAE"
        case .audioVAE:    return spanish ? "VAE de audio" : "Audio VAE"
        case .connectors:  return spanish ? "Conectores" : "Connectors"
        case .llmVision:   return spanish ? "Proyector de visión" : "Vision projector"
        case .tokenizer:   return "Tokenizer"
        case .textEncoder, .t5, .clipL: return spanish ? "Codificador de texto" : "Text encoder"
        }
    }

    /// Install path under the vision root: `vision/<sdCategory>/<fileName>`.
    func path(in visionRoot: URL) -> URL {
        if fileName.hasPrefix("/") { return URL(fileURLWithPath: fileName) }
        return visionRoot
            .appendingPathComponent(sdCategory, isDirectory: true)
            .appendingPathComponent(fileName)
    }
}

/// An image-generation model the app can install and run.
struct ImageGenModel: Identifiable {
    let name: String
    let detailES: String
    let detailEN: String
    var components: [ImageGenComponent]
    /// Steps the model is tuned for (Turbo/distilled models need very few).
    let defaultSteps: Int
    /// Guidance scale the model expects (Turbo models run at 1.0).
    let cfgScale: Double
    /// Rough usable VRAM (GB) below which it won't fit well. Drives the pick. Measured for
    /// SD 1.5, SDXL Turbo and Z-Image (resident weights plus the graph at 1024x1024, 08-19);
    /// derived from the same formula for the rest, cross-checked against what each model
    /// publishes for a comparable quantisation with the text encoder off the card.
    let minVRAMGB: Double
    /// False = shown and usable, but never the auto-recommended pick.
    var recommendable: Bool = true
    /// Largest long-edge (px) to offer as a quality guard: UNet models blur past
    /// their native size. VRAM is handled separately by attnVRAMSq.
    var maxLongEdge: Int = 2048
    /// Lower ceiling for a card that also draws the desktop, where a long diffusion step
    /// starves the window server until the GPU is reset. 0 = same as `maxLongEdge`.
    var displayMaxLongEdge: Int = 0

    /// The ceiling on the card the run goes to.
    func maxLongEdge(drivesDisplay: Bool) -> Int {
        drivesDisplay && displayMaxLongEdge > 0 ? min(maxLongEdge, displayMaxLongEdge) : maxLongEdge
    }
    /// Long edge (px) this model is known to hold up to. Past it the composition
    /// starts to repeat (two horizons, duplicated subjects), which is the model's
    /// limit and not the app's, so the UI says so instead of leaving the user
    /// guessing. 0 means no note: only set it where there is a published figure or
    /// a measurement, never a guess, or the warning cries wolf.
    var nativeLongEdge: Int = 0
    /// Trained on square frames only, so any other shape comes out with colour blotches
    /// however many steps it takes. The newer models saw many aspect ratios and are fine.
    var trainedSquareOnly: Bool = false
    /// Extra VRAM (GB) per pixel² from self-attention built in full, which grows
    /// quadratically with resolution (issue #25: SD1.5 OOMs far below the linear
    /// budget). Only for models the AMD attention kernels do not cover: they take
    /// head sizes of 64, 128, 256 and 512, so SD 1.5 (40, 80, 160) still pays it
    /// while SDXL and every DiT no longer do. 0 means the attention is streamed.
    var attnVRAMSq: Double = 0

    /// Metal reports the working-set limit, not physical VRAM (a 16 GB card
    /// shows ~15 GB), so the class check tolerates a small shortfall.
    func fitsVRAMClass(_ vramGB: Double) -> Bool { vramGB >= minVRAMGB * 0.9 }

    /// Fit check aware of the encoder/VAE split: minVRAMGB assumes everything on
    /// one card, so with an aux GPU gate on the diffusion-resident budget instead,
    /// plus the aux card holding the moved components.
    func fitsGPU(mainVRAM: Double, auxVRAM: Double?) -> Bool {
        guard let auxVRAM else { return fitsVRAMClass(mainVRAM) }
        return auxHolds(auxVRAM)
            && !ImageGenLimits.baseSizes(vramGB: mainVRAM, residentGB: residentGB,
                                         attnVRAMSq: attnVRAMSq, maxLongEdge: maxLongEdge).isEmpty
    }

    /// The components moved to an aux GPU have to fit on it.
    func auxHolds(_ auxVRAM: Double?) -> Bool {
        guard let auxVRAM else { return true }
        return auxVRAM >= totalGB - residentGB
    }

    /// A transformer (DiT) ships its denoiser as a diffusion model file; the UNet models
    /// come as one checkpoint.
    var isTransformer: Bool { components.contains { $0.kind == .diffusion } }

    /// Half precision partial sums in the matrix kernel: faster, but some models (Z-Image)
    /// overflow them and come out blank, so only models checked for it opt in.
    var halfPartials: Bool = false

    /// Reference images the model takes (0 = none). They go to the text encoder and
    /// to the VAE, so the edit follows what is in them instead of a noisy start.
    var maxReferenceImages: Int = 0
    /// Side (px) references are scaled to before the VAE sees them; each one keeps its
    /// aspect. The model was tuned at 1024.
    var referenceResolution: Int = 1024

    /// Extra sd-cli flags this model needs (e.g. Flux 2 samples with euler).
    var extraArgs: [String] = []
    /// VAE tile (px) that replaces the `--vae-tile-size` in `extraArgs` on cards with 12 GB
    /// or more: fewer, larger tiles decode faster but need a bigger VAE buffer.
    var largeVAETile: Int? = nil
    /// Noise levels a step-distilled model was trained on, ending at 0. They replace the
    /// scheduler and fix the step count.
    var sigmas: [Double] = []

    /// Steps a run takes: a fixed schedule wins over the requested count.
    func steps(_ requested: Int) -> Int { sigmas.isEmpty ? requested : sigmas.count - 1 }

    var id: String { name }
    var totalGB: Double { components.reduce(0) { $0 + $1.sizeGB } }

    /// The model that stays resident in VRAM during sampling (checkpoint or
    /// diffusion transformer); VAE and text encoders are transient. Drives the
    /// per-model VRAM budget, so a heavier model correctly allows smaller images.
    var residentGB: Double {
        components.filter { $0.kind == .checkpoint || $0.kind == .diffusion }
            .map(\.sizeGB).max() ?? totalGB
    }

    func detail(_ spanish: Bool) -> String { spanish ? detailES : detailEN }
}

enum ImageGenCatalog {
    /// Non-gated FLUX autoencoder mirror, shared by Z-Image and Flux. The f16 copy: a GPU
    /// without bfloat cannot run any op on bf16 weights and generation stops on the first.
    private static func fluxVAE(_ name: String) -> ImageGenComponent {
        ImageGenComponent(kind: .vae,
            urlString: "https://huggingface.co/wbruna/Z-Image-Turbo-sdcpp-GGUF/resolve/main/ae-f16.gguf",
            fileName: name, sizeGB: 0.16)
    }

    /// Stable Diffusion 1.5 (0.9B, single checkpoint). Tiny and fast for small GPUs.
    static let sd15 = ImageGenModel(
        name: "SD 1.5",
        detailES: "Diminuto y veloz. Para GPUs pequeñas; enorme ecosistema de estilos.",
        detailEN: "Tiny and fast. For small GPUs; huge ecosystem of styles.",
        components: [ImageGenComponent(kind: .checkpoint,
            urlString: "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
            fileName: "sd-v1-5-Q8_0.gguf", sizeGB: 1.68)],
        // square only: measured 08-19, 512x288 and 768x432 come back with magenta blotches
        // in the background at the same steps that give a clean 512x512
        defaultSteps: 20, cfgScale: 7.0, minVRAMGB: 3, maxLongEdge: 768,
        nativeLongEdge: 512, trainedSquareOnly: true, attnVRAMSq: 3.4e-12, halfPartials: true)

    /// SDXL Turbo (3.5B, single checkpoint). Few steps, opens the LoRA/style world.
    static let sdxlTurbo = ImageGenModel(
        name: "SDXL Turbo",
        detailES: "3.5B, pocos pasos. Buena calidad y compatible con LoRAs/estilos SDXL.",
        detailEN: "3.5B, few steps. Strong quality and compatible with SDXL LoRAs/styles.",
        components: [ImageGenComponent(kind: .checkpoint,
            urlString: "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors",
            fileName: "sd_xl_turbo_1.0_fp16.safetensors", sizeGB: 6.6)],
        // measured 08-19: 5060 MB resident (its encoder goes to RAM) + 492 MB of graph.
        // attnVRAMSq only applies on cards without the attention kernels (safe mode).
        defaultSteps: 6, cfgScale: 1.0, minVRAMGB: 6, maxLongEdge: 1280,
        nativeLongEdge: 512, attnVRAMSq: 5e-13, halfPartials: true)

    /// Z-Image Turbo (6B DiT, 8 steps, Apache). Diffusion + VAE + Qwen3-4B encoder.
    static let zImageTurbo = ImageGenModel(
        name: "Z-Image Turbo",
        detailES: "6B, 8 pasos, Apache. Rápido y fotorrealista; el mejor equilibrio en tarjetas AMD.",
        detailEN: "6B, 8 steps, Apache. Fast and photorealistic; the best balance on AMD cards.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_0.gguf",
                fileName: "z_image_turbo-Q4_0.gguf", sizeGB: 3.5),
            fluxVAE("z_image_ae-f16.gguf"),
            ImageGenComponent(kind: .textEncoder,
                urlString: "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                fileName: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", sizeGB: 2.4),
        ],
        // measured 08-19: 3673 MB resident + 690 MB of graph at 1024x1024, and holds up at
        // 2048x2048 and 2560x1440 with no repeated composition
        defaultSteps: 8, cfgScale: 1.0, minVRAMGB: 6, nativeLongEdge: 2048)

    /// Flux.1 schnell (12B, 4 steps, Apache). Top prompt adherence for large GPUs.
    static let fluxSchnell = ImageGenModel(
        name: "Flux.1 schnell",
        detailES: "12B, 4 pasos, Apache. Máxima adherencia al prompt.",
        detailEN: "12B, 4 steps, Apache. Top prompt adherence.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_0.gguf",
                fileName: "flux1-schnell-Q4_0.gguf", sizeGB: 6.4),
            fluxVAE("flux_ae-f16.gguf"),
            ImageGenComponent(kind: .t5,
                urlString: "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q4_K_M.gguf",
                fileName: "t5-v1_1-xxl-encoder-Q4_K_M.gguf", sizeGB: 2.76),
            ImageGenComponent(kind: .clipL,
                urlString: "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                fileName: "clip_l.safetensors", sizeGB: 0.23),
        ],
        // lower minimum than before, but Z-Image stays the curated pick for these cards
        defaultSteps: 4, cfgScale: 1.0, minVRAMGB: 8, recommendable: false)

    /// Non-gated FLUX.2 autoencoder (full encoder + small decoder), shared by the
    /// Flux 2 family; the reference `ae.safetensors` lives in a gated BFL repo.
    private static let flux2VAE = ImageGenComponent(kind: .vae,
        urlString: "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors",
        fileName: "flux2_full_encoder_small_decoder.safetensors", sizeGB: 0.25)

    /// Flux.2 klein 4B: the lightest Flux 2 (4 steps). `recommendable:false` keeps Z-Image the curated AMD pick.
    static let flux2Klein4B = ImageGenModel(
        name: "Flux.2 klein 4B",
        detailES: "4B, 4 pasos, Apache. El Flux 2 más ligero: rápido y deja más VRAM para imágenes grandes.",
        detailEN: "4B, 4 steps, Apache. The lightest Flux 2: fast, and leaves more VRAM for larger images.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_0.gguf",
                fileName: "flux-2-klein-4b-Q4_0.gguf", sizeGB: 2.46),
            flux2VAE,
            ImageGenComponent(kind: .textEncoder,
                urlString: "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
                fileName: "Qwen3-4B-Q4_K_M.gguf", sizeGB: 2.5),
        ],
        defaultSteps: 4, cfgScale: 1.0, minVRAMGB: 6, recommendable: false, halfPartials: true,
        extraArgs: ["--sampling-method", "euler"])

    /// Flux.2 klein 9B (step-distilled, Apache). Flux 2 quality for 16 GB GPUs.
    static let flux2Klein9B = ImageGenModel(
        name: "Flux.2 klein 9B",
        detailES: "9B, 4 pasos, Apache. La calidad de Flux 2 sin la descarga de 30 GB.",
        detailEN: "9B, 4 steps, Apache. Flux 2 quality without the 30 GB download.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q4_0.gguf",
                fileName: "flux-2-klein-9b-Q4_0.gguf", sizeGB: 5.6),
            flux2VAE,
            ImageGenComponent(kind: .textEncoder,
                urlString: "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
                fileName: "Qwen3-8B-Q4_K_M.gguf", sizeGB: 5.0),
        ],
        defaultSteps: 4, cfgScale: 1.0, minVRAMGB: 10, recommendable: false,
        extraArgs: ["--sampling-method", "euler"])

    /// Flux.2 dev (32B). The quality reference; enormous, non-commercial license.
    /// The 24B text encoder forces --offload-to-cpu so only the sampling model
    /// holds VRAM at a time.
    static let flux2Dev = ImageGenModel(
        name: "Flux.2 dev",
        detailES: "32B, la referencia de calidad. Muy pesado (34 GB de descarga). Licencia no comercial.",
        detailEN: "32B, the quality reference. Very heavy (34 GB download). Non-commercial license.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/city96/FLUX.2-dev-gguf/resolve/main/flux2-dev-Q4_K_S.gguf",
                fileName: "flux2-dev-Q4_K_S.gguf", sizeGB: 19.3),
            flux2VAE,
            ImageGenComponent(kind: .textEncoder,
                urlString: "https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/resolve/main/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
                fileName: "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf", sizeGB: 14.3),
        ],
        defaultSteps: 20, cfgScale: 1.0, minVRAMGB: 24, recommendable: false,
        extraArgs: ["--sampling-method", "euler", "--offload-to-cpu"])

    /// Qwen-Image (20B MMDiT). The one that renders legible text inside images.
    /// Heavy: for 24 GB+ GPUs. Text encoder is Qwen2.5-VL.
    static let qwenImage = ImageGenModel(
        name: "Qwen-Image",
        detailES: "20B. Escribe texto legible dentro de la imagen. Muy pesado y lento.",
        detailEN: "20B. Renders legible text inside images. Heavy and slow.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/city96/Qwen-Image-gguf/resolve/main/qwen-image-Q4_0.gguf",
                fileName: "qwen-image-Q4_0.gguf", sizeGB: 11.3),
            ImageGenComponent(kind: .vae,
                urlString: "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
                fileName: "qwen_image_vae.safetensors", sizeGB: 0.24),
            ImageGenComponent(kind: .textEncoder,
                urlString: "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                fileName: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", sizeGB: 4.5),
        ],
        // Its 3D VAE needs IM2COL_3D, which Metal doesn't implement (issue #19):
        // run the VAE on CPU until we ship a kernel.
        defaultSteps: 20, cfgScale: 2.5, minVRAMGB: 16,
        extraArgs: ["--backend", "vae=cpu"])

    /// Turbo's distilled schedule (`sample_sigmas` of its model_index.json) plus the final 0.
    static let qwenImage21TurboSigmas = [1.0, 0.978453, 0.95418, 0.926626, 0.89508,
                                         0.845148, 0.704534, 0.414568, 0.0]

    /// Qwen-Image 2.1 (7B MMDiT, non-commercial license). Writes legible text and edits from
    /// reference images; Turbo shares its VAE and the Qwen3-VL-8B encoder, which runs off the card.
    static func qwenImage21(_ name: String, detailES: String, detailEN: String,
                            repo: String = "leejet/Qwen-Image-2.1-GGUF", revision: String = "main",
                            file: String, formerFile: String? = nil, sizeGB: Double, minVRAMGB: Double,
                            recommendable: Bool, turbo: Bool = false) -> ImageGenModel {
        ImageGenModel(
            name: name, detailES: detailES, detailEN: detailEN,
            components: [
                ImageGenComponent(kind: .diffusion,
                    urlString: "https://huggingface.co/\(repo)/resolve/\(revision)/\(file)",
                    fileName: file, sizeGB: sizeGB, formerFileNames: formerFile.map { [$0] } ?? []),
                ImageGenComponent(kind: .vae,
                    urlString: "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
                    fileName: "qwen_image_2.1_vae_bf16.safetensors", sizeGB: 0.68),
                ImageGenComponent(kind: .textEncoder,
                    urlString: "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                    fileName: "Qwen3VL-8B-Instruct-Q4_K_M.gguf", sizeGB: 5.03),
                ImageGenComponent(kind: .llmVision,
                    urlString: "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
                    fileName: "mmproj-Qwen3VL-8B-Instruct-F16.gguf", sizeGB: 1.16),
            ],
            // The published recipes: 25 steps (Turbo: its own 8) at guidance 1 with Euler; a card
            // that draws the desktop stops at 1920.
            defaultSteps: turbo ? qwenImage21TurboSigmas.count - 1 : 25, cfgScale: 1.0,
            minVRAMGB: minVRAMGB, recommendable: recommendable,
            maxLongEdge: 2048, displayMaxLongEdge: 1920, nativeLongEdge: 2048,
            halfPartials: true,
            maxReferenceImages: 16,
            // 384 px VAE tiles: 256 pays per tile and 512 grows the mid-block attention faster than it saves.
            extraArgs: ["--sampling-method", "euler"] + (turbo ? [] : ["--scheduler", "simple"])
                + ["--vae-tile-size", "384"],
            largeVAETile: 640,
            sigmas: turbo ? qwenImage21TurboSigmas : [])
    }

    static let qwenImage21Q3 = qwenImage21(
        "Qwen-Image 2.1 (Q3)",
        detailES: "7B. Escribe texto legible y edita desde imágenes de referencia. Versión ligera para 8 GB. Licencia no comercial.",
        detailEN: "7B. Writes legible text and edits from reference images. Light build for 8 GB cards. Non-commercial license.",
        file: "qwen_image_2.1-Q3_K.gguf", sizeGB: 3.27, minVRAMGB: 8, recommendable: false)

    static let qwenImage21Q4 = qwenImage21(
        "Qwen-Image 2.1",
        detailES: "7B. Escribe texto legible dentro de la imagen y edita desde imágenes de referencia. Licencia no comercial.",
        detailEN: "7B. Writes legible text inside the image and edits from reference images. Non-commercial license.",
        file: "qwen_image_2.1-Q4_K.gguf", sizeGB: 4.20, minVRAMGB: 12, recommendable: false)

    static let qwenImage21Q6 = qwenImage21(
        "Qwen-Image 2.1 (Q6)",
        detailES: "7B con más precisión. Mismo ritmo que la Q4. Licencia no comercial.",
        detailEN: "7B at higher precision, at the same pace as the Q4 build. Non-commercial license.",
        file: "qwen_image_2.1-Q6_K.gguf", sizeGB: 6.00, minVRAMGB: 12, recommendable: false)

    static let qwenImage21Q8 = qwenImage21(
        "Qwen-Image 2.1 (Q8)",
        detailES: "7B casi sin pérdida, para tarjetas de 16 GB en adelante. Licencia no comercial.",
        detailEN: "7B at near-lossless precision, for 16 GB cards and up. Non-commercial license.",
        file: "qwen_image_2.1-Q8_0.gguf", sizeGB: 7.69, minVRAMGB: 16, recommendable: false)

    /// Pinned to the commit that was measured, so the URL keeps serving those files. The AtomicChat
    /// builds offered before still run when they are already on disk.
    private static let qwenImage21TurboRepo = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
    private static let qwenImage21TurboRevision = "5d1f9f759ea877ae556cda88dbec958b62ad3809"

    static let qwenImage21TurboQ3 = qwenImage21(
        "Qwen-Image 2.1 Turbo (Q3)",
        detailES: "7B en 8 pasos en vez de 25. Texto legible y edición desde referencias. Versión ligera para 8 GB. Licencia no comercial.",
        detailEN: "7B in 8 steps instead of 25. Legible text and edits from references. Light build for 8 GB cards. Non-commercial license.",
        repo: qwenImage21TurboRepo, revision: qwenImage21TurboRevision,
        file: "qwen-image-2.1-turbo-Q3_K_XL.gguf", formerFile: "Qwen-Image-2.1-Turbo-AD-Q3_K.gguf",
        sizeGB: 3.61, minVRAMGB: 8, recommendable: false, turbo: true)

    static let qwenImage21TurboQ4 = qwenImage21(
        "Qwen-Image 2.1 Turbo",
        detailES: "7B en 8 pasos en vez de 25. Escribe texto legible y edita desde imágenes de referencia. Licencia no comercial.",
        detailEN: "7B in 8 steps instead of 25. Writes legible text and edits from reference images. Non-commercial license.",
        repo: qwenImage21TurboRepo, revision: qwenImage21TurboRevision,
        file: "qwen-image-2.1-turbo-Q4_K_M.gguf", formerFile: "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf",
        sizeGB: 4.20, minVRAMGB: 12, recommendable: false, turbo: true)

    static let qwenImage21TurboQ6 = qwenImage21(
        "Qwen-Image 2.1 Turbo (Q6)",
        detailES: "7B en 8 pasos, con más precisión. Licencia no comercial.",
        detailEN: "7B in 8 steps, at higher precision. Non-commercial license.",
        repo: qwenImage21TurboRepo, revision: qwenImage21TurboRevision,
        file: "qwen-image-2.1-turbo-Q6_K_XL.gguf", formerFile: "Qwen-Image-2.1-Turbo-AD-Q6_K.gguf",
        sizeGB: 6.72, minVRAMGB: 12, recommendable: false, turbo: true)

    static let qwenImage21TurboQ8 = qwenImage21(
        "Qwen-Image 2.1 Turbo (Q8)",
        detailES: "7B en 8 pasos, casi sin pérdida, para tarjetas de 16 GB en adelante. Licencia no comercial.",
        detailEN: "7B in 8 steps at near-lossless precision, for 16 GB cards and up. Non-commercial license.",
        repo: qwenImage21TurboRepo, revision: qwenImage21TurboRevision,
        file: "qwen-image-2.1-turbo-Q8_0.gguf", formerFile: "Qwen-Image-2.1-Turbo-Q8_0.gguf",
        sizeGB: 7.64, minVRAMGB: 16, recommendable: false, turbo: true)

    /// FLUX.2 VAE used by MOSS-Image (Comfy mirror). Local copies often keep the
    /// shorter `flux2-vae.safetensors` name from the same tree.
    private static let mossVAE = ImageGenComponent(kind: .vae,
        urlString: "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae.safetensors",
        fileName: "flux2-vae.safetensors", sizeGB: 0.34,
        formerFileNames: ["flux2_ae.safetensors"])

    /// Qwen3.5 vocabulary → `vision/tokenizers/tokenizer.json` (sd-server tokenizer slot).
    private static let mossTokenizer = ImageGenComponent(kind: .tokenizer,
        urlString: "https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/resolve/main/tokenizer/tokenizer.json",
        fileName: "tokenizer.json", sizeGB: 0.02,
        formerFileNames: ["tokenizer_qwen3.5.json"])

    /// Default Qwen3.5-9B encoder for MOSS (Q8). Instance config can swap to Q4.
    static let mossLLMQ8 = ImageGenComponent(kind: .textEncoder,
        urlString: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf",
        fileName: "Qwen3.5-9B-Q8_0.gguf", sizeGB: 9.53)

    /// Q8 fine-tune for MOSS `--llm`.
    static let mossLLMAnimus = ImageGenComponent(kind: .textEncoder,
        urlString: "https://huggingface.co/Darkhn-Quants-3/Qwen3.5-9B-Animus-V13.0-GGUF/resolve/main/Qwen3.5-9B-Animus-V13.0-Q8_0.gguf",
        fileName: "Qwen3.5-9B-Animus-V13.0-Q8_0.gguf", sizeGB: 9.53)

    /// Q8 uncensored fine-tune for MOSS `--llm`.
    static let mossLLMHeretic = ImageGenComponent(kind: .textEncoder,
        urlString: "https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF/resolve/main/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q8_0.gguf",
        fileName: "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q8_0.gguf", sizeGB: 9.8)

    /// MOSS-Image (5B DiT + FLUX.2 VAE + Qwen3.5-9B). Needs an external tokenizer.
    static func mossImage(_ name: String, detailES: String, detailEN: String,
                          repo: String, file: String, sizeGB: Double, minVRAMGB: Double,
                          urlPath: String? = nil) -> ImageGenModel {
        let path = urlPath ?? file
        return ImageGenModel(
            name: name, detailES: detailES, detailEN: detailEN,
            components: [
                ImageGenComponent(kind: .diffusion,
                    urlString: "https://huggingface.co/\(repo)/resolve/main/\(path)",
                    fileName: file, sizeGB: sizeGB),
                mossVAE, mossLLMQ8, mossTokenizer,
            ],
            // Keep the DiT resident. --offload-to-cpu streams it from RAM each step and
            // leaves the GPU half idle; the Qwen encoder stays on CPU via te=cpu because
            // Q8 DiT + Q8 9B together do not fit a 16 GB card.
            defaultSteps: 20, cfgScale: 3.0, minVRAMGB: minVRAMGB, recommendable: false,
            maxLongEdge: 2048, displayMaxLongEdge: 1920, nativeLongEdge: 1024)
    }

    static let mossImageBaseQ8 = mossImage(
        "MOSS-Image (Q8)",
        detailES: "5B DiT Q8 + Qwen3.5-9B. Buen texto en imagen; necesita tokenizer externo.",
        detailEN: "5B DiT Q8 + Qwen3.5-9B. Strong in-image text; needs an external tokenizer.",
        repo: "leejet/MOSS-Image-Base-GGUF", file: "MOSS-Image-Base-Q8_0.gguf",
        sizeGB: 5.48, minVRAMGB: 10)

    static let mossImageSFTQ8 = mossImage(
        "MOSS-Image SFT",
        detailES: "5B SFT. Misma pila que Base, afinada para seguir el prompt.",
        detailEN: "5B SFT. Same stack as Base, tuned for prompt following.",
        repo: "leejet/MOSS-Image-SFT-GGUF", file: "MOSS-Image-SFT-Q8_0.gguf",
        sizeGB: 5.48, minVRAMGB: 10)

    /// Curated order (small to large). Z-Image sits before SDXL so it wins the
    /// 8-12 GB tie as the recommended pick (validated for photorealism on AMD);
    /// klein 9B sits before schnell to win the 16 GB tie the same way.
    static let models: [ImageGenModel] = [sd15, zImageTurbo, sdxlTurbo, qwenImage21Q3, qwenImage21TurboQ3,
                                          flux2Klein4B, qwenImage21Q4, qwenImage21TurboQ4, flux2Klein9B,
                                          fluxSchnell, qwenImage21Q6, qwenImage21TurboQ6, qwenImage,
                                          mossImageBaseQ8, mossImageSFTQ8,
                                          qwenImage21Q8, qwenImage21TurboQ8, flux2Dev]

    static func isMossBase(_ modelID: String) -> Bool {
        modelID == mossImageBaseQ8.id || modelID == "MOSS-Image"
            || modelID == "MOSS-Image (F16)"
    }

    /// Any MOSS catalog entry (Base or SFT) — shows the text-encoder picker.
    static func isMossFamily(_ modelID: String) -> Bool {
        isMossBase(modelID) || modelID == mossImageSFTQ8.id
    }

    /// The best model this GPU can run: the highest min-VRAM tier that fits, and
    /// within a tie the earliest listed (curated preference).
    static func recommended(for hw: HardwareInfo) -> ImageGenModel? {
        let fitting = models.filter { $0.recommendable && $0.fitsVRAMClass(hw.vramGB) }
        guard let top = fitting.map(\.minVRAMGB).max() else { return nil }
        return fitting.first { $0.minVRAMGB == top }
    }

    static func model(id: String) -> ImageGenModel? {
        if let m = models.first(where: { $0.id == id }) { return m }
        // Keep old saved Base IDs resolving to Q8 after the precision choice was removed.
        if id == "MOSS-Image" || id == "MOSS-Image (F16)" { return mossImageBaseQ8 }
        return nil
    }

    /// Sentinel id for the user's own model (files picked from disk).
    static let customID = "__custom__"

    /// Build a model from user-picked files: a checkpoint (or diffusion model) and
    /// an optional VAE. Passed as a full checkpoint via --model, which covers most
    /// community SD/SDXL finetunes; minVRAMGB 0 so it's never blocked.
    static func custom(modelPath: String, vaePath: String, textEncoderPath: String = "",
                       isDiffusion: Bool = false, steps: Int, cfg: Double) -> ImageGenModel {
        func gb(_ p: String) -> Double {
            let bytes = (try? FileManager.default.attributesOfItem(atPath: p))?[.size] as? Int ?? 0
            return Double(bytes) / 1_073_741_824
        }
        var comps = [ImageGenComponent(kind: isDiffusion ? .diffusion : .checkpoint, urlString: "",
                                       fileName: modelPath, sizeGB: gb(modelPath))]
        if !vaePath.isEmpty {
            comps.append(ImageGenComponent(kind: .vae, urlString: "", fileName: vaePath, sizeGB: gb(vaePath)))
        }
        if !textEncoderPath.isEmpty {
            comps.append(ImageGenComponent(kind: .textEncoder, urlString: "",
                                           fileName: textEncoderPath, sizeGB: gb(textEncoderPath)))
        }
        // Assume the SD/SDXL UNet family (the common custom case) for the VRAM cap.
        return ImageGenModel(name: "Custom",
            detailES: "Tu propio modelo. Ajusta pasos y CFG según su ficha.",
            detailEN: "Your own model. Set steps and CFG to match its card.",
            components: comps, defaultSteps: steps, cfgScale: cfg, minVRAMGB: 0,
            attnVRAMSq: 3.4e-12)
    }
}

/// Resolution and command-buffer limits derived from the detected GPU. Large
/// diffusion runs hit two ceilings on AMD: VRAM (the compute buffer grows with
/// pixels) and the macOS GPU watchdog (a single command buffer that runs too long
/// is killed). These pick sizes that clear both.
enum ImageGenLimits {
    /// Estimated peak VRAM (GB): resident weights, half a gigabyte of runtime, and
    /// the larger of the two graphs, which never overlap (the model samples, then
    /// the VAE decodes). Recalibrated 08-19 against the sizes the engine reports,
    /// with the attention going through the AMD kernels: the diffusion graph is a
    /// straight line in pixels (690 MB at 1024x1024, 966 at 1600x900 and 2757 at
    /// 2048x2048 on Z-Image; SDXL is lower, so this is the conservative side) and
    /// the tiled VAE decode is flat at about 0.5 GB whatever the frame size. The
    /// quadratic term only survives for models those kernels do not cover.
    static func estVRAMGB(px: Int, residentGB: Double, attnVRAMSq: Double,
                          streamedAttention: Bool = true) -> Double {
        let p = Double(px)
        guard streamedAttention else {
            // A card the attention kernels do not cover builds the score matrix whole, and
            // then memory grows with the square of the frame again. This is the estimate as
            // it was calibrated for that case, kept for exactly those cards.
            return residentGB + 1.0 + 4.5e-6 * p + attnVRAMSq * p * p
        }
        let graphMB = max(6.6e-4 * p, 500)
        return residentGB + 0.5 + graphMB / 1024
    }

    /// Whether this GPU runs attention through the card's own kernels, which is what keeps
    /// the graph linear. Apple Silicon uses the upstream path and AMD ours; the 64-wide
    /// cards lose it in safe mode, which is the case the estimate must not get wrong.
    static func streamsAttention(gpuName: String, extraArgs: String) -> Bool {
        guard extraArgs.contains("GGML_METAL_WAVE64_SAFEMODE") else { return true }
        return GPUArchitectureClassifier.architecture(for: gpuName) != "GCN / Vega"
    }

    /// Base (long-edge) sizes to offer: a 16:9 frame at that base must fit VRAM,
    /// capped by the model's quality ceiling. Scales with the card and the model.
    /// Does this card stream attention? Read from the selected GPU and the engine's extra
    /// arguments, so a card in safe mode gets the estimate that matches what it really does.
    static func streamsAttention(gpuIndex: Int) -> Bool {
        let d = UserDefaults.standard
        let extra = d.string(forKey: SettingsKeys.extraArgs) ?? ""
        let devices = MTLCopyAllDevices()
        let name = (gpuIndex >= 0 && gpuIndex < devices.count)
            ? devices[gpuIndex].name
            : (MTLCreateSystemDefaultDevice()?.name ?? "")
        return streamsAttention(gpuName: name, extraArgs: extra)
    }

    /// Working set (GB) Metal allows on the selected card.
    static func workingSetGB(gpuIndex: Int) -> Double {
        let devices = MTLCopyAllDevices()
        let dev = (gpuIndex >= 0 && gpuIndex < devices.count) ? devices[gpuIndex] : MTLCreateSystemDefaultDevice()
        return Double(dev?.recommendedMaxWorkingSetSize ?? 0) / 1_073_741_824
    }

    /// Whether the selected card draws the desktop (not headless).
    static func drivesDisplay(gpuIndex: Int) -> Bool {
        let devices = MTLCopyAllDevices()
        guard gpuIndex >= 0 && gpuIndex < devices.count else {
            return !(MTLCreateSystemDefaultDevice()?.isHeadless ?? true)
        }
        return !devices[gpuIndex].isHeadless
    }

    static func baseSizes(vramGB: Double, residentGB: Double,
                          attnVRAMSq: Double = 0, maxLongEdge: Int = .max,
                          streamedAttention: Bool = true) -> [Int] {
        let candidates = [512, 640, 768, 896, 1024, 1152, 1280, 1440,
                          1600, 1792, 1920, 2048, 2304, 2560, 3072]
        return candidates.filter { base in
            guard base <= maxLongEdge else { return false }
            let short = max(256, Int((Double(base) * 9 / 16 / 64).rounded()) * 64)
            return fits(width: base, height: short, vramGB: vramGB,
                        residentGB: residentGB, attnVRAMSq: attnVRAMSq,
                        streamedAttention: streamedAttention)
        }
    }

    /// True when a specific frame's estimated VRAM fits (a square at a large base,
    /// or any UNet frame at high res, may not even when a 16:9 base does).
    static func fits(width: Int, height: Int, vramGB: Double,
                     residentGB: Double, attnVRAMSq: Double = 0,
                     streamedAttention: Bool = true) -> Bool {
        estVRAMGB(px: width * height, residentGB: residentGB, attnVRAMSq: attnVRAMSq,
                  streamedAttention: streamedAttention) <= vramGB
    }

    /// Fraction of VRAM the frame is estimated to use (drives the near-limit note).
    static func vramFraction(width: Int, height: Int, vramGB: Double,
                             residentGB: Double, attnVRAMSq: Double = 0,
                             streamedAttention: Bool = true) -> Double {
        estVRAMGB(px: width * height, residentGB: residentGB, attnVRAMSq: attnVRAMSq,
                  streamedAttention: streamedAttention) / max(0.1, vramGB)
    }

    /// Pixels each reference image is scaled to: full size for one or two, then an even
    /// share of two full-size references, so sixteen of them stay a bounded sequence.
    static func referencePixels(count: Int, resolution: Int) -> Int? {
        guard count > 0, resolution > 0 else { return nil }
        let full = resolution * resolution
        return count <= 2 ? full : max(256 * 256, 2 * full / count)
    }

    /// Command buffers per diffusion step, short enough that the driver's hang check never trips;
    /// 8 is the engine's maximum.
    static func nCB(width: Int, height: Int) -> Int {
        let px = width * height
        return min(8, max(1, Int((Double(px) / 450_000).rounded(.up))))
    }
}

/// Output framing. The base size is the LONG edge; the short edge scales down
/// from it, so no preset ever exceeds base x base pixels. That keeps a single
/// diffusion step under the AMD GPU watchdog (a larger step times out). Rounded
/// to multiples of 64 (the latent grid).
enum ImageAspect: String, CaseIterable, Identifiable {
    case square = "1:1"
    case landscape = "16:9"
    case portrait = "9:16"
    case photo = "4:3"
    case photoTall = "3:4"
    case cinema = "2.39:1"
    case custom = "custom"
    var id: String { rawValue }

    /// Snap to the latent grid (multiples of 64 px), min 256.
    static func gridSnap(_ v: Double) -> Int { max(256, Int((v / 64).rounded()) * 64) }

    /// (width, height) with `base` as the longer edge, rounded to the latent grid.
    func dimensions(base: Int) -> (Int, Int) {
        let short9x16 = Self.gridSnap(Double(base) * 9 / 16)
        let short3x4  = Self.gridSnap(Double(base) * 3 / 4)
        switch self {
        case .square:    return (base, base)
        case .landscape: return (base, short9x16)
        case .portrait:  return (short9x16, base)
        case .photo:     return (base, short3x4)
        case .photoTall: return (short3x4, base)
        case .cinema:    return (base, Self.gridSnap(Double(base) / 2.39))
        case .custom:    return (base, base)   // ratio comes from the config, see customDimensions
        }
    }

    /// Dimensions for a free "W:H" ratio with `base` as the long edge. `base` still
    /// caps the pixel count (VRAM stays safe); falls back to square if unparseable.
    static func customDimensions(ratio: String, base: Int) -> (Int, Int) {
        let parts = ratio.split(whereSeparator: { ":x/ ".contains($0) }).compactMap { Double($0) }
        guard parts.count == 2, parts[0] > 0, parts[1] > 0 else { return (base, base) }
        let (rw, rh) = (parts[0], parts[1])
        return rw >= rh ? (base, gridSnap(Double(base) * rh / rw))
                        : (gridSnap(Double(base) * rw / rh), base)
    }
}

/// Output file format. JPG is much lighter for photographic results; PNG stays
/// lossless for line art or when the user wants to re-edit.
enum ImageFormat: String, CaseIterable, Identifiable {
    case png, jpg
    var id: String { rawValue }
    var ext: String { rawValue }
}

/// Latent preview method passed to sd-cli `--preview` (none / proj / vae / tae).
enum ImagePreviewMode: String, CaseIterable, Identifiable {
    case none, proj, vae, tae
    var id: String { rawValue }

    func label(_ spanish: Bool) -> String {
        switch self {
        case .none: return spanish ? "Ninguna" : "None"
        case .proj: return "proj"
        case .vae:  return "vae"
        case .tae:  return "tae"
        }
    }
}

/// Qwen3.5-9B (and fine-tunes) for MOSS `--llm` (`vision/text_encoders/`).
enum ImageMossTextEncoder: String, CaseIterable, Identifiable {
    case q8 = "q8"
    case animus = "animus"
    case heretic = "heretic"
    var id: String { rawValue }

    var component: ImageGenComponent {
        switch self {
        case .q8: return ImageGenCatalog.mossLLMQ8
        case .animus: return ImageGenCatalog.mossLLMAnimus
        case .heretic: return ImageGenCatalog.mossLLMHeretic
        }
    }

    var shortLabel: String {
        switch self {
        case .q8: return "Q8"
        case .animus: return "Animus"
        case .heretic: return "Heretic"
        }
    }
}

/// Where the text encoder lives during a run (sd.cpp `--params-backend` / Bemly unload).
enum ImageTEResidency: String, CaseIterable, Identifiable {
    case teCpu = "teCpu"
    case fullGpu = "fullGpu"
    case encodeThenUnload = "encodeThenUnload"
    var id: String { rawValue }

    var cliArgs: [String] {
        switch self {
        case .teCpu: return ["--params-backend", "te=cpu"]
        case .fullGpu: return []
        case .encodeThenUnload: return ["--release-te-after-encode"]
        }
    }
}

extension ImageGenModel {
    /// Swap the MOSS LLM file without touching DiT / VAE / tokenizer.
    func withTextEncoder(_ te: ImageGenComponent) -> ImageGenModel {
        var copy = self
        copy.components = components.map { $0.kind == .textEncoder ? te : $0 }
        return copy
    }
}

/// sd-cli `--sampling-method` values (stable-diffusion.h `sample_method_to_str`).
enum ImageSamplingMethod: String, CaseIterable, Identifiable {
    case auto = ""
    case euler, euler_a, heun, dpm2
    case dpmpp2s_a = "dpm++2s_a"
    case dpmpp2m = "dpm++2m"
    case dpmpp2mv2 = "dpm++2mv2"
    case ipndm, ipndm_v, lcm, ddim_trailing, tcd
    case res_multistep, res_2s, er_sde
    case euler_cfg_pp, euler_a_cfg_pp, euler_ge
    case dpmpp2m_sde = "dpm++2m_sde"
    case dpmpp2m_sde_bt = "dpm++2m_sde_bt"
    case lms
    var id: String { rawValue }

    func label(_ spanish: Bool) -> String {
        rawValue.isEmpty ? (spanish ? "Automático" : "Auto") : rawValue
    }
}

/// sd-cli `--scheduler` values (stable-diffusion.h `scheduler_to_str`).
enum ImageScheduler: String, CaseIterable, Identifiable {
    case auto = ""
    case discrete, karras, exponential, ays, gits, sgm_uniform
    case simple, smoothstep, kl_optimal, lcm, bong_tangent
    case ltx2, logit_normal, flux2, flux, beta, llada_image
    var id: String { rawValue }

    func label(_ spanish: Bool) -> String {
        rawValue.isEmpty ? (spanish ? "Automático" : "Auto") : rawValue
    }
}

/// Control image kinds for SDXL ControlNet Union (`--control-type`).
enum ImageControlType: String, CaseIterable, Identifiable {
    case auto = ""
    case openpose, depth, scribble, canny, normal, segment, tile, repaint
    var id: String { rawValue }

    func label(_ spanish: Bool) -> String {
        switch self {
        case .auto: return spanish ? "Automático" : "Auto"
        default: return rawValue
        }
    }
}

/// Step caching in the sampler: a cached step reuses the previous output instead of running
/// the model. Faster, at the cost of detail. Measured on Qwen-Image 2.1 at 1024x1024 and 25
/// steps against no cache: cache-dit 1.46x (PSNR 30.6 dB), spectrum 1.91x (30.6 dB),
/// easycache 2.06x (31.6 dB).
enum ImageFastMode: String, CaseIterable, Identifiable {
    case off, cacheDit = "cache-dit", spectrum, easycache
    var id: String { rawValue }

    /// cache-dit and easycache work on transformer models only; spectrum also on UNets.
    func supports(_ model: ImageGenModel) -> Bool {
        // a distilled schedule has no redundant step to reuse
        if !model.sigmas.isEmpty { return self == .off }
        switch self {
        case .off, .spectrum: return true
        case .cacheDit, .easycache: return model.isTransformer
        }
    }

    /// sd-cli arguments for this mode on this model; empty when off or not supported.
    func args(for model: ImageGenModel) -> [String] {
        self == .off || !supports(model) ? [] : ["--cache-mode", rawValue]
    }
}

/// Drives one text-to-image run: resolves the bundled engine and the installed
/// model, launches sd-cli with the AMD-validated flags, and surfaces progress.
@MainActor
final class ImageGenerator: ObservableObject {
    enum State: Equatable { case idle, generating, done, failed(String) }

    /// Coarse stage so the UI can explain what the long wait is doing.
    enum Stage { case loading, sampling, decoding }

    @Published var state: State = .idle
    @Published var stage: Stage = .loading
    /// 0...1 across the sampling steps, parsed from the engine's progress line.
    @Published var progress: Double = 0
    @Published var stepText: String = ""
    @Published var resultImage: NSImage?
    @Published var resultURL: URL?
    /// Latent preview refreshed by the engine every few steps, so the wait shows
    /// the image forming instead of a bare bar.
    @Published var previewImage: NSImage?
    /// Wall-clock seconds of the last completed run, for the "generated in" caption.
    @Published var lastDuration: Int = 0

    /// Called after a run settles (done, failed or cancelled). The pool uses it to
    /// dispatch the next queued prompt.
    var onFinish: (() -> Void)?
    /// The last run's inputs, so the session gallery can label the result.
    private(set) var lastPrompt = ""
    private(set) var lastNegativePrompt = ""
    private(set) var lastSeed = -1
    private(set) var lastWidth = 0
    private(set) var lastHeight = 0
    private(set) var lastSteps = 0
    private(set) var lastCfgScale = 0.0
    private(set) var lastSamplingMethod = ""
    private(set) var lastScheduler = ""
    private(set) var lastFlowShift = -1.0
    private(set) var lastStrength = 0.6
    private(set) var lastModelID = ""
    private(set) var lastFormat = "png"
    private(set) var lastInitImagePath = ""
    /// Set only for a draft that wrote a latent. Empty on a normal run and on Continue.
    private(set) var lastLatentURL: URL?
    private(set) var lastFullSteps: Int?
    private(set) var lastEndedAtStep: Int?
    /// Multi-seed draft batch from one TE encode; empty for a single-image run.
    private(set) var lastBatch: [FinishedImageItem] = []
    private var pendingLatentURL: URL?
    private var pendingBatchCount = 1
    var pendingOutputPattern: String?
    private var pendingLatentPattern: String?
    private var pendingBaseSeed = -1
    private var pendingFullSteps: Int?
    private var pendingEndedAtStep: Int?

    struct FinishedImageItem {
        let url: URL
        let image: NSImage
        let seed: Int
        let latentURL: URL?
        let fullSteps: Int?
        let endedAtStep: Int?
    }

    private var process: Process?
    var runTask: Task<Void, Never>?
    var activeJobID: String?
    private var startedAt: Date?
    var firstStepAt: Date?
    private var previewURL: URL?
    private var previewTimer: Timer?
    private var previewSeenAt: Date?
    /// Tail of the engine output, kept so a failure can be diagnosed (e.g. a GPU
    /// watchdog timeout) into a helpful message.
    var logTail = ""
    /// Persist the engine output so the Logs tab can show image runs too.
    let fileLog = RotatingFileLog(name: "imagegen")

    /// Newest image-gen log file, for the Logs tab to read (this generator lives in
    /// another window, so the file is the shared handoff).
    static var latestLogURL: URL? {
        let dir = AppSupport.directory.appendingPathComponent("logs", isDirectory: true)
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        return files.filter { $0.lastPathComponent.hasPrefix("imagegen") }
            .max { a, b in (mtime(a) ?? .distantPast) < (mtime(b) ?? .distantPast) }
    }
    private static func mtime(_ u: URL) -> Date? {
        try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
    }

    var isBusy: Bool { if case .generating = state { return true }; return false }

    /// Seconds elapsed since the run started (for the status line).
    var elapsed: Int { startedAt.map { Int(-$0.timeIntervalSinceNow) } ?? 0 }

    /// Estimated seconds left, extrapolated from the pace of completed steps plus
    /// a flat allowance for the VAE decode that follows. Nil until a step lands.
    var etaSeconds: Int? {
        guard let first = firstStepAt, progress > 0.001, progress < 1 else { return nil }
        let spent = -first.timeIntervalSinceNow
        let remainingSampling = spent / progress * (1 - progress)
        return max(0, Int(remainingSampling + spent * 0.2))   // ~20% tail for the decode
    }

    /// The bundled engine, with a dev fallback to the reproducible vendor build.
    static var binary: String {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("bin-image/sd-cli").path,
           FileManager.default.fileExists(atPath: bundled) {
            return bundled
        }
        return NSString(string: "~/dev/repositorios/toshllm/vendor/stable-diffusion.cpp/build-static/bin/sd-cli")
            .expandingTildeInPath
    }

    static var engineInstalled: Bool { FileManager.default.fileExists(atPath: binary) }

    /// True when every component file for `model` is present (and named).
    static func installed(_ model: ImageGenModel, in models: ModelStore) -> Bool {
        !model.components.isEmpty && model.components.allSatisfy {
            !$0.fileName.isEmpty && models.hasComponent($0)
        }
    }

    func cancel() {
        process?.terminate()
        process = nil
        runTask?.cancel()
        if let id = activeJobID {
            Task { await ImageWebUIServer.shared.cancelJob(id) }
        }
        if isBusy { state = .idle }
    }

    /// sd-cli --backend spec for the encoder/VAE split: diffusion on Metal slot 0,
    /// text encoder and VAE on slot 1. Entries from the model's own spec win per
    /// module (keys normalized, since sd-cli accepts synonyms like clip/t5/tae).
    nonisolated static func splitBackendSpec(overriding overrides: String?) -> String {
        func canon(_ key: String) -> String {
            switch key {
            case "model", "unet", "dit": return "diffusion"
            case "clip", "text", "textencoder", "textencoders",
                 "conditioner", "cond", "llm", "t5", "t5xxl": return "te"
            case "firststage", "autoencoder", "tae": return "vae"
            default: return key
            }
        }
        var map = ["diffusion": "mtl0", "te": "mtl1", "vae": "mtl1"]
        for pair in (overrides ?? "").split(separator: ",") {
            let kv = pair.split(separator: "=", maxSplits: 1).map {
                $0.trimmingCharacters(in: .whitespaces).lowercased()
            }
            if kv.count == 2, !kv[0].isEmpty, !kv[1].isEmpty { map[canon(kv[0])] = kv[1] }
        }
        return map.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ",")
    }

    /// Generate an image. `prompt` is required; the rest are the tuned defaults
    /// unless the caller overrides them.
    func generate(model: ImageGenModel, models: ModelStore, prompt: String,
                  negativePrompt: String = "",
                  width: Int, height: Int, steps: Int, seed: Int,
                  format: ImageFormat, offloadToCPU: Bool, gpuIndex: Int,
                  auxGPUIndex: Int = -1,
                  initImagePath: String = "", maskPath: String = "",
                  strength: Double = 0.75,
                  referenceImagePaths: [String] = [],
                  fastMode: ImageFastMode = .off,
                  previewMode: ImagePreviewMode = .proj,
                  cfgScale: Double? = nil,
                  samplingMethod: ImageSamplingMethod = .auto,
                  scheduler: ImageScheduler = .auto,
                  flowShift: Double = -1,
                  slgScale: Double = 0, slgStart: Double = 0.01, slgEnd: Double = 0.2,
                  slgLayers: String = "",
                  controlNetPath: String = "", controlImagePath: String = "",
                  controlStrength: Double = 0.9,
                  controlType: ImageControlType = .auto,
                  handoff: ImageLatentHandoff? = nil,
                  batchCount: Int = 1) {
        guard !isBusy else { return }
        let steps = model.steps(steps)
        let cfg = cfgScale ?? model.cfgScale
        // A reference that went missing would shift every <imageN> after it, so the run
        // stops instead of editing against the wrong picture.
        let refs = Array(referenceImagePaths.filter { !$0.isEmpty }.prefix(max(0, model.maxReferenceImages)))
        if refs.contains(where: { !FileManager.default.fileExists(atPath: $0) }) {
            state = .failed("MISSING_REF")
            onFinish?()
            return
        }
        lastPrompt = prompt
        lastNegativePrompt = negativePrompt
        lastSeed = seed
        lastWidth = width
        lastHeight = height
        lastSteps = steps
        lastCfgScale = cfg
        lastSamplingMethod = samplingMethod.rawValue
        lastScheduler = scheduler.rawValue
        lastFlowShift = flowShift
        lastStrength = strength
        lastModelID = model.id
        lastFormat = format.rawValue
        lastInitImagePath = initImagePath
        lastBatch = []
        let dir = models.imagenDirectory
        // Timestamp plus a short token: a batch fired in the same second (one run
        // per instance) would otherwise share a name and overwrite a single file.
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HH.mm.ss"
        let token = String(UUID().uuidString.prefix(4)).lowercased()
        let batchN = max(1, min(batchCount, 16))
        // %d so one TE encode can write many seeds without clobbering files.
        let out = batchN > 1
            ? dir.appendingPathComponent("toshllm_\(fmt.string(from: Date()))_\(token)_%d.\(format.ext)")
            : dir.appendingPathComponent("toshllm_\(fmt.string(from: Date()))_\(token).\(format.ext)")

        // Each component maps to its own sd-cli flag (a full checkpoint via --model,
        // or a diffusion model plus its VAE and text encoders).
        var args: [String] = []
        for comp in model.components {
            args += [comp.flag, models.componentPath(comp).path]
        }
        // sd-cli picks LoRAs up from the prompt as <lora:file:weight>, but only when it knows
        // where to look. The folder is only passed when it holds something.
        if let lora = ImageGenPool.loraDirectory() {
            args += ["--lora-model-dir", lora.path]
        }
        args += [
            "-p", prompt,
            "--cfg-scale", String(format: "%.2f", cfg),
            "--steps", String(steps),
        ]
        if !model.sigmas.isEmpty {
            args += ["--sigmas", model.sigmas.map { String($0) }.joined(separator: ",")]
        }
        args += [
            "-W", String(width), "-H", String(height),
            "--seed", String(seed),
            // Tiled VAE decode keeps each Metal command buffer under the AMD GPU
            // watchdog; without it the decode times out and the colors corrupt.
            "--vae-tiling",
            // Attention through the card's own kernels instead of building the whole
            // table: 2379 to 690 MB at 1024x1024 on Z-Image and 8243 to 966 at
            // 1600x900, which is what used to run past the largest block the card
            // allows. Also 12 to 14% quicker, same image (45.8 to 50.6 dB).
            "--diffusion-fa",
            // The output format follows the file extension (PNG lossless, JPG lighter).
            "-o", out.path,
        ]
        let negative = negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !negative.isEmpty { args += ["-n", negative] }
        // Preview method is user-picked (none / proj / vae / tae). `proj` needs no
        // extra weights; `tae` needs a Tiny AE on disk (--taesd).
        let preview = FileManager.default.temporaryDirectory
            .appendingPathComponent("toshllm-preview-\(token).png")
        previewURL = preview
        if previewMode != .none {
            args += ["--preview", previewMode.rawValue, "--preview-interval", "3",
                     "--preview-path", preview.path]
            if previewMode == .tae, let tae = ImageGenPool.taesdPath() {
                args += ["--taesd", tae.path, "--taesd-preview-only"]
            }
        } else {
            args += ["--preview", "none"]
        }
        // img2img: seed the run with an existing image. Strength is how much to
        // change it (0 keeps it, 1 regenerates from scratch).
        if !initImagePath.isEmpty {
            args += ["-i", initImagePath, "--strength", String(format: "%.2f", strength)]
            if !maskPath.isEmpty { args += ["--mask", maskPath] }
        }
        // Reference images (edit mode): each one is read by the text encoder and encoded
        // by the VAE. Every reference lengthens the sequence the model attends over, so past
        // two they share a budget of two full-size references instead of one each.
        for ref in refs { args += ["-r", ref] }
        if let px = ImageGenLimits.referencePixels(count: refs.count, resolution: model.referenceResolution) {
            args += ["--ref-image-args", "vae_input_max_pixels=\(px)"]
        }
        let split = auxGPUIndex >= 0 && auxGPUIndex != gpuIndex && gpuIndex >= 0
            && MTLCopyAllDevices().count > 1
        var extra = model.extraArgs
        if let tile = model.largeVAETile, let i = extra.firstIndex(of: "--vae-tile-size"), i + 1 < extra.count,
           ImageGenLimits.workingSetGB(gpuIndex: gpuIndex) >= 11.5 {
            extra[i + 1] = String(tile)
        }
        // User sampler/scheduler picks replace any catalog defaults in extraArgs.
        func stripFlag(_ name: String) {
            while let i = extra.firstIndex(of: name) {
                extra.remove(at: i)
                if i < extra.count { extra.remove(at: i) }
            }
        }
        if !samplingMethod.rawValue.isEmpty { stripFlag("--sampling-method") }
        if !scheduler.rawValue.isEmpty { stripFlag("--scheduler") }
        if flowShift >= 0 { stripFlag("--flow-shift") }
        if split {
            // Merge the split assignment with the model's own --backend (e.g.
            // qwen-image forces vae=cpu), which wins per module.
            var overrides: String? = nil
            if let i = extra.firstIndex(of: "--backend"), i + 1 < extra.count {
                overrides = extra[i + 1]
                extra.removeSubrange(i...(i + 1))
            }
            args += ["--backend", Self.splitBackendSpec(overriding: overrides)]
        }
        // Offloading the diffusion model to CPU trades speed for VRAM; measured to
        // make no difference here, so it's off by default and only a fallback.
        args += extra
        // TE residency (Advanced): te=cpu / full GPU / encode-then-unload.
        // Model --params-backend in extraArgs still wins for that flag; unload can stack.
        let teMode = ImageTEResidency(
            rawValue: UserDefaults.standard.string(forKey: SettingsKeys.imagenTEResidency) ?? "")
            ?? .encodeThenUnload
        if !args.contains("--params-backend") {
            args += teMode.cliArgs
        } else if teMode == .encodeThenUnload, !args.contains("--release-te-after-encode") {
            args.append("--release-te-after-encode")
        }
        if !samplingMethod.rawValue.isEmpty {
            args += ["--sampling-method", samplingMethod.rawValue]
        }
        if !scheduler.rawValue.isEmpty {
            args += ["--scheduler", scheduler.rawValue]
        }
        if flowShift >= 0 {
            args += ["--flow-shift", String(format: "%.3f", flowShift)]
        }
        if slgScale > 0 {
            args += ["--slg-scale", String(format: "%.3f", slgScale),
                     "--skip-layer-start", String(format: "%.3f", slgStart),
                     "--skip-layer-end", String(format: "%.3f", slgEnd)]
            let layers = slgLayers.split(whereSeparator: { ",; ".contains($0) })
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            if !layers.isEmpty {
                args += ["--skip-layers", layers.map(String.init).joined(separator: ",")]
            }
        }
        if !controlNetPath.isEmpty, FileManager.default.fileExists(atPath: controlNetPath) {
            args += ["--control-net", controlNetPath,
                     "--control-strength", String(format: "%.2f", controlStrength)]
            if !controlImagePath.isEmpty, FileManager.default.fileExists(atPath: controlImagePath) {
                args += ["--control-image", controlImagePath]
            }
            if !controlType.rawValue.isEmpty {
                args += ["--control-type", controlType.rawValue]
            }
        }
        if !extra.contains("--cache-mode") {
            args += fastMode.args(for: model)
        }
        if offloadToCPU && !extra.contains("--offload-to-cpu") {
            args.append("--offload-to-cpu")
        }
        pendingLatentURL = nil
        pendingBatchCount = batchN
        pendingOutputPattern = batchN > 1 ? out.path : nil
        pendingLatentPattern = nil
        pendingBaseSeed = seed
        pendingFullSteps = nil
        pendingEndedAtStep = nil
        lastLatentURL = nil
        lastFullSteps = nil
        lastEndedAtStep = nil
        if batchN > 1 {
            args += ["--batch-count", String(batchN), "--output-begin-idx", "0"]
        }
        if let handoff {
            args += ["--start-at-step", String(handoff.startAtStep),
                     "--end-at-step", String(handoff.endAtStep),
                     "--add-noise", handoff.addNoise ? "on" : "off"]
            if !handoff.latentInPath.isEmpty {
                args += ["--latent-in", handoff.latentInPath]
            }
            if handoff.writeLatent {
                let latentPattern = out.deletingPathExtension().appendingPathExtension("sdlt")
                args += ["--latent-out", latentPattern.path]
                pendingLatentPattern = batchN > 1 ? latentPattern.path : nil
                pendingFullSteps = handoff.fullSteps
                pendingEndedAtStep = handoff.endedAtStep
                lastFullSteps = handoff.fullSteps
                lastEndedAtStep = handoff.endedAtStep
                for i in 0..<batchN {
                    var meta = handoff
                    meta.seed = seed + i
                    let latentPath = batchN > 1
                        ? String(format: latentPattern.path, i)
                        : latentPattern.path
                    let latentURL = URL(fileURLWithPath: latentPath)
                    if let data = try? JSONEncoder().encode(meta) {
                        try? data.write(to: URL(fileURLWithPath: latentPath + ".json"))
                    }
                    if i == 0 {
                        pendingLatentURL = latentURL
                        lastLatentURL = latentURL
                    }
                }
            }
        }

        var env = ProcessInfo.processInfo.environment
        // The wide matmul tile is tuned for LLM prefill shapes; on diffusion it costs
        // 1.7% on SDXL and 3% on Wan, with byte-identical output. Measured 08-14.
        env["TOSH_MM_WIDE_DISABLE"] = "1"
        if model.halfPartials { env["TOSH_MM_ACC_HALF"] = "1" }
        // The flash-attention kernels for AMD are opt-in; without this the backend
        // turns --diffusion-fa down and the attention is materialized instead.
        env["TOSH_FA_AMD"] = "1"
        // AMD discrete GPUs corrupt output without disabling Metal concurrency;
        // Apple Silicon keeps it on. Mirrors the LLM engine defaults.
        // Split each step into enough command buffers to clear the GPU watchdog.
        env["GGML_METAL_NCB"] = String(ImageGenLimits.nCB(width: width, height: height))
        // Pin the chosen GPU (slot maps to MTLCopyAllDevices order, as in the picker).
        // Index 0 must also be exported: without it Metal falls back to the system
        // default, which on multi-GPU Macs may be a different card than picked.
        let devices = MTLCopyAllDevices()
        if split {
            // Two Metal slots: 0 = diffusion GPU, 1 = encoder/VAE GPU. The list
            // replaces DEVICE_INDEX (which would offset both slots).
            env["GGML_METAL_DEVICES"] = "2"
            env["GGML_METAL_DEVICE_LIST"] = "\(gpuIndex),\(auxGPUIndex)"
        } else if gpuIndex >= 0 && devices.count > 1 {
            env["GGML_METAL_DEVICE_INDEX"] = String(gpuIndex)
        }
        // Metal defaults eGPUs to shared buffers; force private VRAM like the LLM engine.
        let picked = split ? [gpuIndex, auxGPUIndex] : [gpuIndex]
        let anyExternal = picked.contains { $0 >= 0 && $0 < devices.count && devices[$0].location == .external }
            || (gpuIndex < 0 && MTLCreateSystemDefaultDevice()?.location == .external)
        if UserDefaults.standard.bool(forKey: SettingsKeys.forcePrivateBuffers) || anyExternal {
            env["GGML_METAL_SHARED_BUFFERS_DISABLE"] = "1"
        }

        fileLog.startSession()
        fileLog.append("$ sd-server " + args.joined(separator: " ") + "\n\n")

        resultImage = nil
        resultURL = nil
        previewImage = nil
        progress = 0
        stepText = ""
        stage = .loading
        state = .generating
        startedAt = Date()
        firstStepAt = nil
        runTask = Task { [weak self] in
            await self?.runOnServer(args: args, env: env, output: out, steps: steps)
        }
    }

    /// Poll the preview file while the run lasts. Reading it mid-write just yields
    /// nil, so a failed load is skipped rather than treated as an error.
    private func startPreviewWatch() {
        previewTimer?.invalidate()
        previewSeenAt = nil
        previewTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.refreshPreview() }
        }
    }

    private func refreshPreview() async {
        guard isBusy, let url = previewURL else { return }
        let mod = await Task.detached(priority: .utility) {
            try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        }.value
        guard isBusy, previewURL == url, let mod else { return }
        guard previewSeenAt != mod else { return }
        let data = await Task.detached(priority: .utility) { try? Data(contentsOf: url) }.value
        guard isBusy, previewURL == url else { return }
        if let data, let img = NSImage(data: data) {
            previewSeenAt = mod
            previewImage = img
        }
    }

    /// Parse the engine's per-step progress. Only the sampler prints a line ending
    /// in `- X.Xs/it`; other `N/M` output (image count, saved files) must not move
    /// the bar. Also note the decode stage so the UI can name the wait.
    func consume(_ text: String, steps: Int) {
        logTail = String((logTail + text).suffix(4000))
        fileLog.append(text)
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = String(raw)
            if line.lowercased().contains("decod") { stage = .decoding }
            guard line.contains("s/it"),
                  let r = line.range(of: #"(\d+)/(\d+)"#, options: .regularExpression) else { continue }
            let parts = line[r].split(separator: "/")
            if parts.count == 2, let a = Int(parts[0]), let b = Int(parts[1]), b > 0 {
                if firstStepAt == nil { firstStepAt = Date() }
                stage = .sampling
                progress = Double(a) / Double(b)
                stepText = "\(a)/\(b)"
            }
        }
    }

    func finish(status: Int32, output: URL) {
        process = nil
        previewTimer?.invalidate()
        previewTimer = nil
        if let p = previewURL { try? FileManager.default.removeItem(at: p) }
        previewImage = nil
        defer { onFinish?() }

        func cleanupPendingLatents() {
            let n = max(1, pendingBatchCount)
            if let pattern = pendingLatentPattern {
                for i in 0..<n {
                    let path = String(format: pattern, i)
                    try? FileManager.default.removeItem(atPath: path)
                    try? FileManager.default.removeItem(atPath: path + ".json")
                }
            } else if let latent = pendingLatentURL {
                try? FileManager.default.removeItem(at: latent)
                try? FileManager.default.removeItem(at: URL(fileURLWithPath: latent.path + ".json"))
            }
            pendingLatentURL = nil
            pendingLatentPattern = nil
            lastLatentURL = nil
            lastFullSteps = nil
            lastEndedAtStep = nil
            lastBatch = []
        }

        if status != 0 {
            cleanupPendingLatents()
            // 15/SIGTERM and 2/SIGINT are user cancels, not errors.
            if status == 15 || status == 2 { state = .failed(""); return }
            if logTail.contains("failed to allocate") { state = .failed("OOM"); return }
            let timedOut = logTail.contains("Timeout") || logTail.contains("status 5")
            state = .failed(timedOut ? "TIMEOUT" : "exit \(status)")
            return
        }

        let n = max(1, pendingBatchCount)
        var items: [FinishedImageItem] = []
        for i in 0..<n {
            let path = pendingOutputPattern.map { String(format: $0, i) } ?? output.path
            let url = URL(fileURLWithPath: path)
            guard let img = NSImage(contentsOf: url) else { continue }
            let latentPath = pendingLatentPattern.map { String(format: $0, i) }
                ?? pendingLatentURL?.path
            let latentURL = latentPath.flatMap {
                FileManager.default.fileExists(atPath: $0) ? URL(fileURLWithPath: $0) : nil
            }
            items.append(FinishedImageItem(
                url: url, image: img, seed: pendingBaseSeed >= 0 ? pendingBaseSeed + i : lastSeed,
                latentURL: latentURL, fullSteps: pendingFullSteps, endedAtStep: pendingEndedAtStep))
        }
        guard let first = items.first else {
            cleanupPendingLatents()
            state = .failed("exit \(status)")
            return
        }
        lastBatch = items
        // Canvas shows the first draft; the gallery gets every seed.
        resultImage = first.image
        resultURL = first.url
        lastSeed = first.seed
        lastLatentURL = first.latentURL
        lastFullSteps = first.fullSteps
        lastEndedAtStep = first.endedAtStep
        if let (pw, ph) = ImageGalleryStore.pixelSize(of: first.image) {
            if lastWidth <= 0 { lastWidth = pw }
            if lastHeight <= 0 { lastHeight = ph }
        } else if let (pw, ph) = ImageGalleryStore.pixelSize(of: first.url) {
            if lastWidth <= 0 { lastWidth = pw }
            if lastHeight <= 0 { lastHeight = ph }
        }
        progress = 1
        lastDuration = elapsed
        state = .done
    }
}

// MARK: - Parallel instances

/// Full configuration of one generation slot: every control the sidebar exposes,
/// so each instance can run its own model, prompt and size on its own GPU.
/// Decoding is field-tolerant so older payloads never wipe the user's setup.
struct ImageInstanceConfig: Codable, Identifiable, Equatable {
    var id = UUID()
    var modelID = ""
    var customModelPath = ""
    var customVAEPath = ""
    /// Text encoder for custom models that ship the DiT alone (Z-Image, Flux 2,
    /// Qwen-Image): those need an LLM encoder, a full checkpoint does not.
    var customTextEncoderPath = ""
    /// The custom file is a bare diffusion model (--diffusion-model) instead of a
    /// full checkpoint (--model). The engine cannot tell them apart by extension.
    var customIsDiffusion = false
    var customCfg = 7.0
    var prompt = ""
    /// Ignored at CFG 1: the sampler never runs the unconditional branch there.
    var negativePrompt = ""
    var initImagePath = ""
    /// Reference images for the models that edit from them (Qwen-Image 2.1).
    var referenceImagePaths: [String] = []
    /// Inpainting mask for the init image: white repaints, black keeps.
    var maskPath = ""
    var strength = 0.6
    var aspect = ImageAspect.square.rawValue
    /// Free "W:H" ratio used only when `aspect` is `.custom` and pixels are off.
    var customAspect = "21:9"
    /// When custom framing uses absolute pixels instead of ratio × base size.
    var customUsesPixels = true
    var customWidth = 1024
    var customHeight = 1024
    var baseSize = 1024
    var gpuIndex = 0
    /// GPU that hosts the text encoder + VAE, freeing `gpuIndex` for the
    /// diffusion model; -1 = everything on the same GPU.
    var auxGPUIndex = -1
    var steps = 8
    var seed = -1
    var format = ImageFormat.png.rawValue
    var offloadCPU = false
    var fastMode = ImageFastMode.off.rawValue
    var previewMode = ImagePreviewMode.proj.rawValue
    /// 0 = use the catalog model's `cfgScale`.
    var cfgScale = 0.0
    var samplingMethod = ImageSamplingMethod.auto.rawValue
    var scheduler = ImageScheduler.auto.rawValue
    /// < 0 = engine auto (MOSS default shift is 5).
    var flowShift = -1.0
    var slgScale = 0.0
    var slgStart = 0.01
    var slgEnd = 0.2
    /// Comma-separated DiT layer indices for SLG; empty keeps the engine default.
    var slgLayers = ""
    /// Comfy-style explore: N short drafts, then continue winners at full steps.
    var draftCount = 4
    var draftSteps = 8
    var controlNetPath = ""
    var controlImagePath = ""
    var controlStrength = 0.9
    var controlType = ImageControlType.auto.rawValue
    /// MOSS `--llm` quant: q4 / q8 (`ImageMossTextEncoder`).
    var mossTextEncoder = ImageMossTextEncoder.q8.rawValue

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decode(UUID.self, forKey: .id)) ?? UUID()
        modelID = (try? c.decode(String.self, forKey: .modelID)) ?? ""
        customModelPath = (try? c.decode(String.self, forKey: .customModelPath)) ?? ""
        customVAEPath = (try? c.decode(String.self, forKey: .customVAEPath)) ?? ""
        customTextEncoderPath = (try? c.decode(String.self, forKey: .customTextEncoderPath)) ?? ""
        customIsDiffusion = (try? c.decode(Bool.self, forKey: .customIsDiffusion)) ?? false
        customCfg = (try? c.decode(Double.self, forKey: .customCfg)) ?? 7.0
        prompt = (try? c.decode(String.self, forKey: .prompt)) ?? ""
        negativePrompt = (try? c.decode(String.self, forKey: .negativePrompt)) ?? ""
        initImagePath = (try? c.decode(String.self, forKey: .initImagePath)) ?? ""
        referenceImagePaths = (try? c.decode([String].self, forKey: .referenceImagePaths)) ?? []
        maskPath = (try? c.decode(String.self, forKey: .maskPath)) ?? ""
        strength = (try? c.decode(Double.self, forKey: .strength)) ?? 0.6
        aspect = (try? c.decode(String.self, forKey: .aspect)) ?? ImageAspect.square.rawValue
        customAspect = (try? c.decode(String.self, forKey: .customAspect)) ?? "21:9"
        customUsesPixels = (try? c.decode(Bool.self, forKey: .customUsesPixels)) ?? true
        customWidth = (try? c.decode(Int.self, forKey: .customWidth)) ?? 1024
        customHeight = (try? c.decode(Int.self, forKey: .customHeight)) ?? 1024
        baseSize = (try? c.decode(Int.self, forKey: .baseSize)) ?? 1024
        gpuIndex = (try? c.decode(Int.self, forKey: .gpuIndex)) ?? 0
        auxGPUIndex = (try? c.decode(Int.self, forKey: .auxGPUIndex)) ?? -1
        steps = (try? c.decode(Int.self, forKey: .steps)) ?? 8
        seed = (try? c.decode(Int.self, forKey: .seed)) ?? -1
        format = (try? c.decode(String.self, forKey: .format)) ?? ImageFormat.png.rawValue
        offloadCPU = (try? c.decode(Bool.self, forKey: .offloadCPU)) ?? false
        fastMode = (try? c.decode(String.self, forKey: .fastMode)) ?? ImageFastMode.off.rawValue
        previewMode = (try? c.decode(String.self, forKey: .previewMode)) ?? ImagePreviewMode.proj.rawValue
        cfgScale = (try? c.decode(Double.self, forKey: .cfgScale)) ?? 0
        samplingMethod = (try? c.decode(String.self, forKey: .samplingMethod)) ?? ImageSamplingMethod.auto.rawValue
        scheduler = (try? c.decode(String.self, forKey: .scheduler)) ?? ImageScheduler.auto.rawValue
        flowShift = (try? c.decode(Double.self, forKey: .flowShift)) ?? -1
        slgScale = (try? c.decode(Double.self, forKey: .slgScale)) ?? 0
        slgStart = (try? c.decode(Double.self, forKey: .slgStart)) ?? 0.01
        slgEnd = (try? c.decode(Double.self, forKey: .slgEnd)) ?? 0.2
        slgLayers = (try? c.decode(String.self, forKey: .slgLayers)) ?? ""
        draftCount = (try? c.decode(Int.self, forKey: .draftCount)) ?? 4
        draftSteps = (try? c.decode(Int.self, forKey: .draftSteps)) ?? 8
        controlNetPath = (try? c.decode(String.self, forKey: .controlNetPath)) ?? ""
        controlImagePath = (try? c.decode(String.self, forKey: .controlImagePath)) ?? ""
        controlStrength = (try? c.decode(Double.self, forKey: .controlStrength)) ?? 0.9
        controlType = (try? c.decode(String.self, forKey: .controlType)) ?? ImageControlType.auto.rawValue
        mossTextEncoder = (try? c.decode(String.self, forKey: .mossTextEncoder))
            ?? ImageMossTextEncoder.q8.rawValue
    }

    var isCustom: Bool { modelID == ImageGenCatalog.customID }

    /// Auxiliary GPU for the encoder/VAE split, or nil when the split is off,
    /// points at the main GPU, or the slot no longer exists.
    func auxGPU(gpuCount: Int) -> Int? {
        auxGPUIndex >= 0 && auxGPUIndex != gpuIndex && auxGPUIndex < gpuCount && gpuCount > 1
            ? auxGPUIndex : nil
    }

    var aspectValue: ImageAspect { ImageAspect(rawValue: aspect) ?? .square }
    var formatValue: ImageFormat { ImageFormat(rawValue: format) ?? .png }
    var fastModeValue: ImageFastMode { ImageFastMode(rawValue: fastMode) ?? .off }
    var previewModeValue: ImagePreviewMode { ImagePreviewMode(rawValue: previewMode) ?? .proj }
    var samplingMethodValue: ImageSamplingMethod { ImageSamplingMethod(rawValue: samplingMethod) ?? .auto }
    var schedulerValue: ImageScheduler { ImageScheduler(rawValue: scheduler) ?? .auto }
    var controlTypeValue: ImageControlType { ImageControlType(rawValue: controlType) ?? .auto }
    var mossTextEncoderValue: ImageMossTextEncoder {
        ImageMossTextEncoder(rawValue: mossTextEncoder) ?? .q8
    }

    func effectiveCfg(for model: ImageGenModel) -> Double {
        if isCustom { return customCfg }
        return cfgScale > 0 ? cfgScale : model.cfgScale
    }
    var dimensions: (Int, Int) {
        guard aspectValue == .custom else { return aspectValue.dimensions(base: baseSize) }
        if customUsesPixels {
            return (ImageAspect.gridSnap(Double(customWidth)),
                    ImageAspect.gridSnap(Double(customHeight)))
        }
        return ImageAspect.customDimensions(ratio: customAspect, base: baseSize)
    }

    /// The model this instance runs: a catalog entry, the user's own files, or
    /// the best pick for the hardware.
    func resolvedModel(for hw: HardwareInfo) -> ImageGenModel {
        if isCustom {
            return ImageGenCatalog.custom(modelPath: customModelPath, vaePath: customVAEPath,
                                          textEncoderPath: customTextEncoderPath,
                                          isDiffusion: customIsDiffusion,
                                          steps: steps, cfg: customCfg)
        }
        let base = ImageGenCatalog.model(id: modelID)
            ?? ImageGenCatalog.recommended(for: hw)
            ?? ImageGenCatalog.zImageTurbo
        guard ImageGenCatalog.isMossFamily(modelID) || ImageGenCatalog.isMossFamily(base.id) else {
            return base
        }
        return base.withTextEncoder(mossTextEncoderValue.component)
    }
}

/// Step window plus the identity a draft must match before Continue.
/// Absent on a normal generation, so that run's CLI stays unchanged.
struct ImageLatentHandoff: Codable, Equatable {
    var startAtStep: Int
    var endAtStep: Int
    var addNoise: Bool
    var latentInPath: String
    var writeLatent: Bool
    var fullSteps: Int
    var endedAtStep: Int
    var seed: Int
    var sampler: String
    var scheduler: String
    var flowShift: Double
    var cfgScale: Double
    var modelID: String
    var textEncoder: String
    var width: Int
    var height: Int
    var prompt: String
    var negativePrompt: String
}

/// A prompt (with its own seed) waiting for the next free instance to render it.
struct QueuedPrompt: Identifiable, Equatable {
    let id = UUID()
    var text: String
    var seed: Int = -1
    /// Overrides the instance step count when set (draft explore / continue).
    var steps: Int? = nil
    /// Instance this must run on, by config id; nil = any free instance. If the
    /// target instance is later removed, the scheduler treats it as untargeted.
    var targetInstanceID: UUID? = nil
    /// img2img source for this prompt only; nil = the instance's own init image.
    var initImagePath: String? = nil
    /// Set for a draft or a draft Continue. Nil keeps the original full-schedule run.
    var handoff: ImageLatentHandoff? = nil
    /// Seeds seed..<seed+batchCount in one sd-cli run (one TE encode).
    var batchCount: Int = 1
}

/// A finished image kept in the session gallery, with the inputs that made it.
struct GeneratedImage: Identifiable {
    let id: UUID
    let url: URL
    let image: NSImage
    let prompt: String
    let seed: Int
    let width: Int
    let height: Int
    let duration: Int
    /// "index · model" of the instance that rendered it; nil if it was removed
    /// from the pool before this result was recorded.
    let instanceLabel: String?
    /// Draft latent to resume. Nil for a normal image and for a finished Continue.
    let latentURL: URL?
    let fullSteps: Int?
    let endedAtStep: Int?
    /// Disk sidecar used by the Gallery tab to restore params after relaunch.
    let record: ImageGalleryRecord?

    init(id: UUID = UUID(), url: URL, image: NSImage, prompt: String, seed: Int,
         width: Int, height: Int, duration: Int, instanceLabel: String?,
         latentURL: URL?, fullSteps: Int?, endedAtStep: Int?,
         record: ImageGalleryRecord? = nil) {
        self.id = id
        self.url = url
        self.image = image
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.duration = duration
        self.instanceLabel = instanceLabel
        self.latentURL = latentURL
        self.fullSteps = fullSteps
        self.endedAtStep = endedAtStep
        self.record = record
    }

    var canContinue: Bool {
        guard let latentURL, let fullSteps, let endedAtStep else { return false }
        return endedAtStep > 0 && endedAtStep < fullSteps
            && FileManager.default.fileExists(atPath: latentURL.path)
    }

    static func from(record: ImageGalleryRecord) -> GeneratedImage? {
        let enriched = ImageGalleryStore.enrich(record)
        guard FileManager.default.fileExists(atPath: enriched.imagePath),
              let image = NSImage(contentsOf: enriched.imageURL) else { return nil }
        var w = enriched.width
        var h = enriched.height
        if w <= 0 || h <= 0, let (pw, ph) = ImageGalleryStore.pixelSize(of: image) {
            w = pw; h = ph
        }
        return GeneratedImage(
            id: enriched.id, url: enriched.imageURL, image: image, prompt: enriched.prompt,
            seed: enriched.seed, width: w, height: h,
            duration: enriched.duration, instanceLabel: enriched.instanceLabel,
            latentURL: enriched.latentURL, fullSteps: enriched.fullSteps,
            endedAtStep: enriched.endedAtStep, record: enriched)
    }

    var resolvedSize: (Int, Int) {
        if width > 0, height > 0 { return (width, height) }
        if let r = record, r.width > 0, r.height > 0 { return (r.width, r.height) }
        if let (w, h) = ImageGalleryStore.pixelSize(of: image) { return (w, h) }
        if let (w, h) = ImageGalleryStore.pixelSize(of: url) { return (w, h) }
        return (0, 0)
    }

    var sizeLabel: String {
        let (w, h) = resolvedSize
        return w > 0 && h > 0 ? "\(w)×\(h)" : "—"
    }
}

/// Owns the generation instances and one generator per slot; the sidebar
/// (controls) and the canvas (tiles) share it so both see the same state.
/// There is always at least one instance.
@MainActor
final class ImageGenPool: ObservableObject {
    @Published var configs: [ImageInstanceConfig] { didSet { save() } }
    /// Pending prompts; the scheduler feeds them to idle instances.
    @Published var queue: [QueuedPrompt] = []
    /// Why the last Continue was refused, or empty. Codes: LATENT_MISSING, LATENT_MISMATCH.
    @Published var continueNotice = ""
    /// While true, finished instances pull the next queued prompt automatically.
    @Published var queueActive = false
    /// Results from this process only (Instance strip + Queue feed). Newest first.
    @Published var gallery: [GeneratedImage] = []
    /// Disk library for the Gallery tab (survives relaunch). Newest first.
    @Published var libraryGallery: [GeneratedImage] = []
    /// Set by the view so the scheduler can launch runs (generate() needs it).
    weak var modelStore: ModelStore?
    private var generators: [UUID: ImageGenerator] = [:]
    private var forwards: [UUID: AnyCancellable] = [:]

    init() {
        if let data = UserDefaults.standard.data(forKey: SettingsKeys.imagenInstances),
           let c = try? JSONDecoder().decode([ImageInstanceConfig].self, from: data),
           !c.isEmpty {
            configs = c
        } else {
            configs = [Self.legacyConfig()]
        }
    }

    /// Reload the on-disk library into `libraryGallery` without touching this session.
    func reloadPersistedGallery(from models: ModelStore) {
        let records = ImageGalleryStore.scan(imagenDirectory: models.imagenDirectory)
        libraryGallery = records.compactMap(GeneratedImage.from(record:))
    }

    func importRecord(_ record: ImageGalleryRecord, into instanceID: UUID? = nil) {
        let enriched = ImageGalleryStore.enrich(record)
        let target = instanceID.flatMap { id in configs.firstIndex(where: { $0.id == id }) } ?? 0
        guard configs.indices.contains(target) else { return }
        enriched.apply(to: &configs[target])
    }

    func removeFromLibrary(_ id: UUID) {
        guard let idx = libraryGallery.firstIndex(where: { $0.id == id }) else { return }
        let item = libraryGallery.remove(at: idx)
        if let record = item.record {
            ImageGalleryStore.delete(record)
        } else {
            try? FileManager.default.removeItem(at: item.url)
            try? FileManager.default.removeItem(at: ImageGalleryRecord.sidecar(for: item.url))
        }
        if let models = modelStore {
            ImageGalleryStore.saveIndex(libraryGallery.compactMap(\.record), in: models.imagenDirectory)
        }
        gallery.removeAll { $0.id == id }
    }

    /// Prefer library removal when the Gallery tab asks; falls back to session.
    func removeFromGallery(_ id: UUID) {
        if libraryGallery.contains(where: { $0.id == id }) {
            removeFromLibrary(id)
            return
        }
        gallery.removeAll { $0.id == id }
    }

    /// Seed instance 1 from the pre-multi-instance per-key settings, so an
    /// updated app keeps the user's prompt and choices.
    private static func legacyConfig() -> ImageInstanceConfig {
        let d = UserDefaults.standard
        func int(_ key: String, _ def: Int) -> Int { d.object(forKey: key) == nil ? def : d.integer(forKey: key) }
        func dbl(_ key: String, _ def: Double) -> Double { d.object(forKey: key) == nil ? def : d.double(forKey: key) }
        var c = ImageInstanceConfig()
        c.modelID = d.string(forKey: SettingsKeys.imagenModel) ?? ""
        c.customModelPath = d.string(forKey: SettingsKeys.imagenCustomModel) ?? ""
        c.customVAEPath = d.string(forKey: SettingsKeys.imagenCustomVAE) ?? ""
        c.customTextEncoderPath = d.string(forKey: SettingsKeys.imagenCustomTextEncoder) ?? ""
        c.customIsDiffusion = d.bool(forKey: SettingsKeys.imagenCustomIsDiffusion)
        c.customCfg = dbl(SettingsKeys.imagenCustomCfg, 7.0)
        c.prompt = d.string(forKey: SettingsKeys.imagenPrompt) ?? ""
        c.initImagePath = d.string(forKey: SettingsKeys.imagenInitImage) ?? ""
        c.strength = dbl(SettingsKeys.imagenStrength, 0.6)
        c.aspect = d.string(forKey: SettingsKeys.imagenAspect) ?? ImageAspect.square.rawValue
        c.baseSize = int(SettingsKeys.imagenBaseSize, 1024)
        c.gpuIndex = int(SettingsKeys.imagenGPU, 0)
        c.steps = int(SettingsKeys.imagenSteps, 8)
        c.seed = int(SettingsKeys.imagenSeed, -1)
        c.format = d.string(forKey: SettingsKeys.imagenFormat) ?? ImageFormat.png.rawValue
        c.offloadCPU = d.bool(forKey: SettingsKeys.imagenOffloadCPU)
        return c
    }

    /// Stable generator per slot, created lazily. Its changes are forwarded so
    /// views observing the pool re-render on per-instance progress.
    func generator(for id: UUID) -> ImageGenerator {
        if let g = generators[id] { return g }
        let g = ImageGenerator()
        g.onFinish = { [weak self, weak g] in
            guard let self, let g else { return }
            self.recordResult(from: g, instanceID: id)
            self.pump()
        }
        generators[id] = g
        forwards[id] = g.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        return g
    }

    /// "index · model" label matching the accordion title, for the queue target
    /// picker and the feed badges. Nil if the instance no longer exists.
    func instanceLabel(for id: UUID) -> String? {
        guard let idx = configs.firstIndex(where: { $0.id == id }) else { return nil }
        return "\(idx + 1) · \(configs[idx].resolvedModel(for: hardware).name)"
    }

    var anyBusy: Bool { configs.contains { generator(for: $0.id).isBusy } }
    var anyResult: Bool { configs.contains { generator(for: $0.id).resultImage != nil } }

    /// New instance as a copy of the last one, landing on a free GPU when there
    /// is one. The prompt is not copied: empty means "inherit instance 1's".
    func add() {
        var c = configs.last ?? ImageInstanceConfig()
        c.id = UUID()
        c.prompt = ""
        let used = Set(configs.map(\.gpuIndex))
        let candidates = hardware.gpus.filter { !$0.isIntegrated }.map(\.index)
        if let free = (candidates.isEmpty ? Array(0 ..< max(hardware.gpus.count, 1)) : candidates)
            .first(where: { !used.contains($0) }) {
            c.gpuIndex = free
        }
        configs.append(c)
    }

    /// The prompt an instance will actually run: its own, or instance 1's when
    /// left empty (shared prompt with per-instance override).
    func effectivePrompt(for c: ImageInstanceConfig) -> String {
        let own = c.prompt.trimmingCharacters(in: .whitespaces)
        if !own.isEmpty || c.id == configs.first?.id { return own }
        return (configs.first?.prompt ?? "").trimmingCharacters(in: .whitespaces)
    }

    /// Same inheritance as `effectivePrompt`, for the negative prompt.
    func effectiveNegativePrompt(for c: ImageInstanceConfig) -> String {
        let own = c.negativePrompt.trimmingCharacters(in: .whitespaces)
        if !own.isEmpty || c.id == configs.first?.id { return own }
        return (configs.first?.negativePrompt ?? "").trimmingCharacters(in: .whitespaces)
    }

    func remove(_ id: UUID) {
        guard configs.count > 1 else { return }
        generator(for: id).cancel()
        configs.removeAll { $0.id == id }
        generators[id] = nil
        forwards[id] = nil
    }

    func cancelAll() {
        queueActive = false
        for c in configs { generator(for: c.id).cancel() }
    }

    // MARK: prompt queue

    func enqueue(_ text: String, seed: Int = -1, steps: Int? = nil,
                 targetInstanceID: UUID? = nil, initImagePath: String? = nil,
                 handoff: ImageLatentHandoff? = nil, batchCount: Int = 1) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        let img = handoff == nil && initImagePath?.isEmpty == false ? initImagePath : nil
        queue.append(QueuedPrompt(text: t, seed: seed, steps: steps,
                                  targetInstanceID: targetInstanceID, initImagePath: img,
                                  handoff: handoff, batchCount: max(1, min(batchCount, 16))))
        if queueActive { pump() }
    }

    /// One sd-cli run: TE encodes once, then MOSS samples N seeds on 0…D of S.
    func enqueueDraftBatch(for c: ImageInstanceConfig) {
        let prompt = effectivePrompt(for: c)
        guard !prompt.isEmpty else { return }
        continueNotice = ""
        let model = c.resolvedModel(for: hardware)
        guard model.sigmas.isEmpty else { return }
        let full = model.steps(c.steps)
        let draft = max(1, c.draftSteps)
        guard draft < full else { return }
        let n = max(1, min(c.draftCount, 16))
        let (w, h) = c.dimensions
        let negative = effectiveNegativePrompt(for: c)
        // Leave room so seed+i stays inside Int32 for the engine.
        let seed = Int.random(in: 0..<(Int(Int32.max) - n))
        let handoff = ImageLatentHandoff(
            startAtStep: 0, endAtStep: draft, addNoise: true, latentInPath: "",
            writeLatent: true, fullSteps: full, endedAtStep: draft, seed: seed,
            sampler: c.samplingMethod, scheduler: c.scheduler, flowShift: c.flowShift,
            cfgScale: c.effectiveCfg(for: model), modelID: model.name,
            textEncoder: c.mossTextEncoder, width: w, height: h,
            prompt: prompt, negativePrompt: negative)
        enqueue(prompt, seed: seed, steps: full, targetInstanceID: c.id,
                handoff: handoff, batchCount: n)
        startQueue()
    }

    /// Resume a draft latent to the current Draft steps value (or full S).
    /// Draft steps > ended and < S keeps a continuable sketch; = S finishes a product.
    func continueFull(from result: GeneratedImage, on c: ImageInstanceConfig) {
        continueNotice = ""
        guard let latentURL = result.latentURL, result.canContinue,
              let ended = result.endedAtStep, let full = result.fullSteps else {
            continueNotice = "LATENT_MISSING"
            return
        }
        let sidecar = URL(fileURLWithPath: latentURL.path + ".json")
        guard let data = try? Data(contentsOf: sidecar),
              let saved = try? JSONDecoder().decode(ImageLatentHandoff.self, from: data) else {
            continueNotice = "LATENT_MISSING"
            return
        }
        let model = c.resolvedModel(for: hardware)
        let (w, h) = c.dimensions
        let same = saved.fullSteps == model.steps(c.steps)
            && c.samplingMethod == saved.sampler
            && c.scheduler == saved.scheduler
            && abs(c.flowShift - saved.flowShift) < 0.001
            && abs(c.effectiveCfg(for: model) - saved.cfgScale) < 0.001
            && model.name == saved.modelID
            && c.mossTextEncoder == saved.textEncoder
            && w == saved.width && h == saved.height
            && result.prompt == saved.prompt
            && result.seed == saved.seed
            && effectiveNegativePrompt(for: c) == saved.negativePrompt
            && saved.endedAtStep == ended
            && saved.fullSteps == full
        guard same else {
            continueNotice = "LATENT_MISMATCH"
            return
        }
        // Draft steps is the next stop. At or past full → finish the product.
        var nextEnd = max(1, c.draftSteps)
        if nextEnd > full { nextEnd = full }
        guard nextEnd > ended else {
            continueNotice = "LATENT_STEP"
            return
        }
        let keepSketch = nextEnd < full
        let handoff = ImageLatentHandoff(
            startAtStep: ended, endAtStep: nextEnd, addNoise: false,
            latentInPath: latentURL.path, writeLatent: keepSketch, fullSteps: full,
            endedAtStep: nextEnd, seed: result.seed, sampler: saved.sampler,
            scheduler: saved.scheduler, flowShift: saved.flowShift, cfgScale: saved.cfgScale,
            modelID: saved.modelID, textEncoder: saved.textEncoder, width: w, height: h,
            prompt: result.prompt, negativePrompt: saved.negativePrompt)
        enqueue(result.prompt, seed: result.seed, steps: full,
                targetInstanceID: c.id, handoff: handoff)
        startQueue()
    }

    /// Shared sd-cli launch from an instance config (+ optional step/seed/init overrides).
    func launchGenerate(_ c: ImageInstanceConfig, models: ModelStore,
                        prompt: String, steps: Int, seed: Int,
                        initImagePath: String? = nil,
                        handoff: ImageLatentHandoff? = nil,
                        batchCount: Int = 1) {
        let model = c.resolvedModel(for: hardware)
        let (w, h) = c.dimensions
        generator(for: c.id).generate(
            model: model, models: models, prompt: prompt,
            negativePrompt: effectiveNegativePrompt(for: c),
            width: w, height: h, steps: steps, seed: seed, format: c.formatValue,
            offloadToCPU: c.offloadCPU, gpuIndex: c.gpuIndex,
            auxGPUIndex: c.auxGPU(gpuCount: hardware.gpus.count) ?? -1,
            initImagePath: handoff == nil ? (initImagePath ?? c.initImagePath) : "", maskPath: c.maskPath,
            strength: c.strength, referenceImagePaths: c.referenceImagePaths,
            fastMode: c.fastModeValue, previewMode: c.previewModeValue,
            cfgScale: c.effectiveCfg(for: model),
            samplingMethod: c.samplingMethodValue, scheduler: c.schedulerValue,
            flowShift: c.flowShift,
            slgScale: c.slgScale, slgStart: c.slgStart, slgEnd: c.slgEnd,
            slgLayers: c.slgLayers,
            controlNetPath: c.controlNetPath, controlImagePath: c.controlImagePath,
            controlStrength: c.controlStrength, controlType: c.controlTypeValue,
            handoff: handoff, batchCount: batchCount)
    }

    func removeFromQueue(_ id: UUID) { queue.removeAll { $0.id == id } }

    func startQueue() { queueActive = true; pump() }
    func stopQueue() { queueActive = false }   // in-flight runs finish; nothing new starts

    /// Append a generator's finished result to the session gallery (newest first),
    /// so every render stays visible even as instances move on to the next prompt.
    private func recordResult(from gen: ImageGenerator, instanceID: UUID) {
        let label = instanceLabel(for: instanceID)
        let cfg = configs.first(where: { $0.id == instanceID })
        let items: [ImageGenerator.FinishedImageItem]
        if !gen.lastBatch.isEmpty {
            items = gen.lastBatch
        } else if let url = gen.resultURL, let img = gen.resultImage {
            let latent = gen.lastLatentURL.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil }
            items = [.init(url: url, image: img, seed: gen.lastSeed, latentURL: latent,
                           fullSteps: gen.lastFullSteps, endedAtStep: gen.lastEndedAtStep)]
        } else {
            return
        }
        // Insert last→first so the first seed lands at the top of a newest-first gallery.
        for item in items.reversed() {
            if gallery.first?.url == item.url { continue }
            var width = gen.lastWidth
            var height = gen.lastHeight
            if width <= 0 || height <= 0,
               let (pw, ph) = ImageGalleryStore.pixelSize(of: item.image)
                ?? ImageGalleryStore.pixelSize(of: item.url) {
                width = pw
                height = ph
            }
            let imagePath = ImageGalleryStore.canonicalPath(item.url.path)
            let imageURL = URL(fileURLWithPath: imagePath)
            let latentPath = item.latentURL.map { ImageGalleryStore.canonicalPath($0.path) }
            let record = ImageGalleryStore.enrich(ImageGalleryRecord(
                id: UUID(), imagePath: imagePath, createdAt: Date(),
                prompt: gen.lastPrompt, negativePrompt: gen.lastNegativePrompt,
                seed: item.seed, steps: gen.lastSteps,
                width: width, height: height, duration: gen.lastDuration,
                modelID: gen.lastModelID.isEmpty ? (cfg?.modelID ?? "") : gen.lastModelID,
                cfgScale: gen.lastCfgScale,
                samplingMethod: gen.lastSamplingMethod,
                scheduler: gen.lastScheduler, flowShift: gen.lastFlowShift,
                strength: gen.lastStrength,
                aspect: cfg?.aspect ?? "", baseSize: cfg?.baseSize ?? 1024,
                format: gen.lastFormat, initImagePath: gen.lastInitImagePath,
                mossTextEncoder: cfg?.mossTextEncoder ?? "",
                instanceLabel: label,
                latentPath: latentPath,
                fullSteps: item.fullSteps, endedAtStep: item.endedAtStep))
            if let models = modelStore {
                ImageGalleryStore.upsert(record, in: models.imagenDirectory)
            } else {
                ImageGalleryStore.writeSidecar(record)
            }
            let generated = GeneratedImage(id: record.id, url: imageURL, image: item.image,
                                           prompt: record.prompt, seed: record.seed,
                                           width: record.width, height: record.height,
                                           duration: record.duration, instanceLabel: label,
                                           latentURL: latentPath.map { URL(fileURLWithPath: $0) },
                                           fullSteps: item.fullSteps,
                                           endedAtStep: item.endedAtStep, record: record)
            gallery.insert(generated, at: 0)
            let canon = imagePath
            libraryGallery.removeAll {
                ImageGalleryStore.canonicalPath($0.url.path) == canon
            }
            libraryGallery.insert(generated, at: 0)
        }
        while gallery.count > ImageGalleryStore.maxRecords { gallery.removeLast() }
        while libraryGallery.count > ImageGalleryStore.maxRecords { libraryGallery.removeLast() }
    }

    /// True when `job` may run on `c`: untargeted, targeted at `c`, or its target
    /// was removed from the pool (falls back to "any free instance").
    nonisolated static func runnable(_ job: QueuedPrompt, on c: ImageInstanceConfig, existingIDs: Set<UUID>) -> Bool {
        guard let target = job.targetInstanceID else { return true }
        return target == c.id || !existingIDs.contains(target)
    }

    /// Assigns queued prompts to idle instances, at most one run per GPU (two
    /// Metal contexts on one AMD GPU can hang it). Each free instance takes its
    /// first *runnable* item, so a targeted prompt never blocks untargeted ones.
    func pump() {
        guard queueActive, let models = modelStore else { return }
        let gpuCount = hardware.gpus.count
        // A split instance occupies its encoder/VAE GPU too.
        var busyGPUs = Set<Int>()
        for c in configs where generator(for: c.id).isBusy {
            busyGPUs.insert(c.gpuIndex)
            if let aux = c.auxGPU(gpuCount: gpuCount) { busyGPUs.insert(aux) }
        }
        let existingIDs = Set(configs.map(\.id))
        for c in configs {
            guard !queue.isEmpty else { break }
            let gen = generator(for: c.id)
            let aux = c.auxGPU(gpuCount: gpuCount)
            guard !gen.isBusy, !busyGPUs.contains(c.gpuIndex),
                  aux.map({ !busyGPUs.contains($0) }) ?? true else { continue }
            guard let idx = queue.firstIndex(where: { Self.runnable($0, on: c, existingIDs: existingIDs) }) else { continue }
            let job = queue.remove(at: idx)
            busyGPUs.insert(c.gpuIndex)
            if let aux { busyGPUs.insert(aux) }
            launchGenerate(c, models: models, prompt: job.text,
                           steps: job.steps ?? c.steps, seed: job.seed,
                           initImagePath: job.initImagePath, handoff: job.handoff,
                           batchCount: job.batchCount)
        }
        if queue.isEmpty && !anyBusy { queueActive = false }
    }

    /// Deletes generated outputs (`toshllm_*`) from the imagen folder, when the
    /// user opts in. Upscales of those files are kept: they are the version the
    /// user asked to produce, not accumulated output.
    nonisolated static func cleanupOutputsIfEnabled() {
        guard UserDefaults.standard.bool(forKey: SettingsKeys.imagenCleanupOnClose) else { return }
        let dir = ServerSettings.modelsDirectory.appendingPathComponent("imagen", isDirectory: true)
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        for f in files {
            let name = f.deletingPathExtension().lastPathComponent
            guard name.hasPrefix("toshllm_"), !name.contains("-x2"), !name.contains("-x4") else { continue }
            try? FileManager.default.removeItem(at: f)
        }
    }

    /// LoRA folder for sd-cli `--lora-model-dir`. Prefers the directory that already
    /// holds files (`vision/loras`, then `imagen/lora`, then empty stubs).
    nonisolated static func loraDirectory() -> URL? {
        let root = ServerSettings.modelsDirectory
        let candidates = [
            root.appendingPathComponent("vision/loras", isDirectory: true),
            root.appendingPathComponent("vision/lora", isDirectory: true),
            root.appendingPathComponent("imagen/lora", isDirectory: true),
        ]
        let existing = candidates.filter { FileManager.default.fileExists(atPath: $0.path) }
        if let rich = existing.first(where: { !loraFiles(in: $0).isEmpty }) { return rich }
        return existing.first
    }

    /// Folder opened by the LoRA menu's "Open folder" action (created if missing).
    nonisolated static func loraDirectoryForOpen() -> URL {
        if let dir = loraDirectory() { return dir }
        let dir = ServerSettings.modelsDirectory
            .appendingPathComponent("vision/loras", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    nonisolated static func loraFiles(in dir: URL) -> [URL] {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.filter { ["safetensors", "ckpt", "pt"].contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent.localizedCaseInsensitiveCompare($1.lastPathComponent) == .orderedAscending }
    }

    /// Tiny AutoEncoder for `--preview tae`, if present under the models tree.
    nonisolated static func taesdPath() -> URL? {
        let root = ServerSettings.modelsDirectory
        let names = ["taesd.safetensors", "taef1.safetensors", "taeqi2_1.safetensors",
                     "diffusion_pytorch_model.safetensors"]
        let index = ModelFileIndex.scan(in: root)
        for name in names {
            if let url = index.file(named: name) { return url }
        }
        return nil
    }

    private func save() {
        UserDefaults.standard.set(try? JSONEncoder().encode(configs),
                                  forKey: SettingsKeys.imagenInstances)
    }
}

/// ESRGAN upscaling of an already generated image. Separate from ImageGenerator:
/// it reuses the finished PNG as input, so it never re-runs the diffusion model.
/// One finished upscale. Only paths: keeping every pair decoded in memory would
/// cost hundreds of MB on a batch.
struct UpscaleResult: Identifiable, Equatable {
    let id = UUID()
    let source: URL
    let output: URL
}

@MainActor
final class ImageUpscaler: ObservableObject {
    enum State: Equatable { case idle, running, done, failed(String) }

    @Published var state: State = .idle
    @Published var resultImage: NSImage?
    @Published var resultURL: URL?
    /// Kept so the canvas can show a before/after and the sidebar the file name.
    @Published var sourceImage: NSImage?
    @Published var sourceURL: URL?
    /// Tile progress parsed from the engine, 0...1.
    @Published var progress: Double = 0
    /// Position in the batch, 1-based; total is 1 for a single file.
    @Published var index = 0
    @Published var total = 0
    /// Picked but not started: choosing a file must not launch a GPU run.
    @Published var queued: [URL] = []
    /// Everything finished this session, so a batch does not hide its own results.
    @Published var done: [UpscaleResult] = []
    @Published var selected: UpscaleResult.ID?

    private var pending: [URL] = []
    private var flavor: Flavor = .photo
    private var customPath = ""
    private var scale: Scale = .x4
    private var models: ModelStore?
    private var gpuIndex = -1
    private var startedAt: Date?

    var elapsed: Int { startedAt.map { Int(-$0.timeIntervalSinceNow) } ?? 0 }

    /// Both are ESRGAN-architecture and load in sd-cli; they are specialists, not
    /// ranks. Measured on the same crop: x4plus keeps gradients smooth, UltraSharp
    /// adds contrast and micro-detail, which flatters generated art and invents
    /// texture on a real photo.
    /// sd.cpp only loads plain 4x RRDBNet: RealESRGAN x2plus uses pixel-unshuffle
    /// (12-channel input) and is rejected, and the DAT models that top the 2026
    /// rankings are a different architecture entirely. So 4x native, 2x by
    /// downsampling the 4x result, which also beats a native 2x in practice.
    enum Scale: String, CaseIterable, Identifiable {
        case x2, x4
        var id: String { rawValue }
        var factor: Int { self == .x2 ? 2 : 4 }
        var label: String { self == .x2 ? "×2" : "×4" }
    }

    enum Flavor: String, CaseIterable, Identifiable {
        case photo, art, custom
        var id: String { rawValue }

        func component(customPath: String = "") -> ImageGenComponent? {
            switch self {
            case .photo:
                return ImageGenComponent(kind: .checkpoint,
                    urlString: "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_esrgan/4xNomosWebPhoto_esrgan.pth",
                    fileName: "4xNomosWebPhoto_esrgan.pth", sizeGB: 0.032)
            case .art:
                return ImageGenComponent(kind: .checkpoint,
                    urlString: "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
                    fileName: "4x-UltraSharp.pth", sizeGB: 0.064)
            case .custom:
                guard !customPath.isEmpty else { return nil }
                return ImageGenComponent(kind: .checkpoint, urlString: "",
                                         fileName: customPath, sizeGB: 0)
            }
        }

        func label(_ spanish: Bool) -> String {
            switch self {
            case .photo:  return spanish ? "Fotos" : "Photos"
            case .art:    return spanish ? "Generada" : "Generated"
            case .custom: return spanish ? "Otro" : "Custom"
            }
        }

        func detail(_ spanish: Bool) -> String {
            switch self {
            case .photo:  return spanish ? "4xNomosWebPhoto. Entrenado con ruido, desenfoque y recompresión reales."
                                         : "4xNomosWebPhoto. Trained on real noise, lens blur and re-compression."
            case .art:    return spanish ? "4x-UltraSharp. Más contraste y detalle fino; luce en arte generado."
                                         : "4x-UltraSharp. More contrast and fine detail; shines on generated art."
            case .custom: return spanish ? "Tu propio modelo ESRGAN ×4 (.pth o .safetensors)."
                                         : "Your own 4x ESRGAN model (.pth or .safetensors)."
            }
        }
    }

    static func installed(_ flavor: Flavor, customPath: String, in models: ModelStore) -> Bool {
        guard let c = flavor.component(customPath: customPath) else { return false }
        return models.hasComponent(c)
    }

    var isBusy: Bool { state == .running }

    private var process: Process?

    /// Upscales the queued files one at a time: two ESRGAN runs at once would
    /// contend for the same VRAM the tiles already stretch.
    func upscale(sources: [URL], flavor: Flavor, customPath: String = "", scale: Scale = .x4,
                 models: ModelStore, gpuIndex: Int) {
        guard !isBusy, !sources.isEmpty else { return }
        pending = sources
        queued = []
        self.customPath = customPath
        self.scale = scale
        self.flavor = flavor
        self.models = models
        self.gpuIndex = gpuIndex
        index = 0
        total = sources.count
        startedAt = Date()
        next()
    }

    private func next() {
        guard let models, !pending.isEmpty else {
            if total > 0 { state = .done }
            return
        }
        let source = pending.removeFirst()
        index += 1
        run(source: source, flavor: flavor, models: models, gpuIndex: gpuIndex)
    }

    /// Next to the original, so upscaling a photo doesn't drop the result among
    /// the model weights. Falls back to the imagen folder when that is read-only.
    private func outputURL(for source: URL, fallback: URL) -> URL {
        let folder = source.deletingLastPathComponent()
        let dir = FileManager.default.isWritableFile(atPath: folder.path) ? folder : fallback
        let base = source.deletingPathExtension().lastPathComponent + "-x\(scale.factor)"
        var candidate = dir.appendingPathComponent(base + ".png")
        var n = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = dir.appendingPathComponent("\(base)-\(n).png")
            n += 1
        }
        return candidate
    }

    private func run(source: URL, flavor: Flavor, models: ModelStore, gpuIndex: Int) {
        let dir = models.imagenDirectory
        let out = outputURL(for: source, fallback: dir)

        guard let component = flavor.component(customPath: customPath) else {
            state = .failed("no model"); return
        }
        let args = ["-M", "upscale",
                    "--upscale-model", models.componentPath(component).path,
                    // tiled so a single Metal dispatch stays under the GPU watchdog
                    "--upscale-tile-size", "128",
                    "-i", source.path,
                    "-o", out.path]

        var env = ProcessInfo.processInfo.environment
        let devices = MTLCopyAllDevices()
        if gpuIndex >= 0 && devices.count > 1 {
            env["GGML_METAL_DEVICE_INDEX"] = String(gpuIndex)
        }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: ImageGenerator.binary)
        p.arguments = args
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.consume(text) }
        }
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                guard let self else { return }
                self.process = nil
                guard proc.terminationStatus == 0, let img = NSImage(contentsOf: out) else {
                    if proc.terminationStatus == 15 || proc.terminationStatus == 2 {
                        self.state = .failed("")
                    } else {
                        self.state = .failed("exit \(proc.terminationStatus)")
                    }
                    self.pending = []
                    return
                }
                if self.scale == .x2 { Self.halve(out) }
                self.resultImage = NSImage(contentsOf: out) ?? img
                self.resultURL = out
                self.progress = 1
                if let src = self.sourceURL {
                    let entry = UpscaleResult(source: src, output: out)
                    self.done.append(entry)
                    self.selected = entry.id
                }
                self.next()
            }
        }
        state = .running
        progress = 0
        resultImage = nil
        resultURL = nil
        sourceURL = source
        sourceImage = NSImage(contentsOf: source)
        do {
            try p.run()
            process = p
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func cancel() {
        pending = []
        process?.terminate()
    }

    /// x2 = the 4x result resampled to half. Done with ImageIO so the downsample
    /// is a proper Lanczos pass, not a display-time scale.
    nonisolated static func halve(_ url: URL) {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int else { return }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: max(1, w / 2),
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, cg, nil)
        CGImageDestinationFinalize(dest)
    }

    /// Same tile-progress line the generator prints: `N/M ... s/it`.
    private func consume(_ text: String) {
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = String(raw)
            guard line.contains("s/it"),
                  let r = line.range(of: #"(\d+)/(\d+)"#, options: .regularExpression) else { continue }
            let parts = line[r].split(separator: "/")
            if parts.count == 2, let a = Int(parts[0]), let b = Int(parts[1]), b > 0 {
                progress = Double(a) / Double(b)
            }
        }
    }
}
