# MOSS-Image HTTP API 使用说明（给局域网内的 agent）

服务由 `sd-server` 提供，本机 `http://127.0.0.1:1234`，局域网
`http://192.168.1.53:1234`。**没有任何鉴权**，只适合内网使用。

## 1. 服务端启动

```sh
cd /Users/bemly/Projects/mps/engines/vision/stable-diffusion.cpp
./build-upstream/bin/sd-server \
  --model-dir /Users/bemly/Projects/mps/models/vision \
  --diffusion-model /Users/bemly/Projects/mps/models/vision/diffusion_models/MOSS-Image-Base-Q8_0.gguf \
  --llm            /Users/bemly/Projects/mps/models/vision/text_encoders/Qwen3.5-9B-Q8_0.gguf \
  --vae            /Users/bemly/Projects/mps/models/vision/vae/flux2-vae.safetensors \
  --tokenizer      /Users/bemly/Projects/mps/models/vision/tokenizers/tokenizer.json \
  --auto-fit off --cfg-scale 3.0 --rng cpu \
  -l 0.0.0.0 --listen-port 1234
```

- 加载完（约 4 s）`GET /` 返回 WebUI，`GET /v1/models` 返回 `sd-cpp-local`
- 显存占用 13.43 GB，16 GB 卡不要再并发别的 GPU 任务
- 停止：`kill <pid>`

## 2. 官方推荐参数（MOSS-Image-Base）

| 项目 | 值 | API 字段 |
|---|---|---|
| 采样器 | Flow-matching Euler | `sample_params.sample_method = "euler"` |
| 调度器 | Discrete | `sample_params.scheduler = "discrete"` |
| 步数 | **≥ 50**（20–30 可用，8 只够看构图） | `sample_params.sample_steps` |
| CFG | **3.0**（3.0–6.0） | `sample_params.guidance.txt_cfg` |
| Shift | 5.0（模型默认，一般不用改） | `sample_params.flow_shift`（0 = 用默认） |
| 跳层 | 无条件分支跳过第 22 层（**已内置，不用传**） | — |
| 分辨率 | 0.25–4 MP；推荐 1024×1024 / 1456×1024 / 1456×1456；**宽高必须是 16 的倍数** | `width` / `height` |
| 随机种子 | 固定值可复现，`-1` 随机 | `seed` |
| 语言 | 英文效果最好，中文支持 | `prompt` |

**耗时参考**（RX 6800，全 GPU）：512² 8 步 37 s、512² 50 步 157 s、
1024² 50 步 **808 s（13.5 分钟）**。

**SLG**：`sample_params.guidance.slg.{layers, layer_start, layer_end, scale}`，
`scale = 0` 就是关闭（默认）。它和内置的"跳第 22 层"不是一回事，开了会改画质，
没有明确目的就别动。

## 3. 三种 API 怎么选

| 场景 | 用哪个 |
|---|---|
| **agent 调用（推荐）** | `POST /sdcpp/v1/img_gen` 异步任务 + 轮询，不会被 13 分钟的长任务拖死 |
| 现成的 OpenAI 客户端 | `POST /v1/images/generations`（同步阻塞，参数要用 `sd_cpp_extra_args` 扩展） |
| 老 WebUI 客户端 | `POST /sdapi/v1/txt2img`（同步） |

## 4. 异步任务 API（推荐）

### 4.1 提交

```sh
curl -s -X POST http://192.168.1.53:1234/sdcpp/v1/img_gen \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a photo of a cat sitting on a wooden chair, warm sunlight",
    "negative_prompt": "blurry, low quality, watermark",
    "width": 512,
    "height": 512,
    "seed": 42,
    "batch_count": 1,
    "sample_params": {
      "scheduler": "discrete",
      "sample_method": "euler",
      "sample_steps": 50,
      "flow_shift": 0.0,
      "guidance": { "txt_cfg": 3.0 }
    },
    "preview": "proj",
    "output_format": "png"
  }'
```

返回 `202 Accepted`：

```json
{"id":"job_01HTXYZABC","kind":"img_gen","status":"queued",
 "created":1775401200000,"poll_url":"/sdcpp/v1/jobs/job_01HTXYZABC"}
```

### 4.2 轮询

```sh
curl -s http://192.168.1.53:1234/sdcpp/v1/jobs/job_01HTXYZABC
```

- 建议间隔 2–5 秒；`status` 变成 `completed` / `failed` / `cancelled` 就停
- `progress.stage` 看阶段：`conditioning` → `sampling` → `vae_decode` → `encoding`
- `progress.completed` / `progress.total`（`unit = "steps"`）是**当前 pass** 的进度
- `preview.b64_json` 是当前预览帧（`preview: "proj"` 时几乎零成本），
  发现构图不对可以立刻取消

### 4.3 取结果

```json
{
  "status": "completed",
  "result": {
    "output_format": "png",
    "images": [ { "index": 0, "b64_json": "iVBORw0KGgoAAA..." } ]
  }
}
```

图片是 base64，不含 data URI 前缀，直接 base64 解码写文件即可。

### 4.4 取消

```sh
curl -s -X POST http://192.168.1.53:1234/sdcpp/v1/jobs/job_01HTXYZABC/cancel
```

任务进入 `cancelled`，已占用的显存不释放（模型仍驻留），可以立刻提交下一个。

### 4.5 Python 示例

现成脚本：`scripts/generate_moss.py`（只用标准库，无第三方依赖）

```sh
python3 scripts/generate_moss.py                          # 1024x1024 / 20 步
python3 scripts/generate_moss.py --steps 50 -o final.png  # 正式出图
python3 scripts/generate_moss.py -W 1456 -H 1024 --seed -1
```

下面是同一流程的最小版本，方便嵌到别的代码里：

```python
import base64, time, requests

BASE = "http://192.168.1.53:1234"

def generate(prompt, width=1024, height=1024, steps=50, cfg=3.0, seed=42, timeout=1200):
    r = requests.post(f"{BASE}/sdcpp/v1/img_gen", json={
        "prompt": prompt,
        "width": width, "height": height, "seed": seed,
        "sample_params": {
            "scheduler": "discrete", "sample_method": "euler",
            "sample_steps": steps,
            "guidance": {"txt_cfg": cfg},
        },
        "preview": "proj",
        "output_format": "png",
    }, timeout=30)
    r.raise_for_status()
    job = r.json()["id"]

    t0 = time.time()
    while time.time() - t0 < timeout:
        j = requests.get(f"{BASE}/sdcpp/v1/jobs/{job}", timeout=10).json()
        if j["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(3)
    else:
        requests.post(f"{BASE}/sdcpp/v1/jobs/{job}/cancel", timeout=10)
        raise TimeoutError(job)

    if j["status"] != "completed":
        raise RuntimeError(f"{j['status']}: {j.get('error')}")
    return base64.b64decode(j["result"]["images"][0]["b64_json"])

open("out.png", "wb").write(generate("a photo of a cat"))
```

## 5. OpenAI 兼容 API

`POST /v1/images/generations` 只认 5 个字段：`prompt`、`n`、`size`、`output_format`、
`output_compression`。**CFG / steps / seed / sampler 都要塞进 `prompt` 里的
`<sd_cpp_extra_args>` 标签**，JSON 结构就是第 4.1 节的原生 body。

```sh
curl -s http://192.168.1.53:1234/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a photo of a cat <sd_cpp_extra_args>{\"sample_params\":{\"sample_steps\":50,\"scheduler\":\"discrete\",\"sample_method\":\"euler\",\"guidance\":{\"txt_cfg\":3.0}},\"seed\":42}<\/sd_cpp_extra_args>",
    "size": "512x512",
    "n": 1,
    "output_format": "png"
  }'
```

返回：`{"created":..., "output_format":"png", "data":[{"b64_json":"..."}]}`

要点：

- 这个接口**同步阻塞**，1024² 50 步会让连接挂 13 分钟，客户端超时要设够
- 标签会被服务端从提示词里剔除，不会影响出图
- `model` 字段会被忽略，永远用当前已加载的模型

## 6. 模型管理

```sh
GET  /sdcpp/v1/models/status     # 当前状态
GET  /sdcpp/v1/model-files       # 可选文件清单
GET  /sdcpp/v1/recipes           # recipe 默认值
POST /sdcpp/v1/models/load       # {"recipe":..., "models":{...}}
POST /sdcpp/v1/models/unload
POST /sdcpp/v1/models/refresh    # 重新扫描目录
```

- 三套 API **共用同一个 context**，load 完成后 OpenAI 接口自动用新模型
- 有任务在跑或排队 → `409 model_busy`，先 cancel
- 加载期间生成接口返回 `503 model_loading`
- **加载失败会卸掉当前模型**（`422 model_load_failed`），之后生成一律
  `503 model_not_loaded`，必须重新 load
- 相同 recipe + 文件再次 load 返回 `already_loaded: true`，不重复加载

## 7. 错误码

| HTTP | code | 含义 |
|---|---|---|
| 400 | `invalid_model_config` | 模型配置参数不合法 |
| 404 | — | job 不存在 |
| 409 | `model_busy` | 有任务在跑/排队，不能换模型 |
| 410 | — | job 已过期（取结果太晚） |
| 422 | `model_load_failed` | 模型初始化失败，看服务端日志 |
| 503 | `model_not_loaded` | 没有加载模型 |
| 503 | `model_loading` | 正在加载/卸载模型 |

## 8. 约束（排期前必读）

- **单 worker 串行**：一次只算一张，第二个请求排队（`queue_position`）
- **一次只有一个模型**，切换要卸载 13 GB 再加载，代价高
- **无鉴权、CORS 全开**：任何能访问该端口的人都能出图，只能内网
- 16 GB 显存单卡：不要和其他 GPU 任务（llama.cpp 等）并发
- 1024² 50 步约 13.5 分钟；512² 50 步约 2.6 分钟；扫 seed 用 512² + 20 步

## 9. WebUI

浏览器打开 `http://192.168.1.53:1234/`，字段对应关系：

| 页面 | 值 |
|---|---|
| Diffusion model | `diffusion_models/MOSS-Image-Base-Q8_0.gguf` |
| Text Encoder | `text_encoders/Qwen3.5-9B-Q8_0.gguf` |
| tokenizer | `tokenizers/tokenizer.json` |
| VAE | `vae/flux2-vae.safetensors` |
| **TAE** | **留空**（FLUX.2 没有 TAE，填了会 `model_load_failed`） |
| Sampler / Scheduler | `euler` / `discrete` |
| Steps / CFG | `50` / `3.0` |
| Live preview | `proj`（零成本，看构图够用）；要准就 `vae`（每步多 1~2 s） |
