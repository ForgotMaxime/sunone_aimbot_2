# TensorRT Export (Frozen Contract)

Static, reproducible path for `sunone_aimbot_2` detection.

## 1. ONNX preparation (host or target)

```bash
pip install onnx onnxsim
python -m onnxsim input.onnx input.sim.onnx
```

`onnxsim` (simplifier) runs **before** the in-app builder: shape inference,
const-folding, dead-layer elimination. Hand `*.sim.onnx` to the Inference node.

## 2. Inference node settings

| Parameter | Value | Why |
|---|---|---|
| `fixed_input_size` | `1` | Forces static `1x3x320x320`; required for CUDA-graphs + letterbox math |
| `export_enable_fp16` | `1` | FP16 precision |
| `export_enable_fp8` | `0` | FP8 off unless the GPU is known-good (SM90+) |
| `trt_profile_min/opt/max` | `320 / 320 / 320` | Ignored when static, kept equal as guard |
| `confidence_threshold` | `0.35` (raw) | Per-class `objectness*classScore`; graph FilterConfidence refines |
| `nms_threshold` | `0.5` raw YOLO / `1` EfficientNMS | `1` disables CPU NMS when the ONNX already contains NMS |
| `max_detections` | `20..50` | Raw decode clamped to `[20,50]` in `TrtDetector::postProcess` |
| `use_pinned_memory` | `1` | Required for async DToH + CUDA-graphs |
| `use_cuda_graph` | `1` | After static + pinned are green |

## 3. Builder (in-app, `tensorrt/nvinf.cpp`)

- Input: one NCHW RGB image, static `1x3x320x320`.
- Workspace pool: **2 GiB** (`kWORKSPACE`, up to 4 GiB on large GPUs).
- Flags: `kFP16` (+ `kFP8` if requested). Engine cache key includes
  TRT version + GPU name + SM + ONNX mtime/size, so **build on the target GPU**.
- Copy the generated `.engine` next to the `.onnx`; never mix GPUs.

## 4. Runtime

- Preprocess (GPU, one stream): `cvtColor -> letterbox resize (fill 114)
  -> convertTo(1/255) -> fused hwc_to_chw_norm kernel`. Same math on CPU fallback.
- Letterbox inverse: `x_src = (x_net - padX) / scale`, clipped to the packet.
  Old `img_scale / imageScaleY` stretch hack removed.
- CUDA-graph captures **inference + DToH only** after one blank `enqueueV3`
  warmup + `cudaStreamSynchronize`. OpenCV kernels stay outside the graph.
- `EfficientNMS_TRT` (optional export `nms=True`): output `[N,6]` xyxy decoded
  directly; CPU NMS preserved for raw YOLO, bypassed with `nms_threshold=1`.

## 5. Tracker

`detector/motion_tracker.h` (header-only, motion-only): hysteresis
`0.35` confirm / `0.15` drop, EMA centre/size (`α=0.45`), constant-velocity
extrapolation on `frameTimestamp`, IoU `0.3`, `maxMissed=5`. Runs every frame in
`vision_runtime::processFrame`; supports detect 1/2 frames (prediction fills
skipped frames) without touching capture/mouse/overlay.

## 6. Depth (untouched)

`depth_anything_v2_vits_224` stays async at **10–15 Hz**
(`DepthInference depth_fps`, `DepthToMask depth_mask_fps=5`).
Do not enlarge it; detection path is independent.
