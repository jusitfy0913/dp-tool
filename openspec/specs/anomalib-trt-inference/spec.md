# anomalib-trt-inference Specification

## Purpose
定义 Qt 版 Anomalib 的 TensorRT 推理链路：`anomalib_infer.dll` 按 `metadata.json` 装载引擎，
输出原始分数、归一化分数与异常图，界面据此展示判定、热力图与异常轮廓，并保证中文路径可用。
## Requirements
### Requirement: Anomalib TensorRT Mode
TRTForge SHALL provide an `异常检测 ANOMALY` mode in the inference demo for running existing Anomalib TensorRT engines.

#### Scenario: Select anomaly mode
- **WHEN** the user selects `异常检测 ANOMALY`
- **THEN** the inference demo SHALL switch from YOLO labels/boxes UI to Anomalib metadata/heatmap/mask UI

#### Scenario: Training is out of scope
- **WHEN** the user uses anomaly mode
- **THEN** the system SHALL only load existing engine and metadata files and SHALL NOT start Anomalib training or export

### Requirement: Anomalib Inference DLL
TRTForge SHALL package Anomalib TensorRT inference as an independent `anomalib_infer` dynamic library with a C ABI.

#### Scenario: TRTForge uses DLL
- **WHEN** anomaly mode starts inference
- **THEN** `AnomalyInferWorker` SHALL call `anomalib_infer` C API instead of owning TensorRT Runtime, Engine, Context, and CUDA buffers directly

#### Scenario: External C++ uses DLL
- **WHEN** an external C++ program links or loads `anomalib_infer`
- **THEN** it SHALL be able to create a handle, run RGB image inference, deep-copy returned results, and destroy the handle

#### Scenario: Python uses DLL
- **WHEN** a Python program loads `anomalib_infer.dll` with `ctypes`
- **THEN** it SHALL be able to declare the C structs, call inference, deep-copy returned images, and save overlay/mask outputs

#### Scenario: Result memory lifecycle
- **WHEN** `ai_infer_rgb` returns an `ai_result`
- **THEN** all result pointers SHALL remain valid until the next `ai_infer_rgb` call or `ai_destroy`, and callers SHALL be documented to deep-copy data they keep

### Requirement: Metadata Loading
TRTForge SHALL load Anomalib `metadata.json` and expose the inference parameters needed by post-processing.

#### Scenario: Load valid metadata
- **WHEN** the user selects a valid `metadata.json`
- **THEN** the UI SHALL display input transform size, image threshold, pixel threshold, score range, and anomaly map range

#### Scenario: Load dotted-key metadata
- **WHEN** the metadata uses keys such as `pred_scores.min` and `anomaly_maps.max`
- **THEN** the system SHALL parse them as score and anomaly map normalization ranges

#### Scenario: Metadata load failure
- **WHEN** the metadata file is missing, unreadable, or lacks required fields
- **THEN** the UI SHALL show a clear failure reason and SHALL prevent starting anomaly inference

### Requirement: Metadata Export Contract
The Anomalib example export flow SHALL generate a TRTForge-readable `metadata.json` beside `model.ckpt` and `model.onnx`.

#### Scenario: Export ONNX after training
- **WHEN** the example `train.py` successfully exports `model.onnx`
- **THEN** it SHALL write `metadata.json` containing input size, thresholds, score range, anomaly map range, model type, and trace paths

#### Scenario: Compare checkpoint and ONNX
- **WHEN** the user runs the ckpt/onnx comparison script with `model.ckpt`, `model.onnx`, `metadata.json`, and an image or image directory
- **THEN** the script SHALL produce a CSV report with inference timing, score comparison, OK/NG comparison, and anomaly-map difference summary

### Requirement: Anomaly Output Parsing
TRTForge SHALL parse EfficientAd, PaDiM, PatchCore, and AnomalyDINO TensorRT outputs into anomaly map and anomaly score by resolving outputs **by name first, then by element count**, and the C++ DLL and the Python-side helper MUST apply the same rule.

#### Scenario: 先按名字匹配
- **WHEN** an output tensor name contains `score`
- **THEN** the system SHALL treat it as the anomaly score
- **AND** when a name contains `anomaly_map`, or contains `map` but not `mask`, it SHALL be treated as the anomaly map

#### Scenario: 元素数最多者作为异常图
- **WHEN** any output has more elements than the currently selected anomaly map
- **THEN** that output SHALL replace the anomaly map selection
- **AND** an engine whose extra output is larger than the real map MUST be treated as a contract violation, not silently rendered

#### Scenario: 单元素输出补位成分数
- **WHEN** no output name contains `score`
- **THEN** an output with exactly one element whose name does not contain `label` SHALL be used as the score

#### Scenario: One output tensor
- **WHEN** the engine returns exactly one output tensor
- **THEN** the system SHALL treat it as the anomaly map and derive the score from the map maximum

#### Scenario: Two output tensors
- **WHEN** the engine returns two output tensors
- **THEN** the system SHALL resolve them by name first, then by element count
- **AND** when neither name matches, the larger (element-count) output SHALL be the anomaly map and the single-element output SHALL be the score
- **AND** the result MUST NOT depend on the order of the two outputs

#### Scenario: Three output tensors
- **WHEN** the engine returns three or more output tensors, such as Anomalib's `pred_label`
- **THEN** the system SHALL keep the name-based selections above
- **AND** it MUST NOT fall back to a fixed positional rule such as "the third output is the map"

#### Scenario: AnomalyDINO 两个输出
- **WHEN** the engine returns `anomaly_map` `[1,1,gh,gw]` and `pred_score` `[1,1]` in either order
- **THEN** the system SHALL select `anomaly_map` as the map and `pred_score` as the score
- **AND** the result MUST NOT depend on the order of the two outputs

#### Scenario: 分数与异常图是同一个张量
- **WHEN** the score selection resolves to the same tensor as the anomaly map
- **THEN** the system SHALL drop the explicit score and derive it from the map maximum

### Requirement: Visual Results
TRTForge SHALL display anomaly inference results as heatmap, mask, overlay, score, and OK/NG status.

#### Scenario: Infer an image or frame
- **WHEN** anomaly inference succeeds for an input image or video frame
- **THEN** the UI SHALL display a visual result image and update score, threshold, and OK/NG status

#### Scenario: Score exceeds threshold
- **WHEN** the normalized anomaly score is greater than or equal to the image threshold
- **THEN** the UI SHALL mark the result as NG

#### Scenario: Score below threshold
- **WHEN** the normalized anomaly score is below the image threshold
- **THEN** the UI SHALL mark the result as OK

### Requirement: Unicode Path Compatibility
TRTForge SHALL support Chinese and non-ASCII paths for anomaly engine, metadata, and input media.

#### Scenario: Load files from Chinese path
- **WHEN** engine, metadata, image, or video paths contain Chinese characters
- **THEN** anomaly inference SHALL load them without path encoding failure or provide a specific readable error

