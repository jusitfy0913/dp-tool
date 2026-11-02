# anomalib-trt-inference Specification

## Purpose
TBD - created by archiving change add-anomalib-trt-inference. Update Purpose after archive.
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
TRTForge SHALL parse EfficientAd, PaDiM, and PatchCore TensorRT outputs into anomaly map and anomaly score.

#### Scenario: One output tensor
- **WHEN** the engine returns one output tensor
- **THEN** the system SHALL treat it as anomaly map and derive score from the map maximum

#### Scenario: Two output tensors
- **WHEN** the engine returns two output tensors
- **THEN** the system SHALL treat the first output as score and the second output as anomaly map

#### Scenario: Three output tensors
- **WHEN** the engine returns three or more output tensors
- **THEN** the system SHALL use the EfficientAd-compatible rule that takes the third output as anomaly map and derives score from that map

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

