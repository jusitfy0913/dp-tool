# paddleocr-cpp-inference Specification

## Purpose
定义 `ocr_infer.dll` 承载的 PaddleOCR C++ 推理能力：由 Paddle Inference 直接跑检测与识别模型，
返回带坐标与置信度的原始文本行，再由界面负责可视化，避免为 OCR 单独引入 Python 运行时。
## Requirements
### Requirement: PaddleOCR C++ Inference Mode
TRTForge SHALL provide an `OCR 文本识别 OCR` mode in the inference demo for running PaddleOCR C++ inference.

#### Scenario: Select OCR mode
- **WHEN** the user selects `OCR 文本识别 OCR`
- **THEN** the inference demo SHALL show OCR model path, device, thread, and threshold controls instead of YOLO label controls

#### Scenario: Run OCR on an input frame
- **WHEN** OCR inference succeeds for an image, video frame, or camera frame
- **THEN** the UI SHALL display OCR polygons on the image and show recognized text rows in the result list

### Requirement: OCR Inference DLL
TRTForge SHALL package PaddleOCR C++ inference as an independent `ocr_infer` dynamic library with a C ABI.

#### Scenario: TRTForge uses OCR DLL
- **WHEN** OCR mode starts inference
- **THEN** `OcrInferWorker` SHALL call `ocr_infer` C API instead of owning Paddle Inference predictors directly

#### Scenario: External caller uses OCR DLL
- **WHEN** an external C++ or Python ctypes caller loads `ocr_infer.dll`
- **THEN** it SHALL be able to create a handle, run OCR inference, read line-level raw results, and destroy the handle

#### Scenario: Result memory lifecycle
- **WHEN** `oi_infer_file` or `oi_infer_bgr` returns an `oi_result`
- **THEN** all result pointers SHALL remain valid until the next inference call or `oi_destroy`, and callers SHALL be documented to deep-copy data they keep

### Requirement: OCR Raw Result Contract
The OCR DLL SHALL return line-level raw OCR data with stable one-to-one correspondence between polygon, text, score, angle, and index.

#### Scenario: Read line results
- **WHEN** inference returns `line_count > 0`
- **THEN** each `lines[i]` SHALL include an `index`, polygon point arrays, recognized UTF-8 text, recognition score, and textline angle

#### Scenario: Preserve correspondence
- **WHEN** the caller compares `lines[i]` with `raw_json_utf8`
- **THEN** `lines[i].text_utf8`, `lines[i].rec_score`, and `lines[i].xs/ys` SHALL correspond to `rec_texts[i]`, `rec_scores[i]`, and `rec_polys[i]`

#### Scenario: No text detected
- **WHEN** OCR detects no text lines
- **THEN** the DLL SHALL return success with `line_count == 0` and a valid empty JSON result instead of treating it as an error

### Requirement: OCR Model Configuration
TRTForge SHALL configure PaddleOCR model directories and validate required inference files before starting OCR inference.

#### Scenario: Required models configured
- **WHEN** OCR mode starts
- **THEN** detection and recognition model directories SHALL be required

#### Scenario: Optional modules disabled
- **WHEN** textline orientation, document orientation, or document unwarping model directories are empty
- **THEN** the corresponding PaddleOCR optional modules SHALL be disabled

#### Scenario: Missing model file
- **WHEN** an enabled model directory lacks required files such as `inference.pdmodel` or `inference.json`, `inference.pdiparams`, or `inference.yml`
- **THEN** the system SHALL fail loading with a clear error that names the missing file

### Requirement: Paddle Inference Dependency Configuration
TRTForge SHALL build `ocr_infer.dll` against the Paddle Inference dependency directory under `4-qt-studio/bin/paddle_inference`.

#### Scenario: Link Paddle Inference
- **WHEN** CMake configures `ocr_infer`
- **THEN** it SHALL use `4-qt-studio/bin/paddle_inference` as the default `PADDLE_LIB` and link `paddle/lib/paddle_inference.lib`

#### Scenario: Runtime dependency missing
- **WHEN** Paddle runtime DLLs or third-party DLLs are missing at runtime
- **THEN** OCR mode SHALL provide a readable load or initialization error rather than crashing silently

### Requirement: OCR Visualization
TRTForge SHALL display OCR results as image overlays and a structured text list.

#### Scenario: Display OCR overlay
- **WHEN** OCR inference succeeds
- **THEN** the UI SHALL draw each text line polygon and an index number on the image

#### Scenario: Display OCR list
- **WHEN** OCR inference succeeds
- **THEN** the UI SHALL list each text line with index, recognized text, score, angle, and polygon coordinates

#### Scenario: Highlight selected text line
- **WHEN** the user selects a row in the OCR result list
- **THEN** the UI SHALL highlight the corresponding polygon on the image

### Requirement: OCR Validation Assets
TRTForge SHALL use the user-provided PaddleOCR test image directory for basic OCR validation.

#### Scenario: Collect test images
- **WHEN** OCR validation runs
- **THEN** it SHALL recursively collect common image formats from `ocr/PaddleOCR-main/docs/datasets/images`

#### Scenario: Validate raw data contract
- **WHEN** validation processes a test image
- **THEN** it SHALL check that text, score, and polygon arrays remain aligned for every returned OCR line

