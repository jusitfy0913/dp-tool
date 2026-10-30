# anomalib-training-export Specification

## Purpose
TBD - created by archiving change add-anomalib-training-export. Update Purpose after archive.
## Requirements
### Requirement: Anomalib Training Mode
The system SHALL provide an Anomalib anomaly detection mode in the existing TRTForge training/export page without removing or changing the existing YOLO training modes.

#### Scenario: User selects Anomalib mode
- **WHEN** the user selects `Anomalib 异常检测` as the training type
- **THEN** the UI SHALL show Anomalib model, Folder dataset, training, export, and metadata controls
- **AND** the UI SHALL hide or disable YOLO-only task controls that do not apply to anomaly detection

#### Scenario: User returns to YOLO mode
- **WHEN** the user switches the training type back to YOLO
- **THEN** the UI SHALL restore existing YOLO task, dataset, validation, and ONNX export controls
- **AND** the previous YOLO behavior SHALL remain available for `detect`, `segment`, `pose`, and `obb`

### Requirement: Anomalib Environment Detection
The system SHALL detect a usable Python environment before starting Anomalib training and export.

#### Scenario: Environment is ready
- **WHEN** the user clicks environment detection in Anomalib mode
- **AND** Python, torch, CUDA status, anomalib, lightning, onnx, onnxruntime, PIL, and numpy can be inspected
- **THEN** the launcher SHALL emit an `env` structured event
- **AND** the UI SHALL display dependency status without starting training

#### Scenario: Required dependency is missing
- **WHEN** a required dependency such as `anomalib` or `onnxruntime` is missing
- **THEN** the launcher SHALL emit a `TRTFORGE_JSON` error with code `MISSING_DEP`
- **AND** the Qt UI SHALL show a non-crashing error dialog that names the missing dependency

### Requirement: Unified Anomalib Launcher
The system SHALL expose a Python launcher for Anomalib training/export commands.

#### Scenario: C++ calls environment detection
- **WHEN** Qt starts `tf_anomalib_launcher.py --mode check-env`
- **THEN** the launcher SHALL inspect the runtime environment
- **AND** it SHALL write ordinary logs plus structured `TRTFORGE_JSON` events to stdout or stderr

#### Scenario: C++ calls train and export
- **WHEN** Qt starts `tf_anomalib_launcher.py --mode train-export` with a supported model and valid Folder dataset paths
- **THEN** the launcher SHALL train or fit the selected Anomalib model
- **AND** it SHALL export ONNX and generate metadata before reporting success

#### Scenario: Unsupported mode is requested
- **WHEN** the launcher receives an unsupported mode
- **THEN** it SHALL fail before training starts
- **AND** it SHALL emit a structured error with code `UNSUPPORTED_MODE`

### Requirement: Supported Anomalib Models
The system SHALL support PatchCore, PaDiM, and EfficientAd in the first Anomalib training/export version.

#### Scenario: PatchCore is selected
- **WHEN** the user selects PatchCore
- **THEN** the launcher SHALL use the PatchCore Anomalib model
- **AND** auto `max_epochs` SHALL resolve to `1`

#### Scenario: PaDiM is selected
- **WHEN** the user selects PaDiM
- **THEN** the launcher SHALL use the PaDiM Anomalib model
- **AND** auto `max_epochs` SHALL resolve to `1`

#### Scenario: EfficientAd is selected
- **WHEN** the user selects EfficientAd
- **THEN** the launcher SHALL use the EfficientAd Anomalib model
- **AND** auto `max_epochs` SHALL resolve to `20`
- **AND** the UI SHALL expose the EfficientAd `imagenet_dir` control

### Requirement: Folder Dataset Configuration
The system SHALL support Anomalib Folder datasets through explicit normal and abnormal directory fields.

#### Scenario: Valid Folder dataset is provided
- **WHEN** the user provides `root`, `normal_dir`, `abnormal_dir`, and `normal_test_dir`
- **THEN** the launcher SHALL resolve relative paths against `root`
- **AND** it SHALL validate that the referenced directories exist before training starts

#### Scenario: Dataset path is invalid
- **WHEN** one or more required dataset directories do not exist
- **THEN** the launcher SHALL emit a structured error with code `DATA_NOT_FOUND`
- **AND** the Qt UI SHALL keep the application running and guide the user back to the dataset fields

### Requirement: Training and Export Execution
The system SHALL run Anomalib training, test, ONNX export, and metadata generation as one user-facing operation.

#### Scenario: Train export succeeds
- **WHEN** the user clicks `开始训练并导出` with valid parameters
- **THEN** the launcher SHALL run the Anomalib training flow for the selected model
- **AND** it SHALL export `model.onnx`
- **AND** it SHALL write `metadata.json`
- **AND** it SHALL emit artifact events for `ckpt`, `onnx`, `metadata`, and the run directory when available

#### Scenario: User stops the process
- **WHEN** the user clicks `停止` during training or export
- **THEN** Qt SHALL terminate the child process using the existing controlled stop strategy
- **AND** the launcher or Qt wrapper SHALL report a `CANCELLED` state when possible

#### Scenario: Export fails
- **WHEN** ONNX export raises an exception
- **THEN** the launcher SHALL emit a structured error with code `EXPORT_FAILED`
- **AND** the UI SHALL show the failure without attempting TensorRT conversion

### Requirement: Metadata Contract
The system SHALL generate metadata compatible with TRTForge Anomalib TensorRT inference.

#### Scenario: PatchCore or PaDiM ONNX is post-processed
- **WHEN** PatchCore or PaDiM exports ONNX outputs that include post-processed `pred_score` and `anomaly_map`
- **THEN** the metadata SHALL set `output_contract` to `anomalib_post_processed_0_1`
- **AND** it SHALL set `normalization` to `none`
- **AND** it SHALL set image and pixel thresholds to `0.5` unless explicitly calibrated
- **AND** it SHALL set score and map ranges to `0.0` and `1.0`

#### Scenario: EfficientAd metadata is generated
- **WHEN** EfficientAd training/export completes and metadata calibration is enabled
- **THEN** the launcher SHALL use normal and abnormal calibration samples to estimate image and pixel thresholds
- **AND** it SHALL emit an `anomaly_export_summary` event with the selected thresholds and ranges

#### Scenario: Metadata generation fails
- **WHEN** metadata cannot be written or calibrated
- **THEN** the launcher SHALL emit a structured error with code `METADATA_FAILED`
- **AND** the UI SHALL not mark the Anomalib export as complete

### Requirement: Structured Output Protocol
The system SHALL reuse the `TRTFORGE_JSON` structured output protocol for Anomalib training/export.

#### Scenario: Ordinary Python output is printed
- **WHEN** Python, Anomalib, Lightning, or a dependency writes ordinary stdout or stderr text
- **THEN** the Qt UI SHALL append the text to a visible `QTextBrowser` runtime log
- **AND** the log SHALL include training, validation or test, ONNX export, metadata generation, and error messages when those messages are printed by Python

#### Scenario: Structured event is printed
- **WHEN** a line starts with `TRTFORGE_JSON `
- **THEN** Qt SHALL parse the JSON payload as a structured event
- **AND** parsing failure SHALL degrade to ordinary log display rather than crashing
- **AND** the presence of structured events SHALL NOT remove the original Python line from the user's runtime log when it is useful for troubleshooting

#### Scenario: Export summary is emitted
- **WHEN** training and export complete
- **THEN** the launcher SHALL emit `anomaly_model` and `anomaly_export_summary` events
- **AND** the UI SHALL use those events to show model type, input size, output contract, thresholds, and artifact paths

### Requirement: Anomalib Button Policy
The system SHALL present Anomalib-specific actions that match the train/export workflow.

#### Scenario: Anomalib mode is active
- **WHEN** Anomalib mode is selected
- **THEN** the UI SHALL show `检测环境`, `开始训练并导出`, and `停止`
- **AND** it SHALL hide or disable YOLO-only `验证模型` and separate `导出 ONNX` actions

#### Scenario: YOLO mode is active
- **WHEN** YOLO mode is selected
- **THEN** the UI SHALL show the existing YOLO actions according to the current YOLO training design

### Requirement: Anomalib Settings Persistence
The system SHALL persist Anomalib-specific training settings independently from YOLO settings.

#### Scenario: User edits Anomalib settings
- **WHEN** the user changes model, dataset, output, training, or metadata calibration fields
- **THEN** the UI SHALL save those values under `training/anomalib/*` QSettings keys
- **AND** it SHALL not overwrite YOLO task, model, dataset, validation, or export settings

#### Scenario: User reopens TRTForge
- **WHEN** the training/export page is opened again
- **THEN** the UI SHALL restore the last selected training mode
- **AND** it SHALL restore Anomalib fields from the Anomalib QSettings keys when Anomalib mode is active

### Requirement: Non-Crashing Error Handling
The system SHALL convert recoverable Anomalib runtime and dependency errors into visible UI messages without crashing TRTForge.

#### Scenario: Unknown exception occurs
- **WHEN** the launcher catches an unexpected exception
- **THEN** it SHALL emit an error event with code `UNHANDLED_EXCEPTION`
- **AND** it SHALL include a concise message and optional traceback summary
- **AND** the Qt application SHALL remain running

#### Scenario: Python exits non-zero
- **WHEN** the Anomalib child process exits with a non-zero code
- **THEN** the Qt UI SHALL mark the run as failed
- **AND** it SHALL preserve the log output and visible error message for troubleshooting

