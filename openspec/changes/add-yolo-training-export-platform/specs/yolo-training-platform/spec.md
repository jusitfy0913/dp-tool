## ADDED Requirements

### Requirement: Python Environment Detection
The system SHALL detect a usable Python environment before starting YOLO training, validation, or export.

#### Scenario: Environment is ready
- **WHEN** the user clicks environment detection
- **AND** Python, torch, CUDA status, ultralytics, onnx, and onnxruntime can be inspected
- **THEN** the system SHALL show a ready state in the UI
- **AND** the system SHALL display Python path, Python version, torch version, CUDA availability, and dependency status

#### Scenario: Required dependency is missing
- **WHEN** a required dependency such as `ultralytics` or `onnx` is missing
- **THEN** the Python launcher SHALL output a `TRTFORGE_JSON` error with code `MISSING_DEP`
- **AND** the Qt UI SHALL show a non-crashing error dialog with the missing package and suggested installation command

### Requirement: Unified YOLO Launcher
The system SHALL expose a single Python launcher for C++/Qt calls.

#### Scenario: C++ calls a training command
- **WHEN** Qt starts `tf_yolo_launcher.py --mode train --task detect`
- **THEN** the launcher SHALL call Ultralytics YOLO training for the `detect` task
- **AND** the launcher SHALL use stdout/stderr for logs and `TRTFORGE_JSON` for structured events

#### Scenario: Unsupported task is requested
- **WHEN** the launcher receives a task outside `detect`, `segment`, `pose`, and `obb`
- **THEN** the launcher SHALL fail before training starts
- **AND** it SHALL emit a structured error with code `UNSUPPORTED_TASK`

### Requirement: YOLO Task Head Coverage
The system SHALL support `detect`, `segment`, `pose`, and `obb` task heads in the first version.

#### Scenario: User selects a task head
- **WHEN** the user selects one of `detect`, `segment`, `pose`, or `obb`
- **THEN** the UI SHALL show common training parameters
- **AND** it SHALL show only the task-specific parameters for the selected task

#### Scenario: OBB task is selected
- **WHEN** the user selects `obb`
- **THEN** the UI SHALL expose the `angle` loss weight
- **AND** the launcher SHALL accept YOLO OBB datasets in Ultralytics four-point label format

#### Scenario: Segment task is selected
- **WHEN** the user selects `segment`
- **THEN** the UI SHALL expose `overlap_mask` and `mask_ratio`

#### Scenario: Pose task is selected
- **WHEN** the user selects `pose`
- **THEN** the UI SHALL expose `pose`, `kobj`, and `rle`

### Requirement: Training Execution
The system SHALL train YOLO models through Ultralytics in a Python child process.

#### Scenario: Training succeeds
- **WHEN** the user starts training with a valid model and dataset
- **THEN** the launcher SHALL run Ultralytics `YOLO(...).train(...)`
- **AND** it SHALL emit progress and metric structured events during training
- **AND** it SHALL emit artifact events for `best.pt` and `last.pt` when available
- **AND** it SHALL emit epoch summary, plot, GPU status, and labels JSON artifact events when available

#### Scenario: Training fails
- **WHEN** training raises an exception
- **THEN** the launcher SHALL exit with a non-zero code
- **AND** it SHALL emit an error event containing an error code, message, and optional traceback summary
- **AND** the Qt application SHALL remain running

### Requirement: Training Initialization Mode
The system SHALL let the user choose whether training loads pretrained weights, starts from a model structure, or resumes from a checkpoint.

#### Scenario: Fine-tuning with pretrained weights
- **WHEN** the user selects pretrained fine-tuning
- **THEN** the launcher SHALL pass a `.pt` model or `pretrained=true` or a specific pretrained path to Ultralytics
- **AND** the UI SHALL describe this mode as the recommended default for most small and medium datasets

#### Scenario: Training from scratch
- **WHEN** the user selects training from scratch
- **THEN** the launcher SHALL allow a model `.yaml`
- **AND** it SHALL pass `pretrained=false` to Ultralytics training
- **AND** the UI SHALL warn that this mode needs more data and usually takes longer

#### Scenario: Resuming interrupted training
- **WHEN** the user selects checkpoint resume
- **THEN** the launcher SHALL use a checkpoint such as `last.pt`
- **AND** it SHALL pass `resume=true`
- **AND** the UI SHALL treat resume as different from pretrained fine-tuning

### Requirement: Complete Python Log Echo
The system SHALL show Python stdout and stderr output in the Qt training log panel.

#### Scenario: Python prints ordinary text
- **WHEN** Python code, Ultralytics, or a dependency writes normal stdout or stderr text
- **THEN** the Qt log panel SHALL display that text without requiring it to be valid JSON

#### Scenario: Structured event is printed
- **WHEN** a line starts with `TRTFORGE_JSON `
- **THEN** Qt SHALL parse the JSON payload as a structured event
- **AND** parsing failure SHALL degrade to ordinary log display rather than crashing

### Requirement: Real-Time Training Monitor
The system SHALL provide a monitor view for current YOLO training metrics, plots, and GPU usage.

#### Scenario: Epoch finishes during training
- **WHEN** one epoch finishes
- **THEN** the launcher SHALL emit an `epoch_summary` event with current metrics
- **AND** the Qt UI SHALL append one dark-theme row to the training metrics table
- **AND** the Qt UI SHALL update chart widgets for F1, precision, recall, mAP, and GPU memory based on current epoch values
- **AND** line chart widgets SHALL show X/Y axes, tick values, grid lines, and the latest epoch value
- **AND** confusion matrix widgets SHALL show the latest `confusion_matrix` and `confusion_matrix_normalized` PNG thumbnails when available
- **AND** the Qt UI SHALL keep latest training plot image paths as artifacts rather than using them as the primary chart controls
- **AND** double-clicking one chart widget SHALL enlarge it inside the latest training charts area, and double-clicking again SHALL restore the multi-chart layout

#### Scenario: User starts a new training run
- **WHEN** the user clicks start training
- **THEN** the Qt UI SHALL clear previous metric rows, chart widgets, GPU status, and artifact summaries before the new process starts

#### Scenario: GPU status is available
- **WHEN** the launcher can inspect CUDA or `nvidia-smi`
- **THEN** it SHALL emit `gpu_status` events with GPU name, memory usage, and utilization when available
- **AND** the Qt UI SHALL show GPU memory and utilization as real-time text and progress bars to help the user tune batch size

### Requirement: TRTForge Label JSON Export
The system SHALL generate label JSON files compatible with the TRTForge inference label format after training.

#### Scenario: Training finishes successfully
- **WHEN** a training run finishes and class names can be read from the dataset or model
- **THEN** the launcher SHALL generate `labels_<experiment>.json`
- **AND** the JSON SHALL contain `name` and `classes` fields compatible with `labels_coco.json`
- **AND** it SHALL save a copy in the training run directory and a copy under `trtexec/gui/bin/configs`
- **AND** it SHALL emit `artifact` events with kind `labels_json`

#### Scenario: Label JSON name already exists
- **WHEN** a target labels JSON path already exists
- **THEN** the launcher SHALL create a non-overwriting filename by appending a timestamp or suffix

### Requirement: Model Validation
The system SHALL validate trained YOLO models and present performance metrics.

#### Scenario: Validation succeeds
- **WHEN** the user runs validation against a valid `best.pt` and `dataset.yaml`
- **THEN** the launcher SHALL call Ultralytics `YOLO(...).val(...)`
- **AND** it SHALL emit metrics including precision, recall, mAP50, mAP50-95, and speed values when available

#### Scenario: Validation split is selected
- **WHEN** the user selects `val`, `test`, or `train` as validation split
- **THEN** the launcher SHALL pass the selected split to Ultralytics validation

### Requirement: ONNX Export
The system SHALL export trained YOLO models to ONNX.

#### Scenario: ONNX export succeeds
- **WHEN** the user exports a valid YOLO weight with `format=onnx`
- **THEN** the launcher SHALL call Ultralytics `YOLO(...).export(format="onnx")`
- **AND** it SHALL emit an artifact event with the original ONNX path

#### Scenario: Transposed ONNX is requested
- **WHEN** `make_transd=true` is provided
- **THEN** the launcher SHALL generate a `.transd.onnx` file from the original ONNX
- **AND** it SHALL emit an artifact event with kind `transd_onnx`
- **AND** the UI SHALL label the file as intended for TRTForge/TensorRT custom decode use

### Requirement: Parameter Layering
The system SHALL layer YOLO parameters so normal users are not exposed to every advanced Ultralytics option at once.

#### Scenario: User opens the training page
- **WHEN** the YOLO Training Lab is displayed
- **THEN** common parameters SHALL be visible by default
- **AND** advanced parameters SHALL be grouped and collapsed or visually separated
- **AND** training initialization mode SHALL be visible near the model weight selector

#### Scenario: User opens advanced settings
- **WHEN** the user expands advanced settings
- **THEN** the UI SHALL show training control, resource, optimizer, data strategy, and loss-weight parameters

### Requirement: Non-Crashing Error Dialogs
The system SHALL convert recoverable Python and dependency errors into visible UI messages without crashing TRTForge.

#### Scenario: Dataset path is invalid
- **WHEN** the launcher receives a missing `dataset.yaml`
- **THEN** it SHALL emit an error with code `DATA_NOT_FOUND`
- **AND** the UI SHALL show a dialog that points the user back to the dataset field

#### Scenario: Export dependency is missing
- **WHEN** ONNX export is requested but `onnx` is missing
- **THEN** the launcher SHALL emit an error with code `MISSING_DEP`
- **AND** the UI SHALL not start TensorRT conversion

### Requirement: Future Service Evolution
The design SHALL preserve a path from Python child process execution to a local training service.

#### Scenario: Future service replaces child process
- **WHEN** a later version introduces a local HTTP or gRPC training service
- **THEN** the existing mode, task, parameter, and structured event concepts SHALL remain compatible enough for Qt UI reuse
