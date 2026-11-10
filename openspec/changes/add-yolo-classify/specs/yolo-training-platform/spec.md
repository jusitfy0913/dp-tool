## MODIFIED Requirements

### Requirement: Unified YOLO Launcher
The system SHALL expose a single Python launcher for C++/Qt calls.

#### Scenario: C++ calls a training command
- **WHEN** Qt starts `tf_yolo_launcher.py --mode train --task detect`
- **THEN** the launcher SHALL call Ultralytics YOLO training for the `detect` task
- **AND** the launcher SHALL use stdout/stderr for logs and `TRTFORGE_JSON` for structured events

#### Scenario: Unsupported task is requested
- **WHEN** the launcher receives a task outside `detect`, `segment`, `pose`, `obb`, and `classify`
- **THEN** the launcher SHALL fail before training starts
- **AND** it SHALL emit a structured error with code `UNSUPPORTED_TASK`

### Requirement: YOLO Task Head Coverage
The system SHALL support `detect`, `segment`, `pose`, `obb`, and `classify` task heads.

#### Scenario: User selects a task head
- **WHEN** the user selects one of `detect`, `segment`, `pose`, `obb`, or `classify`
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

#### Scenario: Classify task is selected
- **WHEN** the user selects `classify`
- **THEN** the launcher SHALL train an Ultralytics classification model from a class-folder dataset
- **AND** the launcher SHALL reject a detection model weight before training starts and tell the user to choose a classification weight

### Requirement: Model Validation
The system SHALL validate trained YOLO models and present performance metrics.

#### Scenario: Validation succeeds
- **WHEN** the user runs validation against a valid `best.pt` and `dataset.yaml` for `detect`, `segment`, `pose`, or `obb`
- **THEN** the launcher SHALL call Ultralytics `YOLO(...).val(...)`
- **AND** it SHALL emit metrics including precision, recall, mAP50, mAP50-95, and speed values when available

#### Scenario: Validation split is selected
- **WHEN** the user selects `val`, `test`, or `train` as validation split
- **THEN** the launcher SHALL pass the selected split to Ultralytics validation

#### Scenario: Classification validation
- **WHEN** the user validates a `classify` model
- **THEN** the launcher SHALL emit top-1 accuracy and a class-by-class confusion matrix
- **AND** it SHALL NOT emit detection mAP as the classification result
