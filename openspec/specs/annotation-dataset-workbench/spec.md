# annotation-dataset-workbench Specification

## Purpose
TBD - created by archiving change add-custom-dataset-split-ratio. Update Purpose after archive.
## Requirements
### Requirement: Custom Dataset Split Ratio
The annotation workbench SHALL allow users to configure train/val/test percentages for automatic dataset splitting.

#### Scenario: Apply configured ratio
- **WHEN** the user sets train/val/test percentages that sum to 100 and runs auto split
- **THEN** images are assigned to train, val, and test according to the configured percentages

#### Scenario: Restore last ratio
- **WHEN** the user reopens TRTForge after changing the split percentages
- **THEN** the annotation workbench restores the last saved percentages

#### Scenario: Reject invalid total
- **WHEN** the configured train/val/test percentages do not sum to 100
- **THEN** auto split is not applied and the user is prompted to correct the values

### Requirement: Annotation Workbench UI
TRTForge SHALL provide a native C++/Qt annotation workbench UI for image labeling.

#### Scenario: User opens annotation workbench
- **WHEN** the user opens TRTForge
- **THEN** the application SHALL provide an annotation workbench entry or tab
- **AND** the workbench SHALL show a left drawing toolbar, central image canvas, right label/object/property panels, and bottom image/dataset split area

#### Scenario: UI is approved before implementation
- **WHEN** annotation UI development is planned
- **THEN** a Pencil prototype SHALL be prepared and reviewed with the user before implementation
- **AND** implementation SHALL follow the user-approved UI direction

### Requirement: Single Task Project Mode
TRTForge SHALL store each annotation project with exactly one primary task type.

#### Scenario: User creates a project
- **WHEN** the user creates a new annotation project
- **THEN** the system SHALL require one task type from `detect`, `segment`, `pose`, or `obb`
- **AND** exported annotations SHALL follow that task type

#### Scenario: User changes task type
- **WHEN** the user attempts to change the project task type after annotations exist
- **THEN** the system SHALL warn that existing annotations may be incompatible
- **AND** the system SHALL require explicit user confirmation before changing the task type

### Requirement: TFAnno Project File
TRTForge SHALL use `.tfanno` as the native annotation project file extension.

#### Scenario: User saves annotation project
- **WHEN** the user saves a TRTForge annotation project
- **THEN** the system SHALL create or update a `.tfanno` project file
- **AND** the file SHALL preserve task type, classes, image list, annotations, and dataset split state

#### Scenario: User opens annotation project
- **WHEN** the user opens a `.tfanno` file
- **THEN** the system SHALL restore the project task, class order, images, annotation objects, and split assignments

### Requirement: Shape Drawing Support
TRTForge SHALL support drawing and editing the shapes needed by YOLO detection, segmentation, pose, and OBB workflows.

#### Scenario: User draws supported shapes
- **WHEN** the user selects a drawing tool
- **THEN** the workbench SHALL support horizontal rectangles, rotated rectangles, polygons, lines, points, and COCO 17 keypoints
- **AND** the created shape SHALL be associated with the current image

#### Scenario: User edits an annotation
- **WHEN** the user selects an existing annotation
- **THEN** the system SHALL allow basic selection, movement, deletion, and property editing
- **AND** the object list and property panel SHALL reflect the selected object

### Requirement: Annotation Confirmation Dialog
TRTForge SHALL show a confirmation dialog after a user completes drawing an annotation object.

#### Scenario: User completes a bounding box
- **WHEN** the user finishes drawing a rectangle, rotated rectangle, polygon, line, point, or keypoint object
- **THEN** the system SHALL show a confirmation dialog
- **AND** the dialog SHALL allow the user to confirm label, description, group id, dataset split, and score when applicable

#### Scenario: User confirms annotation dialog
- **WHEN** the user clicks confirm in the annotation dialog
- **THEN** the system SHALL save the annotation to the current image
- **AND** the label list, object list, and property/description panel SHALL update with the confirmed information

#### Scenario: User cancels annotation dialog
- **WHEN** the user cancels the annotation dialog
- **THEN** the system SHALL discard the newly drawn object
- **AND** the object list SHALL remain unchanged

### Requirement: Class Management
TRTForge SHALL support project-level class management.

#### Scenario: User imports TRTForge label file
- **WHEN** the user imports an existing `labels_*.json` file
- **THEN** the system SHALL load class names and colors into the annotation project
- **AND** future exports SHALL preserve the class order

#### Scenario: User edits classes manually
- **WHEN** the user adds, renames, or deletes project classes
- **THEN** the system SHALL update the project class list
- **AND** the system SHALL protect existing annotations from silently losing class meaning

### Requirement: Dataset Split Management
TRTForge SHALL support `train`, `val`, and `test` dataset split management.

#### Scenario: User applies automatic split
- **WHEN** the user configures split ratios and applies automatic split
- **THEN** the system SHALL assign each image to `train`, `val`, or `test`
- **AND** the split SHALL be reproducible with the configured random seed

#### Scenario: User manually adjusts split
- **WHEN** the user changes a single image split assignment
- **THEN** the system SHALL update that image to the selected `train`, `val`, or `test` set
- **AND** the split summary SHALL update immediately

### Requirement: YOLO Export
TRTForge SHALL export standard YOLO datasets for detection, segmentation, pose, and OBB projects.

#### Scenario: Export YOLO detection dataset
- **WHEN** the current project task is `detect` and the user exports YOLO
- **THEN** the system SHALL write YOLO detection label files
- **AND** the system SHALL copy images into `images/train`, `images/val`, and `images/test`
- **AND** the system SHALL generate `dataset.yaml`

#### Scenario: Export YOLO segmentation dataset
- **WHEN** the current project task is `segment` and the user exports YOLO
- **THEN** the system SHALL write YOLO segmentation polygon label files
- **AND** the system SHALL copy images into the standard split directories
- **AND** the system SHALL generate `dataset.yaml`

#### Scenario: Export YOLO pose dataset
- **WHEN** the current project task is `pose` and the user exports YOLO
- **THEN** the system SHALL export COCO 17 keypoint annotations in a YOLO pose compatible structure
- **AND** the system SHALL generate `dataset.yaml` with pose metadata required by Ultralytics

#### Scenario: Export YOLO OBB dataset
- **WHEN** the current project task is `obb` and the user exports YOLO
- **THEN** the system SHALL write YOLO OBB labels using the Ultralytics format `class_index x1 y1 x2 y2 x3 y3 x4 y4`
- **AND** the four OBB corner points SHALL be normalized to `[0, 1]` relative to image width and height
- **AND** the system SHALL copy images into the standard split directories
- **AND** the system SHALL generate `dataset.yaml`

### Requirement: COCO Export
TRTForge SHALL export standard COCO annotations for detection, segmentation, and pose projects.

#### Scenario: Export COCO detection dataset
- **WHEN** the current project task is `detect` and the user exports COCO
- **THEN** the system SHALL write a COCO JSON containing images, annotations, and categories
- **AND** each annotation SHALL include a standard bounding box

#### Scenario: Export COCO segmentation dataset
- **WHEN** the current project task is `segment` and the user exports COCO
- **THEN** the system SHALL write polygon segmentation entries in COCO JSON
- **AND** categories SHALL match the project class order

#### Scenario: Export COCO pose dataset
- **WHEN** the current project task is `pose` and the user exports COCO
- **THEN** the system SHALL write COCO keypoint annotations for COCO 17 keypoints
- **AND** the keypoint order SHALL follow the COCO 17 convention

#### Scenario: User requests COCO from OBB project
- **WHEN** the current project task is `obb` and the user exports COCO
- **THEN** the system SHALL show a clear unsupported-format message
- **AND** the system SHALL NOT silently generate lossy or non-standard COCO output

### Requirement: VOC Export
TRTForge SHALL export standard VOC XML only for horizontal rectangle detection projects.

#### Scenario: Export VOC detection dataset
- **WHEN** the current project task is `detect` and the user exports VOC
- **THEN** the system SHALL write standard VOC XML files for horizontal bounding boxes
- **AND** the system SHALL copy or reference images according to the export option

#### Scenario: User requests VOC from non-detection project
- **WHEN** the current project task is `segment`, `pose`, or `obb` and the user exports VOC
- **THEN** the system SHALL show that VOC supports only standard horizontal rectangle detection
- **AND** the system SHALL NOT generate non-standard VOC XML containing polygons, keypoints, or rotated boxes

### Requirement: Standard Dataset Directory Output
TRTForge SHALL export training-ready datasets with copied images and matching label files.

#### Scenario: User exports training-ready dataset
- **WHEN** the user exports a dataset
- **THEN** the system SHALL create an output directory containing split-specific image and label files
- **AND** the exported structure SHALL be usable without depending on the original scattered image locations

#### Scenario: Export has missing annotations
- **WHEN** an image has no annotation
- **THEN** the system SHALL handle it according to the selected export policy
- **AND** the export summary SHALL report empty images clearly

### Requirement: Format Import
TRTForge SHALL import supported COCO, VOC, and YOLO annotations into the annotation project model.

#### Scenario: Import YOLO annotations
- **WHEN** the user imports a supported YOLO dataset
- **THEN** the system SHALL convert labels into the project annotation model
- **AND** class ids SHALL map to the project class list

#### Scenario: Import COCO annotations
- **WHEN** the user imports a supported COCO dataset
- **THEN** the system SHALL convert images, categories, and annotations into the project annotation model

#### Scenario: Import VOC annotations
- **WHEN** the user imports VOC annotations
- **THEN** the system SHALL import standard horizontal bounding boxes into a detection project
- **AND** the system SHALL reject incompatible project task types with a clear message

#### Scenario: User imports X-AnyLabeling JSON
- **WHEN** the user attempts to import X-AnyLabeling native JSON annotations
- **THEN** the system SHALL show that X-AnyLabeling JSON import is not supported in the first version
- **AND** the system SHALL suggest converting through COCO, VOC, or YOLO when possible

### Requirement: Reload Exported Annotation Files
TRTForge SHALL support reloading annotation files exported by TRTForge so users can edit them in the UI and export them again.

#### Scenario: Reload exported YOLO dataset
- **WHEN** the user imports a YOLO dataset previously exported by TRTForge
- **THEN** the system SHALL reconstruct the project annotations in the UI
- **AND** the user SHALL be able to modify annotations and export YOLO again

#### Scenario: Reload exported COCO annotations
- **WHEN** the user imports COCO annotations previously exported by TRTForge
- **THEN** the system SHALL reconstruct supported detection, segmentation, or pose annotations
- **AND** the user SHALL be able to modify annotations and export COCO again

#### Scenario: Reload exported VOC annotations
- **WHEN** the user imports VOC annotations previously exported by TRTForge
- **THEN** the system SHALL reconstruct horizontal rectangle detection annotations
- **AND** the user SHALL be able to modify annotations and export VOC again

#### Scenario: Reloaded format lacks TRTForge-only metadata
- **WHEN** a reloaded COCO, VOC, or YOLO file lacks TRTForge-only metadata such as manual split details or descriptions
- **THEN** the system SHALL recover all information represented by the source format
- **AND** the system SHALL report any missing metadata and use clear defaults

