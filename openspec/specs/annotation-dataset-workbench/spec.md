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

