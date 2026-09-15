# Change: Add Custom Dataset Split Ratio

## Why
TRTForge annotation workbench currently auto-splits images with a fixed 8/1/1 ratio. Users need to set project-specific train/val/test ratios without manually changing every image.

## What Changes
- Add user-configurable train/val/test percentage controls to the annotation dataset split panel.
- Make the existing auto split action use the configured percentages instead of the fixed 8/1/1 ratio.
- Persist the last selected ratio with Qt settings so the preferred default is restored on next launch.

## Impact
- Affected specs: annotation-dataset-workbench
- Affected code: `trtexec/gui/src/annotation/AnnotationTab.cpp`, `trtexec/gui/include/AnnotationTab.h`
