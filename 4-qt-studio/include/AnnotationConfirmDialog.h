#pragma once

#include "AnnotationProject.h"

#include <QDialog>

class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QPlainTextEdit;
class QSpinBox;

namespace trtforge {

class AnnotationConfirmDialog : public QDialog
{
    Q_OBJECT
public:
    explicit AnnotationConfirmDialog(const AnnotationProject& project,
                                     const AnnotationObject& draft,
                                     DatasetSplit currentSplit,
                                     QWidget* parent = nullptr);

    AnnotationObject annotation() const;
    DatasetSplit split() const;

private:
    AnnotationObject m_annotation;
    QComboBox* m_cbClass = nullptr;
    QPlainTextEdit* m_edDescription = nullptr;
    QSpinBox* m_spGroup = nullptr;
    QComboBox* m_cbSplit = nullptr;
    QDoubleSpinBox* m_spScore = nullptr;
};

} // namespace trtforge

