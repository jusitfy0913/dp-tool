#pragma once

#include "AnnotationCanvas.h"
#include "AnnotationProject.h"

#include <QWidget>

class QComboBox;
class QLabel;
class QListWidget;
class QPushButton;
class QSpinBox;
class QTextEdit;

namespace trtforge {

class AnnotationTab : public QWidget
{
    Q_OBJECT
public:
    explicit AnnotationTab(QWidget* parent = nullptr);

private slots:
    void onNewProject();
    void onOpenProject();
    void onSaveProject();
    void onSaveProjectAs();
    void onImportImages();
    void onImportLabels();
    void onImportAnnotations();
    void onExportYolo();
    void onExportCoco();
    void onExportVoc();
    void onAutoSplit();
    void onEditSelectedObject();
    void onDeleteSelectedObject();
    void onSetCurrentSplitTrain();
    void onSetCurrentSplitVal();
    void onSetCurrentSplitTest();
    void onSplitRatioChanged();
    void onUndo();
    void onCanvasObjectSelected(int row);
    void onCanvasAnnotationsEdited();
    void onCanvasEditStarted();
    void onImageChanged(int row);
    void onDraftCompleted(trtforge::AnnotationObject draft);
    void onToolClicked();

private:
    QWidget* buildToolbar();
    QWidget* buildLeftTools();
    QWidget* buildRightPanel();
    QWidget* buildBottomPanel();

    void refreshAll();
    void refreshClassList();
    void refreshImageList();
    void refreshObjectList();
    void refreshPropertyPanel();
    void loadCurrentImage();
    void setToolButton(QPushButton* button, AnnotationTool tool);
    AnnotationImage* currentImage();
    const AnnotationImage* currentImage() const;
    void showIoResult(const QString& title, bool ok, const QString& message);
    void pushUndoForCurrentImage();
    void loadSplitRatioSettings();
    void saveSplitRatioSettings() const;
    int splitTrainPct() const;
    int splitValPct() const;
    int splitTestPct() const;
    QString splitRatioText() const;

    AnnotationProject m_project;
    int m_currentImage = -1;
    int m_selectedObject = -1;

    struct UndoState {
        int imageIndex = -1;
        QVector<AnnotationObject> annotations;
    };
    QVector<UndoState> m_undoStack;

    AnnotationCanvas* m_canvas = nullptr;
    QComboBox* m_cbTask = nullptr;
    QListWidget* m_classList = nullptr;
    QListWidget* m_objectList = nullptr;
    QListWidget* m_imageList = nullptr;
    QLabel* m_projectInfo = nullptr;
    QLabel* m_splitInfo = nullptr;
    QLabel* m_splitRatioHint = nullptr;
    QSpinBox* m_spSplitTrain = nullptr;
    QSpinBox* m_spSplitVal = nullptr;
    QSpinBox* m_spSplitTest = nullptr;
    QTextEdit* m_propertyText = nullptr;
    QVector<QPushButton*> m_toolButtons;
    QPushButton* m_btnUndo = nullptr;
};

} // namespace trtforge

