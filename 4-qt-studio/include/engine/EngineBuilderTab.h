#pragma once

#include <QThread>
#include <QWidget>

#include "BuildOptions.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProgressBar;
class QProgressDialog;
class QPushButton;
class QTextBrowser;
class QSpinBox;
class QTimer;

namespace trtforge {

class BuilderWorker;
struct OnnxInputInfo;
class TRTProgressMonitor;

class EngineBuilderTab : public QWidget
{
    Q_OBJECT

public:
    explicit EngineBuilderTab(QWidget* parent = nullptr);
    ~EngineBuilderTab() override;

private slots:
    void onBrowseOnnx();
    void onSaveAs();
    void onResetClicked();
    void onBuildClicked();
    void onClearLog();
    void onTestExistingClicked();

    void onLog(int level, QString text);
    void onFinished(bool ok, QString enginePath, qint64 elapsedMs);
    void onTestFinished(trtforge::TestReport report);
    void onProgress(int permille);
    void onStage(int stage);
    void onEta(qint64 etaMs);
    void onElapsedTick();

private:
    QWidget* buildIoCard();
    QWidget* buildPrecisionCard();
    QWidget* buildBuildParamsCard();
    QWidget* buildShapesCard();
    QWidget* buildExtraCard();
    QWidget* buildTestCard();
    QWidget* buildActionRow();
    QWidget* buildConsoleCard();
    QWidget* buildProgressCard();
    QWidget* buildTestReportCard();

    void collectOptions(BuildOptions& opt) const;
    void applyOptions(const BuildOptions& opt);
    void applyOnnxInputInfo(const trtforge::OnnxInputInfo& input);
    void startOnnxInputInspection(const QString& onnxPath);
    void finishOnnxInputInspection(int requestId, const QString& onnxPath, const trtforge::OnnxInputInfo& input);
    void cancelOnnxInputInspection();
    void showOnnxLoadingProgress(const QString& onnxPath);
    void hideOnnxLoadingProgress();
    void setOnnxInspectUi(bool inspecting);
    void syncShapesFromHw();
    QString shapeWithHw(const QString& currentShape, int fallbackBatch) const;
    void setBuildingUi(bool building);
    void appendColored(int level, const QString& text);
    QString formatHms(qint64 ms) const;

    QLineEdit* m_edOnnx { nullptr };
    QLineEdit* m_edEngine { nullptr };
    QComboBox* m_cbPrec { nullptr };
    QComboBox* m_cbRuntime { nullptr };
    QSpinBox* m_spWorkspace { nullptr };
    QSpinBox* m_spMaxBatch { nullptr };
    QSpinBox* m_spDla { nullptr };
    QSpinBox* m_spDevice { nullptr };
    QCheckBox* m_chkSparsity { nullptr };
    QCheckBox* m_chkRefit { nullptr };
    QCheckBox* m_chkVerCompat { nullptr };
    QCheckBox* m_chkGpuFb { nullptr };
    QCheckBox* m_chkTf32 { nullptr };
    QLineEdit* m_edInputName { nullptr };
    QLineEdit* m_edMin { nullptr };
    QLineEdit* m_edOpt { nullptr };
    QLineEdit* m_edMax { nullptr };
    QSpinBox* m_spShapeH { nullptr };
    QSpinBox* m_spShapeW { nullptr };
    QCheckBox* m_chkSyncHw { nullptr };
    QPlainTextEdit* m_edExtra { nullptr };

    QPushButton* m_btnBrowse { nullptr };
    QPushButton* m_btnSaveAs { nullptr };
    QPushButton* m_btnReset { nullptr };
    QPushButton* m_btnBuild { nullptr };
    QPushButton* m_btnClear { nullptr };
    QPushButton* m_btnTestExist { nullptr };

    QCheckBox* m_chkTestAuto { nullptr };
    QSpinBox* m_spTestWarmup { nullptr };
    QSpinBox* m_spTestIter { nullptr };
    QLabel* m_lblTestSummary { nullptr };
    QPlainTextEdit* m_edTestDetail { nullptr };

    QTextBrowser* m_logView { nullptr };
    QProgressBar* m_progress { nullptr };
    QLabel* m_lblPercent { nullptr };
    QLabel* m_lblStageNote { nullptr };
    QLabel* m_lblElapsed { nullptr };
    QLabel* m_lblEta { nullptr };
    QLabel* m_lblLayer { nullptr };
    QLabel* m_lblVram { nullptr };

    QLabel* m_stageDots[5] { nullptr, nullptr, nullptr, nullptr, nullptr };
    QLabel* m_stageTexts[5] { nullptr, nullptr, nullptr, nullptr, nullptr };

    QThread* m_thread { nullptr };
    BuilderWorker* m_worker { nullptr };
    QThread* m_onnxInspectThread { nullptr };
    QProgressDialog* m_onnxLoadingProgress { nullptr };
    int m_onnxInspectRequestId { 0 };
    bool m_onnxInspecting { false };
    TRTProgressMonitor* m_monitor { nullptr };
    QTimer* m_uiTimer { nullptr };
    qint64 m_startMs { 0 };
    qint64 m_etaMs { -1 };
    int m_layerCount { 0 };
};

} // namespace trtforge
