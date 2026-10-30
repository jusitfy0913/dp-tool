#pragma once

#include <QWidget>
#include <QJsonObject>
#include <QMap>
#include <QPoint>

#include "common/AppEvents.h"
#include "training/TrainingArtifactModel.h"

class QCheckBox;
class QComboBox;
class QGridLayout;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QTabWidget;
class QTextBrowser;

namespace trtforge {

class MonitorChartWidget;
class YoloTrainController;

class TrainingTab : public QWidget
{
    Q_OBJECT
public:
    explicit TrainingTab(QWidget* parent = nullptr);
    ~TrainingTab() override = default;

private slots:
    void onBrowsePython();
    void onBrowseModel();
    void onBrowseData();
    void onBrowseProject();
    void onBrowseAnomalibNormalDir();
    void onBrowseAnomalibAbnormalDir();
    void onBrowseAnomalibNormalTestDir();
    void onBrowseAnomalibImagenetDir();
    void onCheckEnv();
    void onTrain();
    void onValidate();
    void onExport();
    void onStop();
    void onTaskChanged();
    void onLogLine(QString text, bool isError);
    void onStructuredEvent(QJsonObject event);
    void onProcessStarted(QString program, QStringList arguments);
    void onProcessFinished(int exitCode, QString reason);
    void onProcessError(QString message);
    void onTrainTypeChanged();
    void onAnomalibModelChanged();

private:
    bool isAnomalibMode() const;
    QStringList baseYoloArgs(const QString& mode) const;
    QStringList baseAnomalibArgs() const;
    void appendLog(const QString& text,
                   const QString& color = QString(),
                   AppLogLevel level = AppLogLevel::Info);
    void loadSettings();
    void saveSettings() const;
    void savePythonPath(const QString& path) const;
    void prepareTrainInitialization();
    bool isDefaultModelAlias(const QString& model) const;
    void resetRunArtifacts(const QString& mode);
    void clearMonitorData();
    void appendEpochSummary(const QJsonObject& event);
    void updateGpuStatus(const QJsonObject& gpu);
    void updatePlotPreview(const QJsonObject& event);
    void addPlotCard(QGridLayout* grid, int row, int col, const QString& key, const QString& title);
    void updateMonitorCharts(const QJsonObject& event);
    void toggleChartZoom(const QString& key);
    void restoreChartGrid();
    void showTrainingArtifactsDialog();
    void setRunning(bool running);
    void updateInitHint();
    QWidget* buildConfigTab();
    QWidget* buildMonitorTab();
    QWidget* buildTaskCard();
    QWidget* buildParamCard();
    QWidget* buildEnvCard();
    QWidget* buildLogCard();
    void updateModeUi();
    QWidget* makeDirectoryRow(QLineEdit* edit, const QString& title, const char* slot);

    YoloTrainController* m_controller { nullptr };

    QLineEdit* m_edPython { nullptr };
    QTabWidget* m_tabs { nullptr };
    QLabel* m_titleLabel { nullptr };
    bool m_loadingSettings { false };
    QString m_currentTask;
    QString m_activeMode;
    TrainingArtifactModel m_artifactModel;
    QMap<QString, MonitorChartWidget*> m_chartWidgets;
    QMap<QString, QWidget*> m_chartCards;
    QMap<QString, QPoint> m_chartPositions;
    QGridLayout* m_chartGrid { nullptr };
    QString m_zoomedChartKey;
    QComboBox* m_cbTrainType { nullptr };
    QComboBox* m_cbTask { nullptr };
    QComboBox* m_cbInitMode { nullptr };
    QWidget* m_yoloTaskBox { nullptr };
    QWidget* m_anomalibTaskBox { nullptr };
    QWidget* m_yoloParamBox { nullptr };
    QWidget* m_anomalibParamBox { nullptr };
    QLineEdit* m_edModel { nullptr };
    QLineEdit* m_edData { nullptr };
    QLineEdit* m_edProject { nullptr };
    QLineEdit* m_edName { nullptr };
    QLineEdit* m_edDevice { nullptr };
    QLineEdit* m_edBatch { nullptr };
    QLineEdit* m_edPretrained { nullptr };
    QComboBox* m_cbAnomalibModel { nullptr };
    QLineEdit* m_edAnomNormalDir { nullptr };
    QLineEdit* m_edAnomAbnormalDir { nullptr };
    QLineEdit* m_edAnomNormalTestDir { nullptr };
    QLineEdit* m_edAnomImagenetDir { nullptr };
    QLineEdit* m_edAnomMaxEpochs { nullptr };
    QLineEdit* m_edAnomTrainBatch { nullptr };
    QLineEdit* m_edAnomEvalBatch { nullptr };
    QLineEdit* m_edAnomCalibNormalDir { nullptr };
    QLineEdit* m_edAnomCalibAbnormalDir { nullptr };
    QSpinBox* m_spEpochs { nullptr };
    QSpinBox* m_spImgsz { nullptr };
    QSpinBox* m_spAnomInputSize { nullptr };
    QSpinBox* m_spAnomWorkers { nullptr };
    QSpinBox* m_spAnomCalibLimit { nullptr };
    QSpinBox* m_spPatience { nullptr };
    QSpinBox* m_spWorkers { nullptr };
    QCheckBox* m_chkHalf { nullptr };
    QCheckBox* m_chkDynamic { nullptr };
    QCheckBox* m_chkMakeTransd { nullptr };
    QCheckBox* m_chkAnomCalibrate { nullptr };
    QLabel* m_lblEnv { nullptr };
    QLabel* m_lblMetric { nullptr };
    QLabel* m_lblArtifact { nullptr };
    QLabel* m_lblInitHint { nullptr };
    QLabel* m_lblGpu { nullptr };
    QLabel* m_lblLabelsJson { nullptr };
    QProgressBar* m_barGpuMemory { nullptr };
    QProgressBar* m_barGpuUtil { nullptr };
    QTableWidget* m_metricTable { nullptr };
    QTextBrowser* m_log { nullptr };
    QPushButton* m_btnCheck { nullptr };
    QPushButton* m_btnTrain { nullptr };
    QPushButton* m_btnVal { nullptr };
    QPushButton* m_btnExport { nullptr };
    QPushButton* m_btnStop { nullptr };
};

} // namespace trtforge
