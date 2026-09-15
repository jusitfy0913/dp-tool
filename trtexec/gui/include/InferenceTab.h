#pragma once

#include <QWidget>
#include <QPointer>
#include <QThread>

#include "AnomalyInferWorker.h"
#include "InferWorker.h"
#include "LabelStore.h"
#include "OcrInferWorker.h"

class QLineEdit;
class QComboBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;
class QCheckBox;
class QListWidget;
class QButtonGroup;
class QStackedWidget;
class QHBoxLayout;
class QProgressDialog;
class QSplitter;

namespace trtforge {

class CaptureWorker;

class InferenceTab : public QWidget
{
    Q_OBJECT
public:
    explicit InferenceTab(QWidget* parent = nullptr);
    ~InferenceTab() override;

private slots:
    void onBrowseEngine();
    void onBrowseOcrRecModel();
    void onBrowseOcrLineOrientationModel();
    void onBrowseLabel();
    void onBrowseMetadata();
    void onReloadLabel();
    void onReloadMetadata();
    void onTaskChanged(int index);
    void onSourceChanged(int id);
    void onConnectCamera();
    void onScanCameras();
    void onBrowseImageOrVideo();
    void previewLocalFile(const QString& path);
    void onStartClicked();
    void onStopClicked();

    void onFrameReady(QImage frame, qint64 ts);
    void onResultReady(QImage frame, QVector<Detection> dets,
                       double preMs, double infMs, double postMs, double totalMs);
    void onAnomalyResultReady(AnomalyResult result,
                              double preMs, double infMs, double postMs, double totalMs);
    void onOcrResultReady(OcrResult result,
                          double preMs, double infMs, double postMs, double totalMs);
    void onAnomalyMetadataReady(AnomalyMetadata meta);
    void onCaptureError(QString msg);
    void onInferError(QString msg);
    void onEngineLoaded();
    void onEngineLoadFailed(QString msg);
    void onCaptureFinished();
    void onSourceInfo(int w, int h, double fps);

private:
    QWidget* buildEngineCard();
    QWidget* buildSourceCard();
    QWidget* buildLabelCard();
    QWidget* buildActionRow();
    QWidget* buildStatsBar();
    QWidget* buildCanvas();
    QWidget* buildDetList();

    void paintOverlay(QImage& img, const QVector<Detection>& dets);
    void updateDetList(const QVector<Detection>& dets);
    void updateAnomalyList(const AnomalyResult& result);
    void updateOcrList(const OcrResult& result);
    void refreshOcrCanvasHighlight();
    bool isAnomalyMode() const;
    bool isOcrMode() const;
    void updateTaskUi();
    void setRunningUi(bool running);
    void showLoadingProgress();
    void hideLoadingProgress();
    void startCaptureAfterEngineLoaded();
    void stopCaptureThread();
    void releaseInferenceWorker();
    void cleanupThreads();
    QString currentModelKey() const;
    bool hasReusableWorker(const QString& modelKey) const;

    // 左列
    QLineEdit*    m_edEngine     = nullptr;
    QLabel*       m_lblEnginePathName = nullptr;
    QComboBox*    m_cbTask       = nullptr;
    QLabel*       m_lblOcrRecModelName = nullptr;
    QWidget*      m_ocrRecModelPanel = nullptr;
    QLineEdit*    m_edOcrRecModel = nullptr;
    QPushButton*  m_btnBrowseOcrRecModel = nullptr;
    QLabel*       m_lblOcrLineOrientationModelName = nullptr;
    QWidget*      m_ocrLineOrientationModelPanel = nullptr;
    QLineEdit*    m_edOcrLineOrientationModel = nullptr;
    QPushButton*  m_btnBrowseOcrLineOrientationModel = nullptr;
    QWidget*      m_ocrRuntimePanel = nullptr;
    QComboBox*    m_cbOcrDevice = nullptr;
    QComboBox*    m_cbOcrPrecision = nullptr;
    QCheckBox*    m_chkOcrMkldnn = nullptr;
    QSpinBox*     m_spOcrThreads = nullptr;
    QSpinBox*     m_spOcrBatch = nullptr;
    QComboBox*    m_cbAnomalyModel = nullptr;
    QLabel*       m_lblAnomalyModelName = nullptr;
    QLabel*       m_lblConfName  = nullptr;
    QSlider*      m_slConf       = nullptr;
    QLabel*       m_lblConfVal   = nullptr;
    QLabel*       m_lblNmsName   = nullptr;
    QSlider*      m_slNms        = nullptr;
    QLabel*       m_lblNmsVal    = nullptr;

    QButtonGroup* m_srcGroup     = nullptr;
    QPushButton*  m_btnSrcImg    = nullptr;
    QPushButton*  m_btnSrcVid    = nullptr;
    QPushButton*  m_btnSrcCam    = nullptr;
    QWidget*      m_camPanel     = nullptr;
    QWidget*      m_filePanel    = nullptr;
    QComboBox*    m_cbCamera     = nullptr;
    QPushButton*  m_btnScanCam   = nullptr;
    QPushButton*  m_btnConnect   = nullptr;
    QLabel*       m_lblCamStatus = nullptr;
    QLineEdit*    m_edFilePath   = nullptr;
    QPushButton*  m_btnBrowseFile= nullptr;

    QLineEdit*    m_edLabel      = nullptr;
    QLabel*       m_lblLabelInfo = nullptr;
    QWidget*      m_labelPanel   = nullptr;
    QWidget*      m_metadataPanel = nullptr;
    QLineEdit*    m_edMetadata   = nullptr;
    QLabel*       m_lblMetadataInfo = nullptr;

    QPushButton*  m_btnStart     = nullptr;
    QPushButton*  m_btnStop      = nullptr;

    // 右列
    QLabel*       m_lblFps       = nullptr;
    QLabel*       m_lblPreMs     = nullptr;
    QLabel*       m_lblInfMs     = nullptr;
    QLabel*       m_lblPostMs    = nullptr;
    QLabel*       m_lblDetCount  = nullptr;
    QLabel*       m_lblCountName = nullptr;
    QLabel*       m_lblResInfo   = nullptr;
    QLabel*       m_lblCanvas    = nullptr;
    QLabel*       m_lblLive      = nullptr;
    QListWidget*  m_detListW     = nullptr;
    QSplitter*    m_resultSplitter = nullptr;

    // 异常检测三画布
    QStackedWidget* m_canvasStack        = nullptr;
    QWidget*        m_singleCanvasPage   = nullptr;
    QWidget*        m_dualCanvasPage     = nullptr;
    QLabel*         m_lblImageCanvas     = nullptr;
    QLabel*         m_lblOverlayCanvas   = nullptr;
    QLabel*         m_lblMaskCanvas      = nullptr;
    QWidget*        m_imageFrame         = nullptr;
    QWidget*        m_overlayFrame       = nullptr;
    QWidget*        m_maskFrame          = nullptr;
    int             m_anomalyFullscreenIdx = -1;  // -1=triple, 0=image, 1=overlay, 2=mask fullscreen

    void toggleAnomalyFullscreen(int idx);

    // 业务
    LabelStore*    m_labels       = nullptr;
    QThread*       m_capThread    = nullptr;
    CaptureWorker* m_capWorker    = nullptr;
    QThread*       m_infThread    = nullptr;
    InferWorker*   m_infWorker    = nullptr;
    AnomalyInferWorker* m_anomalyWorker = nullptr;
    OcrInferWorker*     m_ocrWorker = nullptr;
    AnomalyMetadata     m_anomalyMeta;
    bool                m_anomalyMetaLoaded = false;
    QProgressDialog*    m_loadingProgress = nullptr;
    bool                m_loadingEngine = false;
    QString             m_loadedModelKey;
    QString             m_pendingModelKey;
    bool                m_inferenceActive = false;
    OcrResult           m_lastOcrResult;
    int                 m_selectedOcrLineRow = -1;
    int                 m_pendingSrcId = -1;
    QString             m_pendingSourcePath;
    int                 m_pendingCamBackend = 0;

    bool m_cameraConnected = false;
    int  m_srcW = 0, m_srcH = 0;
    double m_srcFps = 0;
    int  m_frameCount = 0;
    qint64 m_fpsTimestamp = 0;
    double m_currentFps = 0;
};

} // namespace trtforge
