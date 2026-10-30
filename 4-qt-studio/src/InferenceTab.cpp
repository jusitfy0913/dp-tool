#include "InferenceTab.h"
#include "CaptureWorker.h"
#include "common/AppEvents.h"
#include "common/AppFileLogger.h"
#include "inference/AnomalyInferenceAdapter.h"
#include "inference/InferenceResultPresenter.h"
#include "inference/OcrInferenceAdapter.h"
#include "inference/YoloInferenceAdapter.h"

#include <QApplication>
#include <QAbstractItemView>
#include <QCoreApplication>
#include <QEventLoop>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QIODevice>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPolygonF>
#include <cmath>
#include <QProgressDialog>
#include <QPushButton>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <functional>
#include <vector>

namespace trtforge {

namespace {
class ZoomableCanvasLabel : public QLabel
{
public:
    explicit ZoomableCanvasLabel(const QString& text, QWidget* parent = nullptr)
        : QLabel(text, parent)
    {
        setAlignment(Qt::AlignCenter);
        setMouseTracking(true);
    }

    void setImage(const QImage& image, bool resetView = false)
    {
        if (image.isNull()) {
            m_image = QImage();
            setText(QStringLiteral("等待输入..."));
            update();
            return;
        }
        m_image = image;
        setText({});
        if (resetView || !m_hasView) {
            m_zoom = 1.0;
            m_pan = QPointF(0, 0);
            m_hasView = true;
        }
        clampPan();
        update();
    }

    void clearCanvas(const QString& text)
    {
        m_image = QImage();
        m_zoom = 1.0;
        m_pan = QPointF(0, 0);
        m_dragging = false;
        m_hasView = false;
        setText(text);
        update();
    }

    void setTitle(const QString& title) { m_title = title; update(); }

    std::function<void()> onDoubleClick;

protected:
    void paintEvent(QPaintEvent* event) override
    {
        if (m_image.isNull()) {
            QLabel::paintEvent(event);
            return;
        }

        QPainter painter(this);
        painter.setRenderHint(QPainter::SmoothPixmapTransform, m_zoom < 3.0);
        painter.fillRect(rect(), QColor("#020407"));

        const QSizeF fitSize = m_image.size().scaled(size(), Qt::KeepAspectRatio);
        const QSizeF drawSize = fitSize * m_zoom;
        const QPointF topLeft((width() - drawSize.width()) / 2.0 + m_pan.x(),
                              (height() - drawSize.height()) / 2.0 + m_pan.y());
        painter.drawImage(QRectF(topLeft, drawSize), m_image);

        auto drawBadge = [&](const QRect& box, const QString& text, const QColor& textColor) {
            if (text.isEmpty()) {
                return;
            }
            QFontMetrics fm(painter.font());
            const QRect textRect = fm.boundingRect(text);
            QRect badge(box.topLeft(), QSize(textRect.width() + 16, textRect.height() + 10));
            if (box.width() < badge.width()) {
                badge.setWidth(box.width());
            }
            painter.setPen(QPen(QColor("#16334D"), 1));
            painter.setBrush(QColor(2, 4, 7, 205));
            painter.drawRoundedRect(badge, 4, 4);
            painter.setPen(textColor);
            painter.drawText(badge.adjusted(8, 5, -8, -5), Qt::AlignLeft | Qt::AlignVCenter, text);
        };

        if (!m_title.isEmpty()) {
            QFont tf = painter.font(); tf.setBold(true); tf.setPointSize(10);
            painter.setFont(tf);
            drawBadge(rect().adjusted(10, 8, -10, -8), m_title, QColor("#1FB6FF"));
        }

        QFont infoFont = painter.font();
        infoFont.setBold(false);
        infoFont.setPointSize(9);
        painter.setFont(infoFont);
        const QString info = QStringLiteral("滚轮缩放 %1% · 左键拖动 · 双击切换").arg(qRound(m_zoom * 100));
        QFontMetrics fm(infoFont);
        const QRect infoTextRect = fm.boundingRect(info);
        QRect infoBadge(QPoint(width() - infoTextRect.width() - 26, height() - infoTextRect.height() - 18),
                        QSize(infoTextRect.width() + 16, infoTextRect.height() + 10));
        infoBadge = infoBadge.intersected(rect().adjusted(10, 8, -10, -8));
        painter.setPen(QPen(QColor("#16334D"), 1));
        painter.setBrush(QColor(2, 4, 7, 205));
        painter.drawRoundedRect(infoBadge, 4, 4);
        painter.setPen(QColor("#9FC8E8"));
        painter.drawText(infoBadge.adjusted(8, 5, -8, -5), Qt::AlignRight | Qt::AlignVCenter, info);
    }

    void wheelEvent(QWheelEvent* event) override
    {
        if (m_image.isNull()) {
            QLabel::wheelEvent(event);
            return;
        }

        const double oldZoom = m_zoom;
        const double factor = event->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
        m_zoom = qBound(0.2, m_zoom * factor, 12.0);

        const QPointF cursor = event->position();
        const QPointF center(width() / 2.0, height() / 2.0);
        m_pan = (m_pan + center - cursor) * (m_zoom / oldZoom) - center + cursor;
        clampPan();
        update();
        event->accept();
    }

    void mousePressEvent(QMouseEvent* event) override
    {
        if (!m_image.isNull() && event->button() == Qt::LeftButton) {
            m_dragging = true;
            m_lastMousePos = event->position();
            setCursor(Qt::ClosedHandCursor);
            event->accept();
            return;
        }
        QLabel::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event) override
    {
        if (m_dragging) {
            const QPointF delta = event->position() - m_lastMousePos;
            m_lastMousePos = event->position();
            m_pan += delta;
            clampPan();
            update();
            event->accept();
            return;
        }
        QLabel::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent* event) override
    {
        if (m_dragging && event->button() == Qt::LeftButton) {
            m_dragging = false;
            setCursor(Qt::ArrowCursor);
            event->accept();
            return;
        }
        QLabel::mouseReleaseEvent(event);
    }

    void mouseDoubleClickEvent(QMouseEvent* event) override
    {
        if (!m_image.isNull() && event->button() == Qt::LeftButton && onDoubleClick) {
            onDoubleClick();
            event->accept();
            return;
        }
        if (!m_image.isNull() && event->button() == Qt::LeftButton) {
            m_zoom = 1.0;
            m_pan = QPointF(0, 0);
            clampPan();
            update();
            event->accept();
            return;
        }
        QLabel::mouseDoubleClickEvent(event);
    }

    void resizeEvent(QResizeEvent* event) override
    {
        QLabel::resizeEvent(event);
        clampPan();
    }

private:
    void clampPan()
    {
        if (m_image.isNull() || width() <= 0 || height() <= 0) return;

        const QSizeF fitSize = m_image.size().scaled(size(), Qt::KeepAspectRatio);
        const QSizeF drawSize = fitSize * m_zoom;
        const double maxX = qMax(0.0, (drawSize.width() - width()) / 2.0);
        const double maxY = qMax(0.0, (drawSize.height() - height()) / 2.0);
        m_pan.setX(qBound(-maxX, m_pan.x(), maxX));
        m_pan.setY(qBound(-maxY, m_pan.y(), maxY));
    }

    QImage m_image;
    double m_zoom { 1.0 };
    QPointF m_pan;
    QPointF m_lastMousePos;
    bool m_dragging { false };
    bool m_hasView { false };
    QString m_title;
};

bool containsNonAsciiPath(const QString& text)
{
    for (const QChar ch : text) {
        if (ch.unicode() > 127) return true;
    }
    return false;
}

QString makeAsciiVideoPreviewCopy(const QString& source)
{
    const QFileInfo info(source);
    const QString suffix = info.suffix().isEmpty() ? QStringLiteral("video") : info.suffix();
    const QString target = QDir::temp().absoluteFilePath(
        QStringLiteral("trtforge_preview_%1.%2").arg(QDateTime::currentMSecsSinceEpoch()).arg(suffix));
    QFile::remove(target);
    return QFile::copy(source, target) ? target : QString();
}

bool openVideoPreviewUnicode(cv::VideoCapture& cap, const QString& path, QString* tempCopyPath)
{
    if (!containsNonAsciiPath(path)) {
        cap.open(path.toStdString());
        if (cap.isOpened()) return true;
    }

    const QString copied = makeAsciiVideoPreviewCopy(path);
    if (copied.isEmpty()) return false;

    cap.open(copied.toStdString());
    if (!cap.isOpened()) {
        QFile::remove(copied);
        return false;
    }
    if (tempCopyPath) *tempCopyPath = copied;
    return true;
}

// COCO 17 关键点骨架定义，与 5-reference/YOLOv8-TensorRT-main/csrc/pose/normal/main.cpp 完全一致：
// 19 段连接（KPS_COLORS[17] 中前 17 个对应 17 个关键点；SKELETON/LIMB_COLORS 共 19 段）。
static const unsigned int KPS_COLORS[17][3] = {
    {0,255,0},{0,255,0},{0,255,0},{0,255,0},{0,255,0},
    {255,128,0},{255,128,0},{255,128,0},{255,128,0},{255,128,0},{255,128,0},
    {51,153,255},{51,153,255},{51,153,255},{51,153,255},{51,153,255},{51,153,255}
};
static const unsigned int SKELETON[19][2] = {
    {16,14},{14,12},{17,15},{15,13},{12,13},{6,12},{7,13},{6,7},
    {6,8},{7,9},{8,10},{9,11},{2,3},{1,2},{1,3},{2,4},{3,5},{4,6},{5,7}
};
static const unsigned int LIMB_COLORS[19][3] = {
    {51,153,255},{51,153,255},{51,153,255},{51,153,255},
    {255,51,255},{255,51,255},{255,51,255},
    {255,128,0},{255,128,0},{255,128,0},{255,128,0},{255,128,0},
    {0,255,0},{0,255,0},{0,255,0},{0,255,0},{0,255,0},{0,255,0},{0,255,0}
};
static constexpr float KPT_VIS_THRESHOLD = 0.5f;

QFrame* makeCard()
{
    auto* f = new QFrame();
    f->setObjectName("card");
    f->setFrameShape(QFrame::NoFrame);
    return f;
}
QLabel* makeCardTitle(const QString& t)
{
    auto* l = new QLabel(t);
    l->setObjectName("cardTitle");
    return l;
}
QLabel* makeFieldLabel(const QString& t)
{
    auto* l = new QLabel(t);
    l->setObjectName("fieldLabel");
    return l;
}
} // namespace

InferenceTab::InferenceTab(QWidget* parent) : QWidget(parent)
{
    qRegisterMetaType<QVector<trtforge::Detection>>("QVector<trtforge::Detection>");
    qRegisterMetaType<QImage>("QImage");
    qRegisterMetaType<trtforge::AnomalyMetadata>("trtforge::AnomalyMetadata");
    qRegisterMetaType<trtforge::AnomalyResult>("trtforge::AnomalyResult");

    m_labels = new LabelStore(this);

    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(16);

    auto* leftCol = new QWidget(this);
    leftCol->setFixedWidth(380);
    auto* leftV = new QVBoxLayout(leftCol);
    leftV->setContentsMargins(0, 0, 0, 0);
    leftV->setSpacing(14);
    leftV->addWidget(buildEngineCard());
    leftV->addWidget(buildSourceCard());
    leftV->addWidget(buildLabelCard());
    leftV->addStretch(1);
    leftV->addWidget(buildActionRow());
    root->addWidget(leftCol, 0);

    auto* rightCol = new QWidget(this);
    auto* rightV = new QVBoxLayout(rightCol);
    rightV->setContentsMargins(0, 0, 0, 0);
    rightV->setSpacing(12);
    rightV->addWidget(buildStatsBar(), 0);

    m_resultSplitter = new QSplitter(Qt::Horizontal, rightCol);
    m_resultSplitter->setChildrenCollapsible(false);
    m_resultSplitter->setHandleWidth(8);
    m_resultSplitter->addWidget(buildCanvas());
    m_resultSplitter->addWidget(buildDetList());
    m_resultSplitter->setStretchFactor(0, 1);
    m_resultSplitter->setStretchFactor(1, 0);
    m_resultSplitter->setSizes({900, 280});
    rightV->addWidget(m_resultSplitter, 1);

    root->addWidget(rightCol, 1);
    connect(m_detListW, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem* current, QListWidgetItem*) {
                if (!isOcrMode()) {
                    return;
                }
                const int row = current ? current->data(Qt::UserRole).toInt() : -1;
                m_selectedOcrLineRow = row;
                refreshOcrCanvasHighlight();
            });

    QString defaultLabels = QDir(qApp->applicationDirPath())
        .filePath(QStringLiteral("../configs/labels_coco.json"));
    if (!QFile::exists(defaultLabels))
        defaultLabels = QDir(qApp->applicationDirPath()).filePath("configs/labels_coco.json");
    if (m_labels->load(defaultLabels)) {
        m_edLabel->setText(QDir::toNativeSeparators(defaultLabels));
        m_lblLabelInfo->setText(QStringLiteral("已加载 %1 个类别 · %2")
            .arg(m_labels->count()).arg(m_labels->datasetName()));
    } else {
        m_lblLabelInfo->setText(QStringLiteral("⚠ 默认标签未找到，请手动指定"));
    }

    updateTaskUi();
    setRunningUi(false);
}

InferenceTab::~InferenceTab()
{
    cleanupThreads();
}

QWidget* InferenceTab::buildEngineCard()
{
    auto* c = makeCard();
    auto* v = new QVBoxLayout(c);
    v->setContentsMargins(14, 12, 14, 14);
    v->setSpacing(8);

    v->addWidget(makeCardTitle("引擎与任务"));

    m_lblEnginePathName = makeFieldLabel(".trt 引擎路径");
    v->addWidget(m_lblEnginePathName);
    auto* row = new QHBoxLayout(); row->setSpacing(6);
    m_edEngine = new QLineEdit();
    m_edEngine->setPlaceholderText("选择 .trt 引擎文件");
    auto* btnBrowse = new QPushButton("浏览");
    btnBrowse->setProperty("ghost", true);
    row->addWidget(m_edEngine, 1);
    row->addWidget(btnBrowse, 0);
    v->addLayout(row);
    connect(btnBrowse, &QPushButton::clicked, this, &InferenceTab::onBrowseEngine);

    v->addWidget(makeFieldLabel("任务类型"));
    m_cbTask = new QComboBox();
    m_cbTask->addItem("目标检测 DET", static_cast<int>(YI_DETECT));
    m_cbTask->addItem("实例分割 SEG", static_cast<int>(YI_SEGMENT));
    m_cbTask->addItem("姿态估计 POSE", static_cast<int>(YI_POSE));
    m_cbTask->addItem("旋转检测 OBB",  static_cast<int>(YI_OBB));
    m_cbTask->addItem("异常检测 ANOMALY", -100);
    m_cbTask->addItem("OCR 文本识别 OCR", -200);
    v->addWidget(m_cbTask);
    connect(m_cbTask, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InferenceTab::onTaskChanged);

    m_lblAnomalyModelName = makeFieldLabel("异常检测模型");
    v->addWidget(m_lblAnomalyModelName);
    m_cbAnomalyModel = new QComboBox();
    m_cbAnomalyModel->addItem("EfficientAd");
    m_cbAnomalyModel->addItem("PaDiM");
    m_cbAnomalyModel->addItem("PatchCore");
    m_cbAnomalyModel->setVisible(false);
    v->addWidget(m_cbAnomalyModel);
    connect(m_cbAnomalyModel, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this]() {
                if (m_edMetadata && !m_edMetadata->text().isEmpty()) onReloadMetadata();
            });

    m_lblOcrRecModelName = makeFieldLabel("OCR 识别模型目录");
    v->addWidget(m_lblOcrRecModelName);
    m_ocrRecModelPanel = new QWidget();
    auto* ocrRecRow = new QHBoxLayout(m_ocrRecModelPanel);
    ocrRecRow->setContentsMargins(0, 0, 0, 0);
    ocrRecRow->setSpacing(6);
    m_edOcrRecModel = new QLineEdit();
    m_edOcrRecModel->setPlaceholderText("选择 ch_PP-OCRv4_rec_infer 模型目录");
    m_btnBrowseOcrRecModel = new QPushButton("浏览");
    m_btnBrowseOcrRecModel->setProperty("ghost", true);
    ocrRecRow->addWidget(m_edOcrRecModel, 1);
    ocrRecRow->addWidget(m_btnBrowseOcrRecModel, 0);
    v->addWidget(m_ocrRecModelPanel);
    connect(m_btnBrowseOcrRecModel, &QPushButton::clicked,
            this, &InferenceTab::onBrowseOcrRecModel);

    m_lblOcrLineOrientationModelName = makeFieldLabel("OCR 文本行方向模型目录");
    v->addWidget(m_lblOcrLineOrientationModelName);
    m_ocrLineOrientationModelPanel = new QWidget();
    auto* ocrLineOrientationRow = new QHBoxLayout(m_ocrLineOrientationModelPanel);
    ocrLineOrientationRow->setContentsMargins(0, 0, 0, 0);
    ocrLineOrientationRow->setSpacing(6);
    m_edOcrLineOrientationModel = new QLineEdit();
    m_edOcrLineOrientationModel->setPlaceholderText("可选：选择 ch_ppocr_mobile_v2.0_cls_slim_infer 模型目录");
    m_btnBrowseOcrLineOrientationModel = new QPushButton("浏览");
    m_btnBrowseOcrLineOrientationModel->setProperty("ghost", true);
    ocrLineOrientationRow->addWidget(m_edOcrLineOrientationModel, 1);
    ocrLineOrientationRow->addWidget(m_btnBrowseOcrLineOrientationModel, 0);
    v->addWidget(m_ocrLineOrientationModelPanel);
    connect(m_btnBrowseOcrLineOrientationModel, &QPushButton::clicked,
            this, &InferenceTab::onBrowseOcrLineOrientationModel);

    m_ocrRuntimePanel = new QWidget();
    auto* ocrRuntimeGrid = new QGridLayout(m_ocrRuntimePanel);
    ocrRuntimeGrid->setContentsMargins(0, 0, 0, 0);
    ocrRuntimeGrid->setHorizontalSpacing(6);
    ocrRuntimeGrid->setVerticalSpacing(6);
    m_cbOcrDevice = new QComboBox();
    m_cbOcrDevice->addItem("CPU", QStringLiteral("cpu"));
    m_cbOcrDevice->addItem("GPU 0", QStringLiteral("gpu:0"));
    m_cbOcrPrecision = new QComboBox();
    m_cbOcrPrecision->addItem("FP32", QStringLiteral("fp32"));
    m_cbOcrPrecision->addItem("FP16", QStringLiteral("fp16"));
    m_chkOcrMkldnn = new QCheckBox(QStringLiteral("MKLDNN"));
    m_chkOcrMkldnn->setChecked(true);
    m_spOcrThreads = new QSpinBox();
    m_spOcrThreads->setRange(1, 32);
    m_spOcrThreads->setValue(8);
    m_spOcrBatch = new QSpinBox();
    m_spOcrBatch->setRange(1, 32);
    m_spOcrBatch->setValue(6);
    ocrRuntimeGrid->addWidget(makeFieldLabel("OCR 设备"), 0, 0);
    ocrRuntimeGrid->addWidget(m_cbOcrDevice, 0, 1);
    ocrRuntimeGrid->addWidget(makeFieldLabel("精度"), 0, 2);
    ocrRuntimeGrid->addWidget(m_cbOcrPrecision, 0, 3);
    ocrRuntimeGrid->addWidget(m_chkOcrMkldnn, 1, 0);
    ocrRuntimeGrid->addWidget(makeFieldLabel("线程"), 1, 1);
    ocrRuntimeGrid->addWidget(m_spOcrThreads, 1, 2);
    ocrRuntimeGrid->addWidget(makeFieldLabel("rec batch"), 2, 0);
    ocrRuntimeGrid->addWidget(m_spOcrBatch, 2, 1);
    v->addWidget(m_ocrRuntimePanel);

    m_lblConfName = makeFieldLabel("置信度阈值");
    v->addWidget(m_lblConfName);
    auto* h1 = new QHBoxLayout(); h1->setSpacing(8);
    m_slConf = new QSlider(Qt::Horizontal);
    m_slConf->setRange(1, 99); m_slConf->setValue(25);
    m_lblConfVal = new QLabel("0.25"); m_lblConfVal->setMinimumWidth(40);
    m_lblConfVal->setStyleSheet("color:#1FB6FF;");
    h1->addWidget(m_slConf, 1); h1->addWidget(m_lblConfVal, 0);
    v->addLayout(h1);
    connect(m_slConf, &QSlider::valueChanged, this, [this](int v){
        const double value = isAnomalyMode() ? (v / 100.0) : (v / 100.0);
        m_lblConfVal->setText(QString::number(value, 'f', isAnomalyMode() ? 2 : 2));
        if (isAnomalyMode()) {
            m_anomalyMeta.imageThreshold = static_cast<float>(value);
        }
    });

    m_lblNmsName = makeFieldLabel("NMS IoU 阈值");
    v->addWidget(m_lblNmsName);
    auto* h2 = new QHBoxLayout(); h2->setSpacing(8);
    m_slNms = new QSlider(Qt::Horizontal);
    m_slNms->setRange(1, 99); m_slNms->setValue(45);
    m_lblNmsVal = new QLabel("0.45"); m_lblNmsVal->setMinimumWidth(40);
    m_lblNmsVal->setStyleSheet("color:#1FB6FF;");
    h2->addWidget(m_slNms, 1); h2->addWidget(m_lblNmsVal, 0);
    v->addLayout(h2);
    connect(m_slNms, &QSlider::valueChanged, this, [this](int v){
        const double value = isAnomalyMode() ? (v / 100.0) : (v / 100.0);
        m_lblNmsVal->setText(QString::number(value, 'f', isAnomalyMode() ? 2 : 2));
        if (isAnomalyMode()) {
            m_anomalyMeta.pixelThreshold = static_cast<float>(value);
        }
    });

    return c;
}

QWidget* InferenceTab::buildSourceCard()
{
    auto* c = makeCard();
    auto* v = new QVBoxLayout(c);
    v->setContentsMargins(14, 12, 14, 14);
    v->setSpacing(8);
    v->addWidget(makeCardTitle("输入源"));

    auto* h = new QHBoxLayout(); h->setSpacing(6);
    m_btnSrcImg = new QPushButton("图像");
    m_btnSrcVid = new QPushButton("视频");
    m_btnSrcCam = new QPushButton("USB 相机");
    for (auto* b : {m_btnSrcImg, m_btnSrcVid, m_btnSrcCam}) {
        b->setCheckable(true);
        h->addWidget(b);
    }
    m_srcGroup = new QButtonGroup(this);
    m_srcGroup->addButton(m_btnSrcImg, 0);
    m_srcGroup->addButton(m_btnSrcVid, 1);
    m_srcGroup->addButton(m_btnSrcCam, 2);
    m_btnSrcImg->setChecked(true);
    v->addLayout(h);
    connect(m_srcGroup, QOverload<int>::of(&QButtonGroup::idClicked),
            this, &InferenceTab::onSourceChanged);

    // 文件面板（图像/视频共用）
    m_filePanel = new QWidget();
    {
        auto* fv = new QVBoxLayout(m_filePanel);
        fv->setContentsMargins(0, 0, 0, 0); fv->setSpacing(6);
        fv->addWidget(makeFieldLabel("文件路径"));
        auto* fh = new QHBoxLayout(); fh->setSpacing(6);
        m_edFilePath = new QLineEdit();
        m_btnBrowseFile = new QPushButton("浏览");
        m_btnBrowseFile->setProperty("ghost", true);
        fh->addWidget(m_edFilePath, 1); fh->addWidget(m_btnBrowseFile, 0);
        fv->addLayout(fh);
        connect(m_btnBrowseFile, &QPushButton::clicked,
                this, &InferenceTab::onBrowseImageOrVideo);
    }
    v->addWidget(m_filePanel);

    // 相机面板
    m_camPanel = new QWidget();
    {
        auto* cv = new QVBoxLayout(m_camPanel);
        cv->setContentsMargins(0, 0, 0, 0); cv->setSpacing(6);
        cv->addWidget(makeFieldLabel("摄像头设备 (按设备名选择)"));
        m_cbCamera = new QComboBox();
        m_cbCamera->setMinimumHeight(28);
        m_cbCamera->addItem(QStringLiteral("(请点击 [扫描] 列举设备)"), -1);
        cv->addWidget(m_cbCamera);

        auto* ch = new QHBoxLayout(); ch->setSpacing(8);
        m_btnScanCam = new QPushButton("⟳ 扫描");
        m_btnScanCam->setProperty("ghost", true);
        m_btnScanCam->setMaximumWidth(90);
        m_btnConnect = new QPushButton("● 连接相机");
        ch->addWidget(m_btnScanCam, 0); ch->addWidget(m_btnConnect, 1);
        cv->addLayout(ch);

        m_lblCamStatus = new QLabel("未连接");
        m_lblCamStatus->setStyleSheet("color:#5A7896; font-size:10px;");
        cv->addWidget(m_lblCamStatus);

        connect(m_btnScanCam, &QPushButton::clicked,
                this, &InferenceTab::onScanCameras);
        connect(m_btnConnect, &QPushButton::clicked,
                this, &InferenceTab::onConnectCamera);
    }
    v->addWidget(m_camPanel);
    m_camPanel->hide();

    return c;
}

QWidget* InferenceTab::buildLabelCard()
{
    auto* c = makeCard();
    auto* v = new QVBoxLayout(c);
    v->setContentsMargins(14, 12, 14, 14);
    v->setSpacing(8);
    v->addWidget(makeCardTitle("标签 / metadata 配置"));

    m_labelPanel = new QWidget(c);
    {
        auto* lv = new QVBoxLayout(m_labelPanel);
        lv->setContentsMargins(0, 0, 0, 0);
        lv->setSpacing(6);
        lv->addWidget(makeFieldLabel("labels.json 路径"));
        auto* row = new QHBoxLayout(); row->setSpacing(6);
        m_edLabel = new QLineEdit();
        auto* btnB = new QPushButton("浏览");
        auto* btnR = new QPushButton("重载");
        btnB->setProperty("ghost", true); btnR->setProperty("ghost", true);
        row->addWidget(m_edLabel, 1);
        row->addWidget(btnB, 0); row->addWidget(btnR, 0);
        lv->addLayout(row);
        connect(btnB, &QPushButton::clicked, this, &InferenceTab::onBrowseLabel);
        connect(btnR, &QPushButton::clicked, this, &InferenceTab::onReloadLabel);

        m_lblLabelInfo = new QLabel("尚未加载");
        m_lblLabelInfo->setStyleSheet("color:#5A7896; font-size:10px;");
        lv->addWidget(m_lblLabelInfo);
    }
    v->addWidget(m_labelPanel);

    m_metadataPanel = new QWidget(c);
    {
        auto* mv = new QVBoxLayout(m_metadataPanel);
        mv->setContentsMargins(0, 0, 0, 0);
        mv->setSpacing(6);
        mv->addWidget(makeFieldLabel("metadata.json 路径"));
        auto* row = new QHBoxLayout(); row->setSpacing(6);
        m_edMetadata = new QLineEdit();
        m_edMetadata->setPlaceholderText("选择 Anomalib metadata.json");
        auto* btnB = new QPushButton("浏览");
        auto* btnR = new QPushButton("读取");
        btnB->setProperty("ghost", true); btnR->setProperty("ghost", true);
        row->addWidget(m_edMetadata, 1);
        row->addWidget(btnB, 0); row->addWidget(btnR, 0);
        mv->addLayout(row);
        connect(btnB, &QPushButton::clicked, this, &InferenceTab::onBrowseMetadata);
        connect(btnR, &QPushButton::clicked, this, &InferenceTab::onReloadMetadata);

        m_lblMetadataInfo = new QLabel("尚未读取 metadata");
        m_lblMetadataInfo->setWordWrap(true);
        m_lblMetadataInfo->setStyleSheet("color:#5A7896; font-size:10px;");
        mv->addWidget(m_lblMetadataInfo);
    }
    v->addWidget(m_metadataPanel);
    return c;
}

QWidget* InferenceTab::buildActionRow()
{
    auto* w = new QWidget();
    auto* h = new QHBoxLayout(w);
    h->setContentsMargins(0, 0, 0, 0); h->setSpacing(10);
    m_btnStart = new QPushButton("▶  开始推理");
    m_btnStart->setProperty("primary", true);
    m_btnStart->setMinimumHeight(48);
    m_btnStop = new QPushButton("■  停止");
    m_btnStop->setMinimumHeight(48);
    m_btnStop->setStyleSheet(
        "QPushButton{background:#0A1220;color:#FF4D6D;border:1px solid #FF4D6D;}"
        "QPushButton:hover{background:#2a0d12;}"
        "QPushButton:disabled{color:#5A2630;border-color:#5A2630;}");
    m_btnStop->setMaximumWidth(120);
    h->addWidget(m_btnStart, 1); h->addWidget(m_btnStop, 0);
    connect(m_btnStart, &QPushButton::clicked, this, &InferenceTab::onStartClicked);
    connect(m_btnStop,  &QPushButton::clicked, this, &InferenceTab::onStopClicked);
    return w;
}

QWidget* InferenceTab::buildStatsBar()
{
    auto* f = new QFrame();
    f->setObjectName("card");
    f->setMinimumHeight(60);
    auto* h = new QHBoxLayout(f);
    h->setContentsMargins(16, 8, 16, 8); h->setSpacing(24);

    auto makeMetric = [&](const QString& lab, QLabel** valOut, const QString& accent = "", QLabel** labOut = nullptr) {
        auto* col = new QVBoxLayout(); col->setSpacing(2);
        auto* l = new QLabel(lab); l->setObjectName("metricLabel");
        auto* v = new QLabel("-"); v->setObjectName("metricValue");
        if (!accent.isEmpty()) v->setProperty("accent", accent);
        col->addWidget(l); col->addWidget(v);
        h->addLayout(col);
        *valOut = v;
        if (labOut) *labOut = l;
    };

    makeMetric("FPS",    &m_lblFps, "cyan");
    makeMetric("预处理", &m_lblPreMs);
    makeMetric("推理",   &m_lblInfMs);
    makeMetric("后处理", &m_lblPostMs);
    makeMetric("检测数", &m_lblDetCount, "cyan", &m_lblCountName);
    h->addStretch(1);

    m_lblResInfo = new QLabel("-");
    m_lblResInfo->setObjectName("statusPill");
    h->addWidget(m_lblResInfo);
    return f;
}

QWidget* InferenceTab::buildCanvas()
{
    const QString canvasStyle = QStringLiteral(
        "background:#020407; border:1px solid #1FB6FF; border-radius:6px;");
    const QString labelStyle  = QStringLiteral(
        "color:#5A7896; font-size:14px; letter-spacing:2px; border:none;");

    m_canvasStack = new QStackedWidget();

    // --- Page 0: single canvas (YOLO & default) ---
    m_singleCanvasPage = new QFrame();
    m_singleCanvasPage->setStyleSheet(canvasStyle);
    auto* sv = new QVBoxLayout(m_singleCanvasPage);
    sv->setContentsMargins(0, 0, 0, 0);
    m_lblCanvas = new ZoomableCanvasLabel("等待输入...");
    m_lblCanvas->setMinimumSize(640, 360);
    m_lblCanvas->setStyleSheet(labelStyle);
    sv->addWidget(m_lblCanvas);
    m_canvasStack->addWidget(m_singleCanvasPage);

    // --- Page 1: triple canvas (anomaly mode) ---
    m_dualCanvasPage = new QWidget();
    auto* dualH = new QHBoxLayout(m_dualCanvasPage);
    dualH->setContentsMargins(0, 0, 0, 0);
    dualH->setSpacing(6);

    auto makeSubFrame = [&](const QString& title, QLabel*& canvasOut, QWidget*& frameOut) {
        auto* frame = new QFrame();
        frame->setStyleSheet(canvasStyle);
        auto* fv = new QVBoxLayout(frame);
        fv->setContentsMargins(0, 0, 0, 0);
        auto* canvas = new ZoomableCanvasLabel(QStringLiteral("等待输入..."));
        canvas->setMinimumSize(300, 200);
        canvas->setStyleSheet(labelStyle);
        static_cast<ZoomableCanvasLabel*>(canvas)->setTitle(title);
        fv->addWidget(canvas);
        canvasOut = canvas;
        frameOut = frame;
        return frame;
    };

    dualH->addWidget(makeSubFrame(QStringLiteral("Image"),               m_lblImageCanvas,   m_imageFrame),   1);
    dualH->addWidget(makeSubFrame(QStringLiteral("Image + Anomaly Map"), m_lblOverlayCanvas, m_overlayFrame), 1);
    dualH->addWidget(makeSubFrame(QStringLiteral("Image + Pred Mask"),   m_lblMaskCanvas,    m_maskFrame),    1);
    m_canvasStack->addWidget(m_dualCanvasPage);

    static_cast<ZoomableCanvasLabel*>(m_lblImageCanvas)->onDoubleClick = [this]() {
        toggleAnomalyFullscreen(0);
    };
    static_cast<ZoomableCanvasLabel*>(m_lblOverlayCanvas)->onDoubleClick = [this]() {
        toggleAnomalyFullscreen(1);
    };
    static_cast<ZoomableCanvasLabel*>(m_lblMaskCanvas)->onDoubleClick = [this]() {
        toggleAnomalyFullscreen(2);
    };

    m_canvasStack->setCurrentIndex(0);

    m_lblLive = new QLabel(m_canvasStack);
    m_lblLive->setText("● LIVE");
    m_lblLive->setStyleSheet("color:#FF4D6D; background:#0A1220AA; "
                             "padding:3px 8px; border:1px solid #1FB6FF; "
                             "border-radius:3px; font-size:10px; letter-spacing:2px; font-weight:700;");
    m_lblLive->move(14, 14);
    m_lblLive->hide();
    return m_canvasStack;
}

QWidget* InferenceTab::buildDetList()
{
    auto* c = makeCard();
    c->setMinimumWidth(220);
    auto* v = new QVBoxLayout(c);
    v->setContentsMargins(12, 10, 12, 10); v->setSpacing(6);
    v->addWidget(makeCardTitle("检测结果"));
    m_detListW = new QListWidget();
    m_detListW->setSelectionMode(QAbstractItemView::SingleSelection);
    m_detListW->setStyleSheet(
        "QListWidget{background:#03060B;border:1px solid #16334D;color:#9FC8E8;"
        "font-family:Consolas;font-size:11px;}"
        "QListWidget::item{padding:6px;border-left:2px solid #1FB6FF;margin-bottom:2px;}"
        "QListWidget::item:hover{background:#0F2A45;}");
    v->addWidget(m_detListW, 1);
    return c;
}

void InferenceTab::onBrowseEngine()
{
    if (isOcrMode()) {
        QString p = QFileDialog::getExistingDirectory(this, "选择 OCR 检测模型目录", m_edEngine->text());
        if (!p.isEmpty()) m_edEngine->setText(QDir::toNativeSeparators(p));
        return;
    }
    QString p = QFileDialog::getOpenFileName(this, "选择 .trt 引擎",
        m_edEngine->text(), "TensorRT 引擎 (*.trt *.engine);;所有文件 (*.*)");
    if (!p.isEmpty()) m_edEngine->setText(QDir::toNativeSeparators(p));
}

void InferenceTab::onBrowseOcrRecModel()
{
    QString p = QFileDialog::getExistingDirectory(this, "选择 OCR 识别模型目录",
                                                  m_edOcrRecModel ? m_edOcrRecModel->text() : QString());
    if (!p.isEmpty() && m_edOcrRecModel) {
        m_edOcrRecModel->setText(QDir::toNativeSeparators(p));
    }
}

void InferenceTab::onBrowseOcrLineOrientationModel()
{
    QString p = QFileDialog::getExistingDirectory(this, "选择 OCR 文本行方向模型目录",
                                                  m_edOcrLineOrientationModel ? m_edOcrLineOrientationModel->text() : QString());
    if (!p.isEmpty() && m_edOcrLineOrientationModel) {
        m_edOcrLineOrientationModel->setText(QDir::toNativeSeparators(p));
    }
}

void InferenceTab::onBrowseLabel()
{
    QString p = QFileDialog::getOpenFileName(this, "选择 labels.json",
        m_edLabel->text(), "JSON (*.json);;所有文件 (*.*)");
    if (!p.isEmpty()) {
        m_edLabel->setText(QDir::toNativeSeparators(p));
        onReloadLabel();
    }
}

void InferenceTab::onReloadLabel()
{
    if (m_labels->load(m_edLabel->text())) {
        m_lblLabelInfo->setText(QStringLiteral("已加载 %1 个类别 · %2")
            .arg(m_labels->count()).arg(m_labels->datasetName()));
    } else {
        QMessageBox::warning(this, "标签", "无法加载 labels.json");
    }
}

void InferenceTab::onBrowseMetadata()
{
    QString p = QFileDialog::getOpenFileName(this, "选择 metadata.json",
        m_edMetadata->text(), "JSON (*.json);;所有文件 (*.*)");
    if (!p.isEmpty()) {
        m_edMetadata->setText(QDir::toNativeSeparators(p));
        onReloadMetadata();
    }
}

void InferenceTab::onReloadMetadata()
{
    QFile file(m_edMetadata->text());
    if (!file.open(QIODevice::ReadOnly)) {
        m_anomalyMetaLoaded = false;
        m_lblMetadataInfo->setText(QStringLiteral("metadata 读取失败：%1").arg(m_edMetadata->text()));
        QMessageBox::warning(this, "metadata", "无法读取 metadata.json");
        return;
    }

    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        m_anomalyMetaLoaded = false;
        m_lblMetadataInfo->setText(QStringLiteral("metadata JSON 格式错误：%1").arg(err.errorString()));
        return;
    }

    const QJsonObject obj = doc.object();
    AnomalyMetadata meta;
    meta.modelType = obj.value("model_type").isString()
        ? obj.value("model_type").toString()
        : (m_cbAnomalyModel ? m_cbAnomalyModel->currentText() : QStringLiteral("Auto"));
    meta.normalization = obj.value("normalization").isString()
        ? obj.value("normalization").toString().toLower()
        : QString();
    meta.outputContract = obj.value("output_contract").isString()
        ? obj.value("output_contract").toString().toLower()
        : QString();
    if (meta.normalization.isEmpty()) {
        meta.normalization = meta.outputContract == QStringLiteral("anomalib_post_processed_0_1")
            ? QStringLiteral("none")
            : (meta.modelType.contains("efficient", Qt::CaseInsensitive) ? QStringLiteral("none") : QStringLiteral("imagenet"));
    }
    if (m_cbAnomalyModel) {
        int idx = -1;
        for (int i = 0; i < m_cbAnomalyModel->count(); ++i) {
            if (m_cbAnomalyModel->itemText(i).compare(meta.modelType, Qt::CaseInsensitive) == 0) {
                idx = i;
                break;
            }
        }
        if (idx >= 0) {
            QSignalBlocker blocker(m_cbAnomalyModel);
            m_cbAnomalyModel->setCurrentIndex(idx);
        }
    }
    if (obj.value("image_threshold").isDouble())
        meta.imageThreshold = static_cast<float>(obj.value("image_threshold").toDouble());
    if (obj.value("pixel_threshold").isDouble())
        meta.pixelThreshold = static_cast<float>(obj.value("pixel_threshold").toDouble());

    const QJsonObject scores = obj.value("pred_scores").toObject();
    const QJsonObject maps = obj.value("anomaly_maps").toObject();
    if (scores.value("min").isDouble()) meta.scoreMin = static_cast<float>(scores.value("min").toDouble());
    if (scores.value("max").isDouble()) meta.scoreMax = static_cast<float>(scores.value("max").toDouble());
    if (maps.value("min").isDouble()) meta.mapMin = static_cast<float>(maps.value("min").toDouble());
    if (maps.value("max").isDouble()) meta.mapMax = static_cast<float>(maps.value("max").toDouble());
    if (obj.value("pred_scores.min").isDouble()) meta.scoreMin = static_cast<float>(obj.value("pred_scores.min").toDouble());
    if (obj.value("pred_scores.max").isDouble()) meta.scoreMax = static_cast<float>(obj.value("pred_scores.max").toDouble());
    if (obj.value("anomaly_maps.min").isDouble()) meta.mapMin = static_cast<float>(obj.value("anomaly_maps.min").toDouble());
    if (obj.value("anomaly_maps.max").isDouble()) meta.mapMax = static_cast<float>(obj.value("anomaly_maps.max").toDouble());
    if (obj.value("min").isDouble()) {
        meta.scoreMin = meta.mapMin = static_cast<float>(obj.value("min").toDouble());
    }
    if (obj.value("max").isDouble()) {
        meta.scoreMax = meta.mapMax = static_cast<float>(obj.value("max").toDouble());
    }
    if (obj.value("input_size").isArray()) {
        const QJsonArray size = obj.value("input_size").toArray();
        if (size.size() >= 2) {
            meta.inputH = size.at(0).toInt();
            meta.inputW = size.at(1).toInt();
        }
    }
    if (obj.value("transform").isObject()) {
        const QJsonObject transform = obj.value("transform").toObject();
        if (transform.value("height").isDouble()) meta.inputH = transform.value("height").toInt();
        if (transform.value("width").isDouble()) meta.inputW = transform.value("width").toInt();
    }

    m_anomalyMeta = meta;
    m_anomalyMetaLoaded = obj.value("image_threshold").isDouble() && obj.value("pixel_threshold").isDouble();
    m_lblMetadataInfo->setText(m_anomalyMetaLoaded
        ? QStringLiteral("已读取：%1 · normalization=%2 · image_threshold=%3 · pixel_threshold=%4 · score[%5,%6] · map[%7,%8]")
            .arg(meta.modelType)
            .arg(meta.normalization)
            .arg(meta.imageThreshold, 0, 'f', 3)
            .arg(meta.pixelThreshold, 0, 'f', 3)
            .arg(meta.scoreMin, 0, 'f', 3)
            .arg(meta.scoreMax, 0, 'f', 3)
            .arg(meta.mapMin, 0, 'f', 3)
            .arg(meta.mapMax, 0, 'f', 3)
        : QStringLiteral("metadata 缺少 image_threshold 或 pixel_threshold，无法开始异常检测"));
    updateTaskUi();
}

void InferenceTab::onTaskChanged(int /*index*/)
{
    updateTaskUi();
}

void InferenceTab::onSourceChanged(int id)
{
    bool isCam = (id == 2);
    m_camPanel->setVisible(isCam);
    m_filePanel->setVisible(!isCam);

    // 切换源时只停止采集，保留已加载模型，便于下一次开始推理快速复用。
    stopCaptureThread();
    m_lblLive->hide();
    static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->clearCanvas("等待输入...");

    if (!isCam) {
        m_cameraConnected = false;
        m_lblCamStatus->setText("未连接");
        m_lblCamStatus->setStyleSheet("color:#5A7896; font-size:10px;");
        m_btnConnect->setText("● 连接相机");
    }
}

void InferenceTab::onBrowseImageOrVideo()
{
    bool isVideo = m_btnSrcVid->isChecked();
    QString filter = isVideo
        ? "视频 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*.*)"
        : "图像 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)";
    QString p = QFileDialog::getOpenFileName(this,
        isVideo ? "选择视频" : "选择图像",
        m_edFilePath->text(), filter);
    if (!p.isEmpty()) {
        m_edFilePath->setText(QDir::toNativeSeparators(p));
        previewLocalFile(p);
    }
}

void InferenceTab::previewLocalFile(const QString& path)
{
    // 选完图像/视频立即在画布显示首帧（无需等待 [开始推理]）
    QImage img;
    if (m_btnSrcImg->isChecked()) {
        if (!img.load(path)) {
            static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->clearCanvas(QStringLiteral("无法读取图像"));
            return;
        }
    } else {
        cv::VideoCapture cap;
        QString tempVideoPath;
        if (!openVideoPreviewUnicode(cap, path, &tempVideoPath)) {
            static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->clearCanvas(QStringLiteral("无法打开视频"));
            return;
        }
        if (!cap.isOpened()) {
            static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->clearCanvas(QStringLiteral("无法打开视频"));
            return;
        }
        cv::Mat f;
        for (int k = 0; k < 3 && (!cap.read(f) || f.empty()); ++k) {}
        if (f.empty()) {
            if (!tempVideoPath.isEmpty()) QFile::remove(tempVideoPath);
            static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->clearCanvas(QStringLiteral("视频读取失败"));
            return;
        }
        cv::Mat rgb; cv::cvtColor(f, rgb, cv::COLOR_BGR2RGB);
        img = QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                     QImage::Format_RGB888).copy();
        cap.release();
        if (!tempVideoPath.isEmpty()) QFile::remove(tempVideoPath);
    }
    m_srcW = img.width(); m_srcH = img.height();
    m_lblResInfo->setText(QStringLiteral("%1×%2  (预览)").arg(m_srcW).arg(m_srcH));
    static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->setImage(img, true);
}

void InferenceTab::onScanCameras()
{
    m_btnScanCam->setEnabled(false);
    m_lblCamStatus->setText(QStringLiteral("正在扫描设备 ..."));
    m_lblCamStatus->setStyleSheet("color:#FFB72A; font-size:10px;");
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    auto cams = CaptureWorker::enumerateCameras(6);

    QApplication::restoreOverrideCursor();
    m_cbCamera->clear();
    if (cams.isEmpty()) {
        m_cbCamera->addItem(QStringLiteral("(未找到任何摄像头设备)"), -1);
        m_lblCamStatus->setText(QStringLiteral("未找到摄像头，请检查 USB 连接"));
        m_lblCamStatus->setStyleSheet("color:#FF4D6D; font-size:10px;");
        m_btnScanCam->setEnabled(true);
        QMessageBox::warning(this, "相机", QStringLiteral(
            "未检测到任何视频输入设备。\n\n"
            "排查：\n"
            "  1. USB 摄像头是否已正确插入\n"
            "  2. 是否被其他程序占用 (微信/QQ/Teams 等)\n"
            "  3. 驱动是否安装"));
        return;
    }

    for (const auto& c : cams) {
        const char* beName = (c.backend == cv::CAP_DSHOW) ? "DSHOW"
                          : (c.backend == cv::CAP_MSMF)  ? "MSMF"
                          : "ANY";
        QString text = QStringLiteral("#%1  %2  [%3]  %4×%5")
            .arg(c.index).arg(c.name).arg(beName).arg(c.width).arg(c.height);
        // userData 编码: (backend << 16) | index
        int data = (c.backend << 16) | (c.index & 0xFFFF);
        m_cbCamera->addItem(text, data);
    }

    // 默认优先选择名字里不含 "Integrated" 的项 (大概率为外置 USB 相机)
    int sel = 0;
    for (int i = 0; i < cams.size(); ++i) {
        if (!cams[i].name.contains("Integrated", Qt::CaseInsensitive)
            && !cams[i].name.contains("Internal",   Qt::CaseInsensitive)) {
            sel = i; break;
        }
    }
    m_cbCamera->setCurrentIndex(sel);

    m_lblCamStatus->setText(QStringLiteral("发现 %1 个设备，请点击 [连接相机] 试用").arg(cams.size()));
    m_lblCamStatus->setStyleSheet("color:#22E27F; font-size:10px;");
    m_btnScanCam->setEnabled(true);
}

void InferenceTab::onConnectCamera()
{
    int data = m_cbCamera->currentData().toInt();
    if (data < 0) {
        QMessageBox::warning(this, "相机",
            QStringLiteral("请先点击 [扫描] 列举可用设备")); return;
    }
    int idx     = data & 0xFFFF;
    int backend = (data >> 16) & 0xFFFF;

    int w = 0, h = 0; double fps = 0;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    bool ok = CaptureWorker::probeCamera(idx, backend, &w, &h, &fps);
    QApplication::restoreOverrideCursor();
    if (!ok) {
        m_cameraConnected = false;
        m_lblCamStatus->setText("未连接");
        m_lblCamStatus->setStyleSheet("color:#FF4D6D; font-size:10px;");
        m_btnConnect->setText("● 连接相机");
        QMessageBox::warning(this, "相机",
            QStringLiteral("无法打开所选相机：%1\n请尝试切换到其他条目")
                .arg(m_cbCamera->currentText()));
        return;
    }
    m_cameraConnected = true;
    m_srcW = w; m_srcH = h; m_srcFps = fps > 0 ? fps : 30.0;
    m_lblCamStatus->setText(QStringLiteral("● 已连接  %1×%2 @ %3fps")
        .arg(w).arg(h).arg(QString::number(m_srcFps, 'f', 0)));
    m_lblCamStatus->setStyleSheet("color:#22E27F; font-size:10px;");
    m_btnConnect->setText("✓ 已连接 (重新检测)");
    m_lblResInfo->setText(QStringLiteral("%1×%2 @ %3fps  (实时预览)")
        .arg(w).arg(h).arg(QString::number(m_srcFps, 'f', 0)));

    // 启动一路"仅预览"的采集线程，连续把帧贴到画布
    stopCaptureThread();

    m_capThread = new QThread(this);
    m_capWorker = new CaptureWorker();
    m_capWorker->moveToThread(m_capThread);
    m_capWorker->setSource(SourceType::Camera, QString::number(idx), backend);

    connect(m_capWorker, &CaptureWorker::frameReady,
            this, &InferenceTab::onFrameReady, Qt::QueuedConnection);
    connect(m_capWorker, &CaptureWorker::errorOccurred,
            this, &InferenceTab::onCaptureError, Qt::QueuedConnection);
    connect(m_capWorker, &CaptureWorker::finished,
            this, &InferenceTab::onCaptureFinished, Qt::QueuedConnection);
    connect(m_capThread, &QThread::started, m_capWorker, &CaptureWorker::start);
    connect(m_capThread, &QThread::finished, m_capWorker, &QObject::deleteLater);

    m_lblLive->show();
    setRunningUi(true);   // 让"停止"按钮在预览态也能用
    // 预览态允许用户直接切到推理（onStartClicked 内会先 cleanupThreads 收掉预览）
    m_btnStart->setEnabled(true);
    m_btnConnect->setEnabled(true);
    m_btnScanCam->setEnabled(true);
    m_cbCamera->setEnabled(true);
    m_capThread->start();
}

void InferenceTab::onStartClicked()
{
    if (m_edEngine->text().isEmpty()) {
        QMessageBox::warning(this, "提示", isOcrMode() ? "请选择 OCR 检测模型目录" : "请选择 .trt 引擎文件"); return;
    }
    if (isOcrMode() && (!m_edOcrRecModel || m_edOcrRecModel->text().isEmpty())) {
        QMessageBox::warning(this, "提示", "OCR 模式请先选择识别模型目录"); return;
    }
    if (isAnomalyMode()) {
        if (m_edMetadata->text().isEmpty()) {
            QMessageBox::warning(this, "提示", "异常检测模式请先选择 ONNX 导出同目录的 metadata.json"); return;
        }
        if (!m_anomalyMetaLoaded) {
            onReloadMetadata();
            if (!m_anomalyMetaLoaded) {
                QMessageBox::warning(this, "提示", "metadata.json 未通过读取检查，不能开始异常检测"); return;
            }
        }
    }
    int srcId = m_srcGroup->checkedId();
    if (srcId == 2 && !m_cameraConnected) {
        QMessageBox::warning(this, "提示", "请先点击 [连接相机] 检测相机"); return;
    }
    if (srcId != 2 && m_edFilePath->text().isEmpty()) {
        QMessageBox::warning(this, "提示", "请选择输入文件"); return;
    }

    m_pendingSrcId = srcId;
    m_pendingCamBackend = 0;
    if (srcId == 2) {
        const int data = m_cbCamera->currentData().toInt();
        m_pendingSourcePath = QString::number(data & 0xFFFF);
        m_pendingCamBackend = (data >> 16) & 0xFFFF;
    } else {
        m_pendingSourcePath = m_edFilePath->text();
    }

    stopCaptureThread();

    const QString modelKey = currentModelKey();
    if (hasReusableWorker(modelKey)) {
        m_loadedModelKey = modelKey;
        AppFileLogger::append({
            AppLogLevel::Stage,
            AppEventSource::Inference,
            QStringLiteral("模型未变化，复用已加载推理模型：%1").arg(m_edEngine->text())
        });
        setRunningUi(true);
        startCaptureAfterEngineLoaded();
        return;
    }

    releaseInferenceWorker();

    m_infThread = new QThread(this);
    m_loadingEngine = true;
    m_pendingModelKey = modelKey;
    setRunningUi(true);
    showLoadingProgress();
    AppFileLogger::append({
        AppLogLevel::Stage,
        AppEventSource::Inference,
        QStringLiteral("开始加载推理模型：%1").arg(m_edEngine->text())
    });

    if (isAnomalyMode()) {
        m_anomalyWorker = new AnomalyInferWorker();
        m_anomalyWorker->moveToThread(m_infThread);
        connect(m_infThread, &QThread::finished, m_anomalyWorker, &QObject::deleteLater);
        connect(m_anomalyWorker, &AnomalyInferWorker::engineLoaded,
                this, &InferenceTab::onEngineLoaded, Qt::QueuedConnection);
        connect(m_anomalyWorker, &AnomalyInferWorker::engineLoadFailed,
                this, &InferenceTab::onEngineLoadFailed, Qt::QueuedConnection);
        connect(m_anomalyWorker, &AnomalyInferWorker::metadataReady,
                this, &InferenceTab::onAnomalyMetadataReady, Qt::QueuedConnection);
        connect(m_anomalyWorker, &AnomalyInferWorker::resultReady,
                this, &InferenceTab::onAnomalyResultReady, Qt::QueuedConnection);
        connect(m_anomalyWorker, &AnomalyInferWorker::errorOccurred,
                this, &InferenceTab::onInferError, Qt::QueuedConnection);
        m_infThread->start();

        const QString modelType = m_cbAnomalyModel ? m_cbAnomalyModel->currentText() : QStringLiteral("Auto");
        m_anomalyMeta.imageThreshold = m_slConf->value() / 100.0f;
        m_anomalyMeta.pixelThreshold = m_slNms->value() / 100.0f;
        AnomalyInferenceAdapter::loadEngineAsync(m_anomalyWorker,
                                                 m_edEngine->text(),
                                                 m_edMetadata->text(),
                                                 modelType,
                                                 m_anomalyMeta.imageThreshold,
                                                 m_anomalyMeta.pixelThreshold);
    } else if (isOcrMode()) {
        m_ocrWorker = new OcrInferWorker();
        m_ocrWorker->moveToThread(m_infThread);
        connect(m_infThread, &QThread::finished, m_ocrWorker, &QObject::deleteLater);
        connect(m_ocrWorker, &OcrInferWorker::engineLoaded,
                this, &InferenceTab::onEngineLoaded, Qt::QueuedConnection);
        connect(m_ocrWorker, &OcrInferWorker::engineLoadFailed,
                this, &InferenceTab::onEngineLoadFailed, Qt::QueuedConnection);
        connect(m_ocrWorker, &OcrInferWorker::resultReady,
                this, &InferenceTab::onOcrResultReady, Qt::QueuedConnection);
        connect(m_ocrWorker, &OcrInferWorker::errorOccurred,
                this, &InferenceTab::onInferError, Qt::QueuedConnection);
        m_infThread->start();

        OcrConfig config;
        config.detModelDir = m_edEngine->text();
        config.recModelDir = m_edOcrRecModel ? m_edOcrRecModel->text() : QString();
        config.textlineOrientationModelDir = m_edOcrLineOrientationModel ? m_edOcrLineOrientationModel->text() : QString();
        config.useTextlineOrientation = !config.textlineOrientationModelDir.trimmed().isEmpty();
        config.device = m_cbOcrDevice ? m_cbOcrDevice->currentData().toString() : QStringLiteral("cpu");
        config.precision = m_cbOcrPrecision ? m_cbOcrPrecision->currentData().toString() : QStringLiteral("fp32");
        config.enableMkldnn = m_chkOcrMkldnn ? m_chkOcrMkldnn->isChecked() : true;
        config.cpuThreads = m_spOcrThreads ? m_spOcrThreads->value() : 8;
        config.recBatchSize = m_spOcrBatch ? m_spOcrBatch->value() : 6;
        config.detThresh = m_slConf->value() / 100.0f;
        config.detBoxThresh = m_slNms->value() / 100.0f;
        config.detUnclipRatio = 1.5f;
        config.recScoreThresh = 0.0f;
        OcrInferenceAdapter::loadEngineAsync(m_ocrWorker, config);
    } else {
        m_infWorker = new InferWorker();
        m_infWorker->moveToThread(m_infThread);
        connect(m_infThread, &QThread::finished, m_infWorker, &QObject::deleteLater);
        connect(m_infWorker, &InferWorker::engineLoaded,
                this, &InferenceTab::onEngineLoaded, Qt::QueuedConnection);
        connect(m_infWorker, &InferWorker::engineLoadFailed,
                this, &InferenceTab::onEngineLoadFailed, Qt::QueuedConnection);
        connect(m_infWorker, &InferWorker::resultReady,
                this, &InferenceTab::onResultReady, Qt::QueuedConnection);
        connect(m_infWorker, &InferWorker::errorOccurred,
                this, &InferenceTab::onInferError, Qt::QueuedConnection);
        m_infThread->start();

        int task = m_cbTask->currentData().toInt();
        float conf = m_slConf->value() / 100.0f;
        float nms  = m_slNms->value()  / 100.0f;
        YoloInferenceAdapter::loadEngineAsync(m_infWorker, m_edEngine->text(), task, conf, nms);
    }
}

void InferenceTab::onStopClicked()
{
    // 一键停止：正常推理时只停采集并保留模型；加载过程中则做全量清理。
    AppFileLogger::append({ AppLogLevel::Stage, AppEventSource::Inference, QStringLiteral("停止推理/预览") });
    static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->clearCanvas("正在停止...");
    if (m_loadingEngine) {
        m_loadingEngine = false;
        cleanupThreads();
    } else {
        stopCaptureThread();
    }
    m_lblLive->hide();
    setRunningUi(false);
    m_cameraConnected = false;
    m_btnConnect->setText("连接相机");
    static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->clearCanvas("已停止");
}

void InferenceTab::onCaptureFinished()
{
    m_inferenceActive = false;
    setRunningUi(false);
    m_lblLive->hide();
}

void InferenceTab::onSourceInfo(int w, int h, double fps)
{
    m_srcW = w; m_srcH = h; m_srcFps = fps;
    m_lblResInfo->setText(QStringLiteral("%1×%2%3")
        .arg(w).arg(h)
        .arg(fps > 0 ? QStringLiteral(" @ %1fps").arg(QString::number(fps, 'f', 0))
                     : QString()));
}

void InferenceTab::onCaptureError(QString msg)
{
    m_inferenceActive = false;
    setRunningUi(false);
    m_lblLive->hide();
    AppFileLogger::append({ AppLogLevel::Error, AppEventSource::Inference, QStringLiteral("采集错误：%1").arg(msg) });
    QMessageBox::critical(this, "采集错误", msg);
}

void InferenceTab::onInferError(QString msg)
{
    AppFileLogger::append({ AppLogLevel::Error, AppEventSource::Inference, QStringLiteral("推理错误：%1").arg(msg) });
    QMessageBox::critical(this, "推理错误", msg);
}

void InferenceTab::onEngineLoaded()
{
    if (!m_loadingEngine) {
        return;
    }
    m_loadingEngine = false;
    hideLoadingProgress();
    m_loadedModelKey = m_pendingModelKey;
    m_pendingModelKey.clear();
    AppFileLogger::append({ AppLogLevel::Success, AppEventSource::Inference, QStringLiteral("推理模型加载完成，启动采集") });
    startCaptureAfterEngineLoaded();
}

void InferenceTab::onEngineLoadFailed(QString msg)
{
    if (!m_loadingEngine) {
        return;
    }
    m_loadingEngine = false;
    m_pendingModelKey.clear();
    hideLoadingProgress();
    cleanupThreads();
    setRunningUi(false);
    m_lblLive->hide();
    AppFileLogger::append({ AppLogLevel::Error, AppEventSource::Inference, QStringLiteral("模型加载失败：%1").arg(msg) });
    QMessageBox::critical(this, "模型加载失败", msg);
}

void InferenceTab::onFrameReady(QImage frame, qint64 /*ts*/)
{
    // 仅在"预览模式"时直接显示原始帧；
    // 推理运行时由 onResultReady 负责贴叠加了检测框/掩膜的帧，避免重复刷新。
    if (m_inferenceActive) return;
    if (frame.isNull()) return;
    static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->setImage(frame);
}

void InferenceTab::onResultReady(QImage frame, QVector<Detection> dets,
                                 double preMs, double infMs, double postMs, double totalMs)
{
    paintOverlay(frame, dets);
    static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->setImage(frame);

    m_lblPreMs->setText(InferenceResultPresenter::milliseconds(preMs));
    m_lblInfMs->setText(InferenceResultPresenter::milliseconds(infMs));
    m_lblPostMs->setText(InferenceResultPresenter::milliseconds(postMs));
    m_lblDetCount->setText(InferenceResultPresenter::detectionCount(dets.size()));

    m_frameCount++;
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    qint64 dt = now - m_fpsTimestamp;
    if (dt >= 500) {
        m_currentFps = m_frameCount * 1000.0 / dt;
        m_lblFps->setText(QString::number(m_currentFps, 'f', 1));
        m_frameCount = 0;
        m_fpsTimestamp = now;
    }

    updateDetList(dets);
}

void InferenceTab::onAnomalyResultReady(AnomalyResult result,
                                        double preMs, double infMs, double postMs, double /*totalMs*/)
{
    if (m_lblImageCanvas)
        static_cast<ZoomableCanvasLabel*>(m_lblImageCanvas)->setImage(result.sourceImage);
    if (m_lblOverlayCanvas)
        static_cast<ZoomableCanvasLabel*>(m_lblOverlayCanvas)->setImage(result.overlayImage);
    if (m_lblMaskCanvas)
        static_cast<ZoomableCanvasLabel*>(m_lblMaskCanvas)->setImage(result.maskImage);

    m_lblPreMs->setText(InferenceResultPresenter::milliseconds(preMs));
    m_lblInfMs->setText(InferenceResultPresenter::milliseconds(infMs));
    m_lblPostMs->setText(InferenceResultPresenter::milliseconds(postMs));
    m_lblDetCount->setText(result.statusText);
    m_lblDetCount->setStyleSheet(InferenceResultPresenter::anomalyStatusStyle(result.isAnomaly));

    m_frameCount++;
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    qint64 dt = now - m_fpsTimestamp;
    if (dt >= 500) {
        m_currentFps = m_frameCount * 1000.0 / dt;
        m_lblFps->setText(QString::number(m_currentFps, 'f', 1));
        m_frameCount = 0;
        m_fpsTimestamp = now;
    }

    updateAnomalyList(result);
}

void InferenceTab::onOcrResultReady(OcrResult result,
                                    double preMs, double infMs, double postMs, double /*totalMs*/)
{
    m_lastOcrResult = result;
    m_selectedOcrLineRow = -1;
    refreshOcrCanvasHighlight();

    m_lblPreMs->setText(InferenceResultPresenter::milliseconds(preMs));
    m_lblInfMs->setText(InferenceResultPresenter::milliseconds(infMs));
    m_lblPostMs->setText(InferenceResultPresenter::milliseconds(postMs));
    m_lblDetCount->setStyleSheet("");
    m_lblDetCount->setText(InferenceResultPresenter::detectionCount(result.lines.size()));
    if (m_lblResInfo) {
        m_lblResInfo->setText(QStringLiteral("OCR %1 | %2 | %3 | threads=%4 | batch=%5 | io=%6ms | pipe=%7ms | result=%8ms")
            .arg(result.device.isEmpty() ? QStringLiteral("unknown") : result.device)
            .arg(result.runMode.isEmpty() ? QStringLiteral("paddle") : result.runMode)
            .arg(result.mkldnnEnabled ? QStringLiteral("mkldnn:on") : QStringLiteral("mkldnn:off"))
            .arg(result.cpuThreads)
            .arg(result.recBatchSize)
            .arg(result.ioMs, 0, 'f', 1)
            .arg(result.pipelineMs, 0, 'f', 1)
            .arg(result.resultMs, 0, 'f', 1));
    }

    m_frameCount++;
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    qint64 dt = now - m_fpsTimestamp;
    if (dt >= 500) {
        m_currentFps = m_frameCount * 1000.0 / dt;
        m_lblFps->setText(QString::number(m_currentFps, 'f', 1));
        m_frameCount = 0;
        m_fpsTimestamp = now;
    }

    updateOcrList(result);
}

void InferenceTab::onAnomalyMetadataReady(AnomalyMetadata meta)
{
    m_anomalyMeta = meta;
    m_anomalyMetaLoaded = true;
    if (m_lblMetadataInfo) {
        m_lblMetadataInfo->setText(QStringLiteral("已加载：%1 · score[%2,%3] · map[%4,%5]")
            .arg(meta.summary())
            .arg(meta.scoreMin, 0, 'f', 3)
            .arg(meta.scoreMax, 0, 'f', 3)
            .arg(meta.mapMin, 0, 'f', 3)
            .arg(meta.mapMax, 0, 'f', 3));
    }
    updateTaskUi();
}

void InferenceTab::paintOverlay(QImage& img, const QVector<Detection>& dets)
{
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing, false);

    // 标签字号随图像短边缩放（640px 基准约 16px），避免大图上过小
    const int labelFontPx = qBound(16, qMin(img.width(), img.height()) / 40, 40);
    const int boxPenW = qMax(2, labelFontPx / 7);

    QFont f = p.font();
    f.setPixelSize(labelFontPx);
    f.setBold(true);
    p.setFont(f);

    for (const auto& d : dets) {
        QColor col = m_labels->classColor(d.classLabel);

        if (d.hasMask && d.maskW > 0 && d.maskH > 0 && !d.maskData.isEmpty()) {
            // 与 5-reference/YOLOv8-seg/src/segment.cpp:plot_mask 等价：
            // pixel = pixel*(1-beta) + color*beta，其中 beta = 0.5 * gray/255
            // 用 ARGB 软 alpha 由 QPainter 的 SourceOver 完成同样的线性混合，
            // 最大不透明度 50%（gray=255 时 alpha=127）。
            QImage maskImg(d.maskW, d.maskH, QImage::Format_ARGB32_Premultiplied);
            const uchar* src = reinterpret_cast<const uchar*>(d.maskData.constData());
            const int cr = col.red();
            const int cg = col.green();
            const int cb = col.blue();
            for (int yy = 0; yy < d.maskH; ++yy) {
                QRgb* dst = reinterpret_cast<QRgb*>(maskImg.scanLine(yy));
                const uchar* s = src + yy * d.maskW;
                for (int xx = 0; xx < d.maskW; ++xx) {
                    // beta = 0.5 * gray/255，对应 alpha = gray/2
                    const int a = s[xx] >> 1;
                    if (a == 0) {
                        dst[xx] = 0u;
                    } else {
                        // Premultiplied: 各通道需 * alpha / 255
                        const int pr = (cr * a + 127) / 255;
                        const int pg = (cg * a + 127) / 255;
                        const int pb = (cb * a + 127) / 255;
                        dst[xx] = qRgba(pr, pg, pb, a);
                    }
                }
            }
            p.drawImage(QRect(d.maskX, d.maskY, d.maskW, d.maskH), maskImg);
        }

        p.setPen(QPen(col, boxPenW));
        p.setBrush(Qt::NoBrush);

        // OBB: 画 4 角点旋转矩形；其它任务画 AABB
        QPointF labelAnchor;
        if (d.hasOBB) {
            const float ca = std::cos(d.obbAngle);
            const float sa = std::sin(d.obbAngle);
            const float hw = d.obbOw * 0.5f;
            const float hh = d.obbOh * 0.5f;
            const QPointF local[4] = {
                {-hw, -hh}, { hw, -hh}, { hw,  hh}, {-hw,  hh}
            };
            QPolygonF poly;
            float minY = 1e9f; QPointF topPt(d.obbCx, d.obbCy);
            for (int i = 0; i < 4; ++i) {
                float x = d.obbCx + local[i].x() * ca - local[i].y() * sa;
                float y = d.obbCy + local[i].x() * sa + local[i].y() * ca;
                poly << QPointF(x, y);
                if (y < minY) { minY = y; topPt = QPointF(x, y); }
            }
            p.drawPolygon(poly);
            labelAnchor = topPt;
        } else {
            QRectF rc(d.left, d.top, d.right - d.left, d.bottom - d.top);
            p.drawRect(rc);
            labelAnchor = QPointF(d.left, d.top);
        }

        QString text;
        if (d.hasOBB) {
            const float angDeg = d.obbAngle * 180.0f / 3.14159265358979323846f;
            text = QStringLiteral("%1  %2  θ%3°")
                .arg(m_labels->className(d.classLabel))
                .arg(QString::number(d.confidence, 'f', 2))
                .arg(QString::number(angDeg, 'f', 1));
        } else {
            text = QStringLiteral("%1  %2")
                .arg(m_labels->className(d.classLabel))
                .arg(QString::number(d.confidence, 'f', 2));
        }
        QFontMetrics fm(p.font());
        const int tagPadH = labelFontPx / 2;
        const int tagPadV = labelFontPx / 5;
        int tw = fm.horizontalAdvance(text) + tagPadH * 2;
        int th = fm.height() + tagPadV * 2;
        QRectF tagRc(labelAnchor.x(), labelAnchor.y() - th, tw, th);
        if (tagRc.top() < 0) tagRc.moveTop(labelAnchor.y());
        p.fillRect(tagRc, col);
        p.setPen(Qt::black);
        p.drawText(tagRc, Qt::AlignCenter, text);

        // 姿态估计：画 17 关键点 + 19 段骨架（与 triple-Mu pose demo 同源常量）
        if (d.hasKpts && d.numKpts == 17 &&
            d.kptsData.size() >= int(17 * 3 * sizeof(float))) {
            const float* kps = reinterpret_cast<const float*>(d.kptsData.constData());
            // 关键点
            for (int k = 0; k < 17; ++k) {
                float kx = kps[k * 3 + 0];
                float ky = kps[k * 3 + 1];
                float kv = kps[k * 3 + 2];
                if (kv < KPT_VIS_THRESHOLD) continue;
                QColor kc(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                p.setPen(Qt::NoPen);
                p.setBrush(kc);
                p.drawEllipse(QPointF(kx, ky), 4.0, 4.0);
            }
            // 骨架（SKELETON 用 1-based 索引）
            for (int s = 0; s < 19; ++s) {
                int a = SKELETON[s][0] - 1;
                int b = SKELETON[s][1] - 1;
                if (a < 0 || a >= 17 || b < 0 || b >= 17) continue;
                float va = kps[a * 3 + 2];
                float vb = kps[b * 3 + 2];
                if (va < KPT_VIS_THRESHOLD || vb < KPT_VIS_THRESHOLD) continue;
                QColor lc(LIMB_COLORS[s][0], LIMB_COLORS[s][1], LIMB_COLORS[s][2]);
                p.setPen(QPen(lc, 2));
                p.drawLine(QPointF(kps[a * 3], kps[a * 3 + 1]),
                           QPointF(kps[b * 3], kps[b * 3 + 1]));
            }
            p.setBrush(Qt::NoBrush);
        }
    }
}

void InferenceTab::updateDetList(const QVector<Detection>& dets)
{
    m_detListW->clear();
    int shown = 0;
    for (const auto& d : dets) {
        if (shown++ >= 30) break;
        QString text = QStringLiteral("%1  %2\n  x:%3 y:%4 w:%5 h:%6")
            .arg(m_labels->className(d.classLabel))
            .arg(QString::number(d.confidence, 'f', 2))
            .arg(int(d.left)).arg(int(d.top))
            .arg(int(d.right - d.left)).arg(int(d.bottom - d.top));
        auto* item = new QListWidgetItem(text);
        QColor col = m_labels->classColor(d.classLabel);
        item->setForeground(col);
        m_detListW->addItem(item);
    }
}

void InferenceTab::updateAnomalyList(const AnomalyResult& result)
{
    m_detListW->clear();
    auto* status = new QListWidgetItem(QStringLiteral("状态：%1\n  normalized_score=%2")
        .arg(result.statusText)
        .arg(QString::number(result.normalizedScore, 'f', 3)));
    status->setForeground(result.isAnomaly ? QColor("#FF4D6D") : QColor("#3AE675"));
    m_detListW->addItem(status);

    auto* score = new QListWidgetItem(QStringLiteral("raw_score：%1\n  image_threshold=%2")
        .arg(QString::number(result.rawScore, 'f', 4))
        .arg(QString::number(m_anomalyMeta.imageThreshold, 'f', 4)));
    score->setForeground(QColor("#FFB020"));
    m_detListW->addItem(score);

    auto* mask = new QListWidgetItem(QStringLiteral("pred_mask：pixel_threshold=%1\n  输出 image / anomaly map / red contours")
        .arg(QString::number(m_anomalyMeta.pixelThreshold, 'f', 4)));
    mask->setForeground(QColor("#9FC8E8"));
    m_detListW->addItem(mask);
}

void InferenceTab::refreshOcrCanvasHighlight()
{
    if (m_lastOcrResult.overlayImage.isNull() || !m_lblCanvas) {
        return;
    }

    QImage image = m_lastOcrResult.overlayImage.copy();
    if (m_selectedOcrLineRow >= 0 && m_selectedOcrLineRow < m_lastOcrResult.lines.size()) {
        const OcrTextLine& line = m_lastOcrResult.lines.at(m_selectedOcrLineRow);
        if (line.polygon.size() >= 3) {
            QPainter painter(&image);
            painter.setRenderHint(QPainter::Antialiasing, true);

            QPolygonF poly(line.polygon);
            const QColor highlight("#FF4D6D");
            painter.setBrush(QColor(255, 77, 109, 45));
            painter.setPen(QPen(highlight, 4, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            painter.drawPolygon(poly);

            const QRectF bounds = poly.boundingRect();
            const QString badgeText = QStringLiteral("#%1").arg(line.index + 1);
            QFont font = painter.font();
            font.setBold(true);
            font.setPointSize(12);
            painter.setFont(font);
            const QFontMetrics fm(font);
            QRectF badge(bounds.topLeft() + QPointF(0, -fm.height() - 8),
                         QSizeF(fm.horizontalAdvance(badgeText) + 14, fm.height() + 8));
            if (badge.top() < 0) {
                badge.moveTop(bounds.top());
            }
            painter.fillRect(badge, highlight);
            painter.setPen(Qt::black);
            painter.drawText(badge, Qt::AlignCenter, badgeText);
        }
    }

    static_cast<ZoomableCanvasLabel*>(m_lblCanvas)->setImage(image);
}

void InferenceTab::updateOcrList(const OcrResult& result)
{
    QSignalBlocker blocker(m_detListW);
    m_detListW->clear();
    auto* summary = new QListWidgetItem(QStringLiteral("OCR runtime: device=%1 precision=%2 run_mode=%3 mkldnn=%4 threads=%5 rec_batch=%6\n  io=%7ms pipeline=%8ms result=%9ms total=%10ms")
        .arg(result.device)
        .arg(result.precision)
        .arg(result.runMode)
        .arg(result.mkldnnEnabled ? QStringLiteral("on") : QStringLiteral("off"))
        .arg(result.cpuThreads)
        .arg(result.recBatchSize)
        .arg(result.ioMs, 0, 'f', 1)
        .arg(result.pipelineMs, 0, 'f', 1)
        .arg(result.resultMs, 0, 'f', 1)
        .arg(result.totalMs, 0, 'f', 1));
    summary->setForeground(QColor("#FFB020"));
    summary->setData(Qt::UserRole, -1);
    m_detListW->addItem(summary);
    int shown = 0;
    for (const auto& line : result.lines) {
        if (shown++ >= 80) break;
        QString poly;
        for (const auto& pt : line.polygon) {
            if (!poly.isEmpty()) poly += QStringLiteral(" ");
            poly += QStringLiteral("(%1,%2)").arg(qRound(pt.x())).arg(qRound(pt.y()));
        }
        QString text = QStringLiteral("#%1  %2\n  score=%3 angle=%4\n  poly=%5")
            .arg(line.index + 1)
            .arg(line.text)
            .arg(QString::number(line.score, 'f', 3))
            .arg(line.angle)
            .arg(poly);
        auto* item = new QListWidgetItem(text);
        item->setForeground(QColor("#9FC8E8"));
        item->setData(Qt::UserRole, shown - 1);
        m_detListW->addItem(item);
    }
    if (result.lines.isEmpty()) {
        auto* item = new QListWidgetItem(QStringLiteral("未检测到文本行\n  line_count=0"));
        item->setForeground(QColor("#FFB020"));
        item->setData(Qt::UserRole, -1);
        m_detListW->addItem(item);
    }
}

bool InferenceTab::isAnomalyMode() const
{
    return m_cbTask && m_cbTask->currentData().toInt() == -100;
}

bool InferenceTab::isOcrMode() const
{
    return m_cbTask && m_cbTask->currentData().toInt() == -200;
}

void InferenceTab::updateTaskUi()
{
    const bool anomaly = isAnomalyMode();
    const bool ocr = isOcrMode();
    if (m_lblAnomalyModelName) m_lblAnomalyModelName->setVisible(anomaly);
    if (m_cbAnomalyModel) m_cbAnomalyModel->setVisible(anomaly);
    if (m_lblOcrRecModelName) m_lblOcrRecModelName->setVisible(ocr);
    if (m_ocrRecModelPanel) m_ocrRecModelPanel->setVisible(ocr);
    if (m_lblOcrLineOrientationModelName) m_lblOcrLineOrientationModelName->setVisible(ocr);
    if (m_ocrLineOrientationModelPanel) m_ocrLineOrientationModelPanel->setVisible(ocr);
    if (m_ocrRuntimePanel) m_ocrRuntimePanel->setVisible(ocr);
    if (m_lblEnginePathName) m_lblEnginePathName->setText(ocr ? QStringLiteral("OCR 检测模型目录") : QStringLiteral(".trt 引擎路径"));
    if (m_edEngine) m_edEngine->setPlaceholderText(ocr ? QStringLiteral("选择 ch_PP-OCRv4_det_infer 模型目录") : QStringLiteral("选择 .trt 引擎文件"));
    if (m_labelPanel) m_labelPanel->setVisible(!anomaly && !ocr);
    if (m_metadataPanel) m_metadataPanel->setVisible(anomaly);
    if (m_lblConfName) m_lblConfName->setText(anomaly ? QStringLiteral("image_threshold") : (ocr ? QStringLiteral("det_thresh") : QStringLiteral("置信度阈值")));
    if (m_lblNmsName) m_lblNmsName->setText(anomaly ? QStringLiteral("pixel_threshold") : (ocr ? QStringLiteral("box_thresh") : QStringLiteral("NMS IoU 阈值")));
    if (m_lblCountName) m_lblCountName->setText(anomaly ? QStringLiteral("状态") : (ocr ? QStringLiteral("文本行") : QStringLiteral("检测数")));

    if (m_canvasStack) {
        m_canvasStack->setCurrentIndex(anomaly ? 1 : 0);
        if (!anomaly) {
            m_anomalyFullscreenIdx = -1;
            if (m_imageFrame) m_imageFrame->show();
            if (m_overlayFrame) m_overlayFrame->show();
            if (m_maskFrame) m_maskFrame->show();
        }
    }

    if (anomaly) {
        auto applyRawSlider = [](QSlider* slider, float minVal, float maxVal, float value) {
            int lo = qRound(minVal * 100.0f);
            int hi = qRound(maxVal * 100.0f);
            if (hi <= lo) {
                lo = qRound((value - 1.0f) * 100.0f);
                hi = qRound((value + 1.0f) * 100.0f);
            }
            const int current = qBound(lo, qRound(value * 100.0f), hi);
            QSignalBlocker blocker(slider);
            slider->setRange(lo, hi);
            slider->setValue(current);
        };

        applyRawSlider(m_slConf, m_anomalyMeta.scoreMin, m_anomalyMeta.scoreMax, m_anomalyMeta.imageThreshold);
        applyRawSlider(m_slNms, m_anomalyMeta.mapMin, m_anomalyMeta.mapMax, m_anomalyMeta.pixelThreshold);
        m_lblConfVal->setText(QString::number(m_anomalyMeta.imageThreshold, 'f', 2));
        m_lblNmsVal->setText(QString::number(m_anomalyMeta.pixelThreshold, 'f', 2));
        m_slConf->setEnabled(true);
        m_slNms->setEnabled(true);
        m_lblDetCount->setText("-");
    } else {
        m_lblDetCount->setStyleSheet("");
        m_slConf->setRange(1, 99);
        m_slNms->setRange(1, 99);
        m_slConf->setEnabled(true);
        m_slNms->setEnabled(true);
        m_lblConfVal->setText(QString::number(m_slConf->value() / 100.0, 'f', 2));
        m_lblNmsVal->setText(QString::number(m_slNms->value() / 100.0, 'f', 2));
    }
}

void InferenceTab::toggleAnomalyFullscreen(int idx)
{
    if (m_anomalyFullscreenIdx == idx) {
        m_imageFrame->show();
        m_overlayFrame->show();
        m_maskFrame->show();
        m_anomalyFullscreenIdx = -1;
    } else {
        m_imageFrame->setVisible(idx == 0);
        m_overlayFrame->setVisible(idx == 1);
        m_maskFrame->setVisible(idx == 2);
        m_anomalyFullscreenIdx = idx;
    }
}

void InferenceTab::setRunningUi(bool running)
{
    m_btnStart->setEnabled(!running);
    m_btnStop->setEnabled(running);
    m_edEngine->setEnabled(!running);
    m_cbTask->setEnabled(!running);
    m_btnSrcImg->setEnabled(!running);
    m_btnSrcVid->setEnabled(!running);
    m_btnSrcCam->setEnabled(!running);
    m_btnConnect->setEnabled(!running);
    m_btnScanCam->setEnabled(!running);
    m_cbCamera->setEnabled(!running);
    m_edFilePath->setEnabled(!running);
    m_btnBrowseFile->setEnabled(!running);
    if (m_edMetadata) m_edMetadata->setEnabled(!running);
    if (m_edOcrRecModel) m_edOcrRecModel->setEnabled(!running);
    if (m_btnBrowseOcrRecModel) m_btnBrowseOcrRecModel->setEnabled(!running);
    if (m_edOcrLineOrientationModel) m_edOcrLineOrientationModel->setEnabled(!running);
    if (m_btnBrowseOcrLineOrientationModel) m_btnBrowseOcrLineOrientationModel->setEnabled(!running);
    if (m_cbOcrDevice) m_cbOcrDevice->setEnabled(!running);
    if (m_cbOcrPrecision) m_cbOcrPrecision->setEnabled(!running);
    if (m_chkOcrMkldnn) m_chkOcrMkldnn->setEnabled(!running);
    if (m_spOcrThreads) m_spOcrThreads->setEnabled(!running);
    if (m_spOcrBatch) m_spOcrBatch->setEnabled(!running);
    if (m_edLabel) m_edLabel->setEnabled(!running);
    if (!running) updateTaskUi();
}

void InferenceTab::showLoadingProgress()
{
    hideLoadingProgress();

    m_loadingProgress = new QProgressDialog(QStringLiteral("正在加载模型，请稍候..."),
                                            QStringLiteral("停止"),
                                            0,
                                            0,
                                            this);
    m_loadingProgress->setWindowTitle(QStringLiteral("模型加载中"));
    m_loadingProgress->setWindowModality(Qt::WindowModal);
    m_loadingProgress->setMinimumDuration(0);
    m_loadingProgress->setAutoClose(false);
    m_loadingProgress->setAutoReset(false);
    connect(m_loadingProgress, &QProgressDialog::canceled,
            this, &InferenceTab::onStopClicked);
    m_loadingProgress->show();
    m_loadingProgress->raise();
    m_loadingProgress->repaint();
    QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents, 50);
}

void InferenceTab::hideLoadingProgress()
{
    if (!m_loadingProgress) {
        return;
    }
    disconnect(m_loadingProgress, nullptr, this, nullptr);
    m_loadingProgress->hide();
    m_loadingProgress->deleteLater();
    m_loadingProgress = nullptr;
}

void InferenceTab::startCaptureAfterEngineLoaded()
{
    if (!m_infWorker && !m_anomalyWorker && !m_ocrWorker) {
        m_inferenceActive = false;
        setRunningUi(false);
        return;
    }

    stopCaptureThread();

    m_capThread = new QThread(this);
    m_capWorker = new CaptureWorker();
    m_capWorker->moveToThread(m_capThread);

    SourceType st = (m_pendingSrcId == 0) ? SourceType::Image
                  : (m_pendingSrcId == 1) ? SourceType::Video
                                          : SourceType::Camera;
    m_capWorker->setSource(st, m_pendingSourcePath, m_pendingCamBackend);

    if (m_anomalyWorker) {
        connect(m_capWorker, &CaptureWorker::frameReady,
                m_anomalyWorker, &AnomalyInferWorker::onFrame, Qt::QueuedConnection);
    } else if (m_ocrWorker) {
        connect(m_capWorker, &CaptureWorker::frameReady,
                m_ocrWorker, &OcrInferWorker::onFrame, Qt::QueuedConnection);
    } else {
        connect(m_capWorker, &CaptureWorker::frameReady,
                m_infWorker, &InferWorker::onFrame, Qt::QueuedConnection);
    }
    connect(m_capWorker, &CaptureWorker::errorOccurred,
            this, &InferenceTab::onCaptureError, Qt::QueuedConnection);
    connect(m_capWorker, &CaptureWorker::sourceInfo,
            this, &InferenceTab::onSourceInfo, Qt::QueuedConnection);
    connect(m_capWorker, &CaptureWorker::finished,
            this, &InferenceTab::onCaptureFinished, Qt::QueuedConnection);
    connect(m_capThread, &QThread::started, m_capWorker, &CaptureWorker::start);
    connect(m_capThread, &QThread::finished, m_capWorker, &QObject::deleteLater);

    m_frameCount = 0;
    m_fpsTimestamp = QDateTime::currentMSecsSinceEpoch();
    m_currentFps = 0;
    if (st == SourceType::Camera) m_lblLive->show(); else m_lblLive->hide();
    m_inferenceActive = true;
    setRunningUi(true);

    m_capThread->start();
}

void InferenceTab::cleanupThreads()
{
    hideLoadingProgress();
    stopCaptureThread();
    releaseInferenceWorker();
}

void InferenceTab::stopCaptureThread()
{
    // 关键：先把 worker → UI 的所有连接全部断开，否则旧 worker 退出时
    // 排队到主线程的 finished()/errorOccurred() 等信号会在下一轮启动后才被
    // 处理，把刚 setRunningUi(true) 启用的停止按钮再次置灰。
    m_inferenceActive = false;
    if (m_capWorker) {
        disconnect(m_capWorker, nullptr, this, nullptr);
        if (m_infWorker) disconnect(m_capWorker, nullptr, m_infWorker, nullptr);
        if (m_anomalyWorker) disconnect(m_capWorker, nullptr, m_anomalyWorker, nullptr);
        if (m_ocrWorker) disconnect(m_capWorker, nullptr, m_ocrWorker, nullptr);
    }
    if (m_capWorker) m_capWorker->stop();
    if (m_capThread) {
        m_capThread->quit();
        // 分段等待并 processEvents，避免 UI 长时间冻结
        for (int i = 0; i < 10 && !m_capThread->wait(100); ++i) {
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents, 50);
        }
        if (m_capThread->isRunning()) m_capThread->terminate();
        m_capThread->wait(200);
        delete m_capThread;
        m_capThread = nullptr;
        m_capWorker = nullptr;
    }
}

void InferenceTab::releaseInferenceWorker()
{
    if (m_infWorker) {
        disconnect(m_infWorker, nullptr, this, nullptr);
    }
    if (m_anomalyWorker) {
        disconnect(m_anomalyWorker, nullptr, this, nullptr);
    }
    if (m_ocrWorker) {
        disconnect(m_ocrWorker, nullptr, this, nullptr);
    }
    if (m_infThread) {
        m_infThread->quit();
        for (int i = 0; i < 10 && !m_infThread->wait(100); ++i) {
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents, 50);
        }
        if (m_infThread->isRunning()) m_infThread->terminate();
        m_infThread->wait(200);
        delete m_infThread;
        m_infThread = nullptr;
        m_infWorker = nullptr;
        m_anomalyWorker = nullptr;
        m_ocrWorker = nullptr;
    }
    m_loadedModelKey.clear();
    m_pendingModelKey.clear();
}

QString InferenceTab::currentModelKey() const
{
    auto normalizedPath = [](const QString& path) {
        const QFileInfo info(path.trimmed());
        return QDir::fromNativeSeparators(info.absoluteFilePath()).toLower();
    };

    const int task = m_cbTask ? m_cbTask->currentData().toInt() : 0;
    if (isAnomalyMode()) {
        return QStringLiteral("anomaly|engine=%1|metadata=%2|model=%3|image_thr=%4|pixel_thr=%5")
            .arg(normalizedPath(m_edEngine->text()))
            .arg(normalizedPath(m_edMetadata->text()))
            .arg(m_cbAnomalyModel ? m_cbAnomalyModel->currentText().trimmed().toLower() : QStringLiteral("auto"))
            .arg(QString::number(m_slConf->value() / 100.0, 'f', 4))
            .arg(QString::number(m_slNms->value() / 100.0, 'f', 4));
    }
    if (isOcrMode()) {
        return QStringLiteral("ocr|det=%1|rec=%2|line_ori=%3|det_thr=%4|box_thr=%5|device=%6|precision=%7|mkldnn=%8|threads=%9|batch=%10")
            .arg(normalizedPath(m_edEngine->text()))
            .arg(normalizedPath(m_edOcrRecModel ? m_edOcrRecModel->text() : QString()))
            .arg(normalizedPath(m_edOcrLineOrientationModel ? m_edOcrLineOrientationModel->text() : QString()))
            .arg(QString::number(m_slConf->value() / 100.0, 'f', 4))
            .arg(QString::number(m_slNms->value() / 100.0, 'f', 4))
            .arg(m_cbOcrDevice ? m_cbOcrDevice->currentData().toString() : QStringLiteral("cpu"))
            .arg(m_cbOcrPrecision ? m_cbOcrPrecision->currentData().toString() : QStringLiteral("fp32"))
            .arg(m_chkOcrMkldnn && m_chkOcrMkldnn->isChecked() ? 1 : 0)
            .arg(m_spOcrThreads ? m_spOcrThreads->value() : 8)
            .arg(m_spOcrBatch ? m_spOcrBatch->value() : 6);
    }

    return QStringLiteral("yolo|task=%1|engine=%2|conf=%3|nms=%4")
        .arg(task)
        .arg(normalizedPath(m_edEngine->text()))
        .arg(QString::number(m_slConf->value() / 100.0, 'f', 4))
        .arg(QString::number(m_slNms->value() / 100.0, 'f', 4));
}

bool InferenceTab::hasReusableWorker(const QString& modelKey) const
{
    if (modelKey.isEmpty() || modelKey != m_loadedModelKey || !m_infThread || !m_infThread->isRunning()) {
        return false;
    }

    if (isAnomalyMode()) {
        return m_anomalyWorker && m_anomalyWorker->isLoaded();
    }
    if (isOcrMode()) {
        return m_ocrWorker && m_ocrWorker->isLoaded();
    }
    return m_infWorker && m_infWorker->isLoaded();
}

} // namespace trtforge
