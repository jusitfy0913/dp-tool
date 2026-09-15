#include "TrainingTab.h"

#include "YoloTrainController.h"
#include "common/AppEventFormatter.h"
#include "common/AppEvents.h"
#include "common/AppFileLogger.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSizePolicy>
#include <QSplitter>
#include <QSpinBox>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QTextBrowser>
#include <QVBoxLayout>
#include <QVector>
#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>

namespace trtforge {

class MonitorChartWidget : public QWidget
{
public:
    enum class Mode { Line, Image };

    explicit MonitorChartWidget(QString title, QColor color, Mode mode = Mode::Line, QWidget* parent = nullptr)
        : QWidget(parent), m_title(std::move(title)), m_color(std::move(color)), m_mode(mode)
    {
        setMinimumSize(220, 120);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setToolTip("双击放大，再双击恢复多图显示");
    }

    void setDoubleClickHandler(std::function<void()> handler)
    {
        m_doubleClickHandler = std::move(handler);
    }

    void addPoint(int epoch, double value)
    {
        if (m_mode != Mode::Line) return;
        if (epoch <= 0) return;
        m_points.push_back(QPointF(epoch, value));
        update();
    }

    void setFixedYRange(double minY, double maxY, QString unit = {})
    {
        m_hasFixedYRange = true;
        m_fixedMinY = minY;
        m_fixedMaxY = maxY;
        m_unit = std::move(unit);
        update();
    }

    void clear()
    {
        m_points.clear();
        m_artifactPath.clear();
        m_image = QImage();
        update();
    }

    void setArtifactPath(const QString& path)
    {
        m_artifactPath = path;
        if (m_mode == Mode::Image && !path.isEmpty()) {
            m_image = QImage(path);
        }
        setToolTip(path.isEmpty() ? "双击放大，再双击恢复多图显示" : "图表产物：" + path + "\n双击放大，再双击恢复");
        update();
    }

protected:
    void mouseDoubleClickEvent(QMouseEvent* event) override
    {
        if (event->button() == Qt::LeftButton && m_doubleClickHandler) {
            m_doubleClickHandler();
        }
        QWidget::mouseDoubleClickEvent(event);
    }

    void paintEvent(QPaintEvent*) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor("#07111F"));

        painter.setPen(QColor("#D6F3FF"));
        painter.drawText(rect().adjusted(10, 6, -10, -6), Qt::AlignTop | Qt::AlignLeft, m_title);

        if (m_mode == Mode::Image) {
            paintImageMode(painter);
            return;
        }

        const bool enlarged = width() > 520 || height() > 260;
        const int leftMargin = enlarged ? 64 : 52;
        const int bottomMargin = enlarged ? 42 : 34;
        const QRect plot = rect().adjusted(leftMargin, 34, -18, -bottomMargin);
        painter.setPen(QPen(QColor("#1F3B55"), 1));
        painter.drawRect(plot);

        if (m_points.isEmpty()) {
            painter.setPen(QColor("#64748B"));
            painter.drawText(plot, Qt::AlignCenter, "等待 epoch 数据");
            painter.setPen(QColor("#83A7C4"));
            painter.drawText(rect().adjusted(10, 0, -10, -8), Qt::AlignBottom | Qt::AlignLeft, "X: epoch   Y: value");
            return;
        }

        double minY = m_hasFixedYRange ? m_fixedMinY : m_points.front().y();
        double maxY = m_hasFixedYRange ? m_fixedMaxY : minY;
        int minX = static_cast<int>(m_points.front().x());
        int maxX = minX;
        for (const auto& point : m_points) {
            minX = std::min(minX, static_cast<int>(point.x()));
            maxX = std::max(maxX, static_cast<int>(point.x()));
            if (!m_hasFixedYRange) {
                minY = std::min(minY, point.y());
                maxY = std::max(maxY, point.y());
            }
        }
        if (maxX == minX) maxX = minX + 1;
        if (qFuzzyCompare(minY, maxY)) {
            minY -= 1.0;
            maxY += 1.0;
        } else if (!m_hasFixedYRange) {
            const double padding = (maxY - minY) * 0.12;
            minY -= padding;
            maxY += padding;
            if (minY > 0.0 && minY - padding < 0.0) minY = 0.0;
        }

        paintAxes(painter, plot, minX, maxX, minY, maxY, enlarged);

        auto mapPoint = [&](const QPointF& point) {
            const double nx = (point.x() - minX) / static_cast<double>(maxX - minX);
            const double ny = (point.y() - minY) / (maxY - minY);
            return QPointF(plot.left() + nx * plot.width(), plot.bottom() - ny * plot.height());
        };

        QPainterPath path;
        path.moveTo(mapPoint(m_points.front()));
        for (int i = 1; i < m_points.size(); ++i) {
            path.lineTo(mapPoint(m_points[i]));
        }
        painter.setPen(QPen(m_color, 2.2));
        painter.drawPath(path);

        painter.setBrush(m_color);
        painter.setPen(Qt::NoPen);
        const QPointF last = mapPoint(m_points.back());
        painter.drawEllipse(last, 4, 4);

        painter.setPen(QColor("#83A7C4"));
        painter.drawText(rect().adjusted(10, 0, -10, -8), Qt::AlignBottom | Qt::AlignLeft,
                         QString("epoch %1  value %2%3")
                             .arg(static_cast<int>(m_points.back().x()))
                             .arg(m_points.back().y(), 0, 'f', valueDecimals(maxY - minY))
                             .arg(m_unit));
        if (!m_artifactPath.isEmpty()) {
            painter.drawText(rect().adjusted(10, 0, -10, -8), Qt::AlignBottom | Qt::AlignRight, "PNG 已生成");
        }
    }

private:
    int valueDecimals(double range) const
    {
        if (range >= 100.0) return 0;
        if (range >= 10.0) return 1;
        return 3;
    }

    void paintAxes(QPainter& painter, const QRect& plot, int minX, int maxX, double minY, double maxY, bool enlarged) const
    {
        const int yTicks = enlarged ? 6 : 4;
        const int xTicks = enlarged ? 6 : 4;
        const QColor gridColor("#18354D");
        const QColor axisColor("#83A7C4");
        const int decimals = valueDecimals(maxY - minY);

        painter.setPen(QPen(gridColor, 1));
        for (int i = 0; i <= yTicks; ++i) {
            const double ratio = i / static_cast<double>(yTicks);
            const int y = plot.bottom() - static_cast<int>(ratio * plot.height());
            painter.drawLine(plot.left(), y, plot.right(), y);
            const double value = minY + ratio * (maxY - minY);
            painter.setPen(axisColor);
            painter.drawText(QRect(4, y - 9, plot.left() - 8, 18), Qt::AlignRight | Qt::AlignVCenter,
                             QString::number(value, 'f', decimals) + m_unit);
            painter.setPen(QPen(gridColor, 1));
        }

        int lastEpoch = -1;
        for (int i = 0; i <= xTicks; ++i) {
            const double ratio = i / static_cast<double>(xTicks);
            const int x = plot.left() + static_cast<int>(ratio * plot.width());
            const int epoch = qRound(minX + ratio * (maxX - minX));
            if (epoch == lastEpoch && i != xTicks) continue;
            lastEpoch = epoch;
            painter.setPen(QPen(gridColor, 1));
            painter.drawLine(x, plot.top(), x, plot.bottom());
            painter.setPen(axisColor);
            painter.drawText(QRect(x - 26, plot.bottom() + 4, 52, 18), Qt::AlignCenter, QString::number(epoch));
        }

        painter.setPen(axisColor);
        painter.drawText(QRect(plot.left(), rect().bottom() - 22, plot.width(), 16), Qt::AlignCenter, "epoch");
    }

    void paintImageMode(QPainter& painter)
    {
        const QRect imageRect = rect().adjusted(16, 34, -16, -30);
        painter.setPen(QPen(QColor("#1F3B55"), 1));
        painter.drawRect(imageRect);
        if (m_image.isNull()) {
            painter.setPen(QColor("#64748B"));
            painter.drawText(imageRect, Qt::AlignCenter, "等待验证后生成混淆矩阵");
        } else {
            const QImage scaled = m_image.scaled(imageRect.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            const QPoint topLeft(imageRect.center().x() - scaled.width() / 2, imageRect.center().y() - scaled.height() / 2);
            painter.drawImage(topLeft, scaled);
        }
        painter.setPen(QColor("#83A7C4"));
        painter.drawText(rect().adjusted(10, 0, -10, -8), Qt::AlignBottom | Qt::AlignLeft,
                         m_artifactPath.isEmpty() ? "等待 PNG 产物" : "混淆矩阵 PNG 已生成");
        painter.drawText(rect().adjusted(10, 0, -10, -8), Qt::AlignBottom | Qt::AlignRight, "双击放大");
    }

    QString m_title;
    QColor m_color;
    Mode m_mode { Mode::Line };
    QVector<QPointF> m_points;
    QString m_artifactPath;
    QString m_unit;
    bool m_hasFixedYRange { false };
    double m_fixedMinY { 0.0 };
    double m_fixedMaxY { 1.0 };
    QImage m_image;
    std::function<void()> m_doubleClickHandler;
};

namespace {

QFrame* makeCard()
{
    auto* card = new QFrame();
    card->setObjectName("card");
    card->setFrameShape(QFrame::NoFrame);
    return card;
}

QLabel* makeTitle(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName("cardTitle");
    return label;
}

QLabel* makeField(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName("fieldLabel");
    return label;
}

void keepFormControlHeight(QWidget* widget)
{
    if (!widget) return;
    widget->setMinimumHeight(28);
    widget->setMaximumHeight(32);
}

void applyPanelScrollStyle(QScrollArea* scroll, QWidget* content)
{
    if (!scroll) return;
    scroll->setObjectName("panelScroll");
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    scroll->viewport()->setAutoFillBackground(false);
    if (content) {
        content->setObjectName("panelScrollContent");
        content->setAutoFillBackground(false);
    }
}

void keepInfoLabelReadable(QLabel* label)
{
    if (!label) return;
    label->setWordWrap(true);
    label->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    label->setMinimumHeight(18);
}

QString jsonValueToText(const QJsonValue& value)
{
    if (value.isString()) return value.toString();
    if (value.isDouble()) return QString::number(value.toDouble());
    if (value.isBool()) return value.toBool() ? "true" : "false";
    if (value.isArray()) {
        QStringList items;
        const auto array = value.toArray();
        for (const QJsonValue& item : array) items << jsonValueToText(item);
        return items.join(", ");
    }
    if (value.isObject()) {
        return QString::fromUtf8(QJsonDocument(value.toObject()).toJson(QJsonDocument::Compact));
    }
    return {};
}

QString defaultModelAlias(const QString& task)
{
    if (task == "segment") return "yolov8n-seg.pt";
    if (task == "pose") return "yolov8n-pose.pt";
    if (task == "obb") return "yolov8n-obb.pt";
    return "yolov8n.pt";
}

} // namespace

TrainingTab::TrainingTab(QWidget* parent) : QWidget(parent)
{
    m_controller = new YoloTrainController(this);
    connect(m_controller, &YoloTrainController::logLine, this, &TrainingTab::onLogLine);
    connect(m_controller, &YoloTrainController::structuredEvent, this, &TrainingTab::onStructuredEvent);
    connect(m_controller, &YoloTrainController::processStarted, this, &TrainingTab::onProcessStarted);
    connect(m_controller, &YoloTrainController::processFinished, this, &TrainingTab::onProcessFinished);
    connect(m_controller, &YoloTrainController::processError, this, &TrainingTab::onProcessError);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(14);

    m_titleLabel = new QLabel("YOLO TRAINING LAB · 训练 / 验证 / ONNX 导出");
    m_titleLabel->setObjectName("titleMain");
    root->addWidget(m_titleLabel);

    m_tabs = new QTabWidget();
    m_tabs->addTab(buildConfigTab(), "训练配置 / 日志");
    m_tabs->addTab(buildMonitorTab(), "训练监控");
    root->addWidget(m_tabs, 1);

    loadSettings();
    if (m_edPython->text().trimmed().isEmpty()) {
        m_edPython->setText(m_controller->resolvePythonPath());
        savePythonPath(m_edPython->text());
    }
    updateInitHint();
    updateModeUi();
}

QWidget* TrainingTab::buildConfigTab()
{
    auto* page = new QWidget();
    auto* root = new QVBoxLayout(page);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(14);

    auto* topSplitter = new QSplitter(Qt::Horizontal);
    topSplitter->setChildrenCollapsible(false);
    topSplitter->setHandleWidth(8);
    topSplitter->addWidget(buildTaskCard());
    topSplitter->addWidget(buildParamCard());
    topSplitter->addWidget(buildLogCard());
    topSplitter->setStretchFactor(0, 3);
    topSplitter->setStretchFactor(1, 7);
    topSplitter->setStretchFactor(2, 3);
    topSplitter->setSizes({ 390, 760, 340 });

    auto* mainSplitter = new QSplitter(Qt::Vertical);
    mainSplitter->setChildrenCollapsible(false);
    mainSplitter->setHandleWidth(8);
    mainSplitter->addWidget(topSplitter);
    mainSplitter->addWidget(buildEnvCard());
    mainSplitter->setStretchFactor(0, 6);
    mainSplitter->setStretchFactor(1, 1);
    mainSplitter->setSizes({ 640, 140 });
    root->addWidget(mainSplitter, 1);
    return page;
}

QWidget* TrainingTab::buildMonitorTab()
{
    auto* page = new QWidget();
    auto* root = new QHBoxLayout(page);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(14);

    auto* tableCard = makeCard();
    tableCard->setFixedWidth(460);
    auto* tableLayout = new QVBoxLayout(tableCard);
    tableLayout->setContentsMargins(16, 16, 16, 16);
    tableLayout->setSpacing(10);
    tableLayout->addWidget(makeTitle("01 / 实时指标数表"));
    auto* tableHint = makeField("每个 epoch 结束追加一行；开始训练前自动清空旧数据。");
    tableHint->setWordWrap(true);
    tableLayout->addWidget(tableHint);

    m_metricTable = new QTableWidget(0, 10);
    m_metricTable->setHorizontalHeaderLabels({
        "Epoch", "Box", "Cls", "Dfl", "P", "R", "mAP50", "mAP50-95", "LR", "GPU"
    });
    m_metricTable->verticalHeader()->setVisible(false);
    m_metricTable->horizontalHeader()->setStretchLastSection(true);
    m_metricTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_metricTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_metricTable->setSelectionMode(QAbstractItemView::NoSelection);
    m_metricTable->setAlternatingRowColors(true);
    m_metricTable->setStyleSheet(
        "QTableWidget { background:#020617; color:#D6F3FF; gridline-color:#1E293B; alternate-background-color:#0B2238; }"
        "QHeaderView::section { background:#123A5A; color:#DFF7FF; border:1px solid #1F4F70; padding:4px; }"
        "QTableCornerButton::section { background:#123A5A; border:1px solid #1F4F70; }"
        "QTableWidget::item { padding:4px; }");
    tableLayout->addWidget(m_metricTable, 1);
    root->addWidget(tableCard, 0);

    auto* plotCard = makeCard();
    auto* plotLayout = new QVBoxLayout(plotCard);
    plotLayout->setContentsMargins(16, 16, 16, 16);
    plotLayout->setSpacing(10);
    plotLayout->addWidget(makeTitle("02 / 最新训练图表"));
    auto* plotHint = makeField("每个 epoch 更新曲线控件；双击单个小图可在本区域放大，再双击恢复多图。PNG 产物只作为路径记录。");
    plotHint->setWordWrap(true);
    plotLayout->addWidget(plotHint);
    m_chartGrid = new QGridLayout();
    m_chartGrid->setHorizontalSpacing(10);
    m_chartGrid->setVerticalSpacing(10);
    addPlotCard(m_chartGrid, 0, 0, "BoxF1_curve", "BoxF1_curve / F1");
    addPlotCard(m_chartGrid, 0, 1, "BoxPR_curve", "BoxPR_curve / mAP50");
    addPlotCard(m_chartGrid, 1, 0, "BoxP_curve", "BoxP_curve / Precision");
    addPlotCard(m_chartGrid, 1, 1, "BoxR_curve", "BoxR_curve / Recall");
    addPlotCard(m_chartGrid, 2, 0, "confusion_matrix", "Confusion Matrix");
    addPlotCard(m_chartGrid, 2, 1, "confusion_matrix_normalized", "Confusion Matrix Normalized");
    addPlotCard(m_chartGrid, 3, 0, "labels", "GPU Memory");
    addPlotCard(m_chartGrid, 3, 1, "results", "mAP50-95 / Results");
    plotLayout->addLayout(m_chartGrid, 1);
    root->addWidget(plotCard, 1);

    auto* sideCard = makeCard();
    sideCard->setFixedWidth(320);
    auto* sideLayout = new QVBoxLayout(sideCard);
    sideLayout->setContentsMargins(16, 16, 16, 16);
    sideLayout->setSpacing(12);
    sideLayout->addWidget(makeTitle("03 / GPU 与标签 JSON"));
    m_lblGpu = new QLabel("GPU：等待训练采集");
    m_lblGpu->setObjectName("fieldLabel");
    m_lblGpu->setWordWrap(true);
    sideLayout->addWidget(m_lblGpu);

    auto* gpuBarStyle =
        "QProgressBar { background:#020617; color:#D6F3FF; border:1px solid #1E3A56; border-radius:6px; text-align:center; height:18px; }"
        "QProgressBar::chunk { background:#22D3EE; border-radius:5px; }";
    sideLayout->addWidget(makeField("显存占用"));
    m_barGpuMemory = new QProgressBar();
    m_barGpuMemory->setRange(0, 100);
    m_barGpuMemory->setValue(0);
    m_barGpuMemory->setFormat("等待采集");
    m_barGpuMemory->setStyleSheet(gpuBarStyle);
    sideLayout->addWidget(m_barGpuMemory);

    sideLayout->addWidget(makeField("GPU 利用率"));
    m_barGpuUtil = new QProgressBar();
    m_barGpuUtil->setRange(0, 100);
    m_barGpuUtil->setValue(0);
    m_barGpuUtil->setFormat("等待采集");
    m_barGpuUtil->setStyleSheet(gpuBarStyle);
    sideLayout->addWidget(m_barGpuUtil);

    m_lblLabelsJson = new QLabel("标签 JSON：等待训练结束生成 labels_<实验名称>.json");
    m_lblLabelsJson->setObjectName("fieldLabel");
    m_lblLabelsJson->setWordWrap(true);
    sideLayout->addWidget(m_lblLabelsJson);
    auto* rule = makeField("规则：不覆盖 labels_coco.json；同名时自动追加时间戳。");
    rule->setWordWrap(true);
    sideLayout->addWidget(rule);
    sideLayout->addStretch(1);
    root->addWidget(sideCard, 0);
    return page;
}

QWidget* TrainingTab::makeDirectoryRow(QLineEdit* edit, const QString& title, const char* slot)
{
    auto* wrapper = new QWidget();
    auto* layout = new QVBoxLayout(wrapper);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);
    layout->addWidget(makeField(title));
    auto* row = new QHBoxLayout();
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(6);
    auto* button = new QPushButton("目录");
    button->setProperty("ghost", true);
    keepFormControlHeight(edit);
    keepFormControlHeight(button);
    connect(button, SIGNAL(clicked()), this, slot);
    row->addWidget(edit, 1);
    row->addWidget(button);
    layout->addLayout(row);
    return wrapper;
}

QWidget* TrainingTab::buildTaskCard()
{
    auto* card = makeCard();
    card->setMinimumWidth(320);
    card->setMinimumHeight(360);
    card->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* outer = new QVBoxLayout(card);
    outer->setContentsMargins(16, 16, 16, 16);
    outer->setSpacing(10);
    outer->addWidget(makeTitle("01 / 任务与数据"));

    auto* formWidget = new QWidget();
    auto* v = new QVBoxLayout(formWidget);
    v->setContentsMargins(0, 0, 6, 0);
    v->setSpacing(10);
    v->setSizeConstraint(QLayout::SetMinAndMaxSize);
    auto* scroll = new QScrollArea();
    applyPanelScrollStyle(scroll, formWidget);
    scroll->setWidget(formWidget);
    outer->addWidget(scroll, 1);

    m_edPython = new QLineEdit();
    keepFormControlHeight(m_edPython);
    connect(m_edPython, &QLineEdit::textChanged, this, [this](const QString& text) {
        if (!m_loadingSettings) savePythonPath(text);
    });
    auto* pyBtn = new QPushButton("选择 Python");
    pyBtn->setProperty("ghost", true);
    keepFormControlHeight(pyBtn);
    connect(pyBtn, &QPushButton::clicked, this, &TrainingTab::onBrowsePython);
    auto* pyRow = new QHBoxLayout();
    pyRow->addWidget(m_edPython, 1);
    pyRow->addWidget(pyBtn);
    v->addWidget(makeField("Python 环境"));
    v->addLayout(pyRow);

    m_cbTrainType = new QComboBox();
    keepFormControlHeight(m_cbTrainType);
    m_cbTrainType->addItem("YOLO 目标检测/分割/姿态/OBB", "yolo");
    m_cbTrainType->addItem("Anomalib 异常检测", "anomalib");
    connect(m_cbTrainType, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] { onTrainTypeChanged(); });
    v->addWidget(makeField("训练类型"));
    v->addWidget(m_cbTrainType);

    m_yoloTaskBox = new QWidget();
    auto* yoloBox = new QVBoxLayout(m_yoloTaskBox);
    yoloBox->setContentsMargins(0, 0, 0, 0);
    yoloBox->setSpacing(10);
    m_cbTask = new QComboBox();
    keepFormControlHeight(m_cbTask);
    m_cbTask->addItems({ "detect", "segment", "pose", "obb" });
    connect(m_cbTask, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] { onTaskChanged(); });
    yoloBox->addWidget(makeField("任务头 task"));
    yoloBox->addWidget(m_cbTask);

    m_cbInitMode = new QComboBox();
    keepFormControlHeight(m_cbInitMode);
    m_cbInitMode->addItem("加载预训练权重微调", "finetune");
    m_cbInitMode->addItem("从模型结构从零训练", "scratch");
    m_cbInitMode->addItem("从 last.pt 断点恢复", "resume");
    connect(m_cbInitMode, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] { updateInitHint(); });
    yoloBox->addWidget(makeField("训练初始化方式"));
    yoloBox->addWidget(m_cbInitMode);
    m_lblInitHint = new QLabel();
    m_lblInitHint->setObjectName("fieldLabel");
    keepInfoLabelReadable(m_lblInitHint);
    yoloBox->addWidget(m_lblInitHint);

    m_edModel = new QLineEdit("yolov8n.pt");
    keepFormControlHeight(m_edModel);
    auto* modelBtn = new QPushButton("浏览");
    modelBtn->setProperty("ghost", true);
    keepFormControlHeight(modelBtn);
    connect(modelBtn, &QPushButton::clicked, this, &TrainingTab::onBrowseModel);
    auto* modelRow = new QHBoxLayout();
    modelRow->addWidget(m_edModel, 1);
    modelRow->addWidget(modelBtn);
    yoloBox->addWidget(makeField("模型入口 model / weights"));
    yoloBox->addLayout(modelRow);

    m_edPretrained = new QLineEdit("true");
    keepFormControlHeight(m_edPretrained);
    yoloBox->addWidget(makeField("预训练 pretrained（true / false / 权重路径）"));
    yoloBox->addWidget(m_edPretrained);
    v->addWidget(m_yoloTaskBox);

    m_anomalibTaskBox = new QWidget();
    auto* anomBox = new QVBoxLayout(m_anomalibTaskBox);
    anomBox->setContentsMargins(0, 0, 0, 0);
    anomBox->setSpacing(10);
    m_cbAnomalibModel = new QComboBox();
    keepFormControlHeight(m_cbAnomalibModel);
    m_cbAnomalibModel->addItem("PatchCore", "patchcore");
    m_cbAnomalibModel->addItem("PaDiM", "padim");
    m_cbAnomalibModel->addItem("EfficientAd", "efficientad");
    connect(m_cbAnomalibModel, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] { onAnomalibModelChanged(); });
    anomBox->addWidget(makeField("Anomalib 模型"));
    anomBox->addWidget(m_cbAnomalibModel);
    m_edAnomNormalDir = new QLineEdit("train/good");
    m_edAnomAbnormalDir = new QLineEdit("test/abnormal");
    m_edAnomNormalTestDir = new QLineEdit("test/good");
    anomBox->addWidget(makeDirectoryRow(m_edAnomNormalDir, "正常训练目录 normal_dir", SLOT(onBrowseAnomalibNormalDir())));
    anomBox->addWidget(makeDirectoryRow(m_edAnomAbnormalDir, "异常测试目录 abnormal_dir", SLOT(onBrowseAnomalibAbnormalDir())));
    anomBox->addWidget(makeDirectoryRow(m_edAnomNormalTestDir, "正常测试目录 normal_test_dir", SLOT(onBrowseAnomalibNormalTestDir())));
    v->addWidget(m_anomalibTaskBox);

    m_edData = new QLineEdit();
    keepFormControlHeight(m_edData);
    auto* dataBtn = new QPushButton("浏览");
    dataBtn->setProperty("ghost", true);
    keepFormControlHeight(dataBtn);
    connect(dataBtn, &QPushButton::clicked, this, &TrainingTab::onBrowseData);
    auto* dataRow = new QHBoxLayout();
    dataRow->addWidget(m_edData, 1);
    dataRow->addWidget(dataBtn);
    v->addWidget(makeField("YOLO: dataset.yaml / Anomalib: Folder root"));
    v->addLayout(dataRow);

    m_edProject = new QLineEdit("runs/trtforge");
    keepFormControlHeight(m_edProject);
    auto* projectBtn = new QPushButton("目录");
    projectBtn->setProperty("ghost", true);
    keepFormControlHeight(projectBtn);
    connect(projectBtn, &QPushButton::clicked, this, &TrainingTab::onBrowseProject);
    auto* projectRow = new QHBoxLayout();
    projectRow->addWidget(m_edProject, 1);
    projectRow->addWidget(projectBtn);
    v->addWidget(makeField("输出目录 project"));
    v->addLayout(projectRow);

    m_edName = new QLineEdit("exp_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
    keepFormControlHeight(m_edName);
    v->addWidget(makeField("实验名称 name"));
    v->addWidget(m_edName);
    v->addStretch(1);
    return card;
}

QWidget* TrainingTab::buildParamCard()
{
    auto* card = makeCard();
    card->setMinimumWidth(520);
    card->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(12);
    v->addWidget(makeTitle("02 / 训练、验证与导出参数"));

    m_yoloParamBox = new QWidget();
    auto* yoloParamLayout = new QVBoxLayout(m_yoloParamBox);
    yoloParamLayout->setContentsMargins(0, 0, 0, 0);
    yoloParamLayout->setSpacing(12);

    auto* grid = new QGridLayout();
    grid->setHorizontalSpacing(12);
    grid->setVerticalSpacing(10);
    auto addSpin = [&](int row, int col, const QString& label, QSpinBox*& spin, int minV, int maxV, int defV) {
        auto* box = new QVBoxLayout();
        box->setSpacing(4);
        box->addWidget(makeField(label));
        spin = new QSpinBox();
        spin->setRange(minV, maxV);
        spin->setValue(defV);
        spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
        box->addWidget(spin);
        grid->addLayout(box, row, col);
    };
    addSpin(0, 0, "epochs", m_spEpochs, 1, 100000, 100);
    addSpin(0, 1, "imgsz", m_spImgsz, 32, 8192, 640);
    addSpin(0, 2, "patience", m_spPatience, 0, 100000, 100);
    addSpin(0, 3, "workers", m_spWorkers, 0, 128, 8);

    auto addLine = [&](int row, int col, const QString& label, QLineEdit*& edit, const QString& def) {
        auto* box = new QVBoxLayout();
        box->setSpacing(4);
        box->addWidget(makeField(label));
        edit = new QLineEdit(def);
        box->addWidget(edit);
        grid->addLayout(box, row, col);
    };
    addLine(1, 0, "batch", m_edBatch, "16");
    addLine(1, 1, "device", m_edDevice, "0");
    yoloParamLayout->addLayout(grid);

    auto* flags = new QHBoxLayout();
    m_chkHalf = new QCheckBox("验证/导出 FP16 half");
    m_chkDynamic = new QCheckBox("导出 dynamic");
    m_chkMakeTransd = new QCheckBox("生成 .transd.onnx");
    m_chkMakeTransd->setChecked(true);
    flags->addWidget(m_chkHalf);
    flags->addWidget(m_chkDynamic);
    flags->addWidget(m_chkMakeTransd);
    flags->addStretch(1);
    yoloParamLayout->addLayout(flags);

    auto* hint = new QLabel("高级参数将由 Python launcher 支持：optimizer、lr0、lrf、momentum、weight_decay、freeze、segment/pose/obb 专属损失参数。");
    hint->setObjectName("fieldLabel");
    hint->setWordWrap(true);
    yoloParamLayout->addWidget(hint);
    v->addWidget(m_yoloParamBox);

    m_anomalibParamBox = new QWidget();
    auto* anomLayout = new QGridLayout(m_anomalibParamBox);
    anomLayout->setContentsMargins(0, 0, 0, 0);
    anomLayout->setHorizontalSpacing(12);
    anomLayout->setVerticalSpacing(10);
    auto addAnomLine = [&](int row, int col, const QString& label, QLineEdit*& edit, const QString& def) {
        auto* box = new QVBoxLayout();
        box->setSpacing(4);
        box->addWidget(makeField(label));
        edit = new QLineEdit(def);
        box->addWidget(edit);
        anomLayout->addLayout(box, row, col);
    };
    auto addAnomSpin = [&](int row, int col, const QString& label, QSpinBox*& spin, int minV, int maxV, int defV) {
        auto* box = new QVBoxLayout();
        box->setSpacing(4);
        box->addWidget(makeField(label));
        spin = new QSpinBox();
        spin->setRange(minV, maxV);
        spin->setValue(defV);
        spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
        box->addWidget(spin);
        anomLayout->addLayout(box, row, col);
    };
    addAnomSpin(0, 0, "input_size", m_spAnomInputSize, 32, 4096, 256);
    addAnomLine(0, 1, "max_epochs（auto 或数字）", m_edAnomMaxEpochs, "auto");
    addAnomLine(0, 2, "train_batch_size", m_edAnomTrainBatch, "4");
    addAnomLine(0, 3, "eval_batch_size", m_edAnomEvalBatch, "4");
    addAnomSpin(1, 0, "num_workers", m_spAnomWorkers, 0, 64, 0);
    addAnomLine(1, 1, "calib_normal_dir", m_edAnomCalibNormalDir, "");
    addAnomLine(1, 2, "calib_abnormal_dir", m_edAnomCalibAbnormalDir, "");
    addAnomSpin(1, 3, "calib_limit", m_spAnomCalibLimit, 1, 10000, 64);
    m_chkAnomCalibrate = new QCheckBox("启用 metadata 校准");
    m_chkAnomCalibrate->setChecked(true);
    anomLayout->addWidget(m_chkAnomCalibrate, 2, 0);
    m_edAnomImagenetDir = new QLineEdit("anomalib-main/datasets/imagenette");
    auto* imagenetButton = new QPushButton("选择 EfficientAd 辅助数据");
    imagenetButton->setProperty("ghost", true);
    connect(imagenetButton, &QPushButton::clicked, this, &TrainingTab::onBrowseAnomalibImagenetDir);
    auto* imagenetRow = new QHBoxLayout();
    imagenetRow->addWidget(m_edAnomImagenetDir, 1);
    imagenetRow->addWidget(imagenetButton);
    anomLayout->addWidget(makeField("EfficientAd imagenet_dir"), 2, 1);
    anomLayout->addLayout(imagenetRow, 2, 2, 1, 2);
    v->addWidget(m_anomalibParamBox);

    auto* buttons = new QHBoxLayout();
    m_btnCheck = new QPushButton("检测环境");
    m_btnTrain = new QPushButton("开始训练");
    m_btnVal = new QPushButton("验证模型");
    m_btnExport = new QPushButton("导出 ONNX");
    m_btnStop = new QPushButton("停止");
    m_btnTrain->setProperty("primary", true);
    m_btnStop->setProperty("danger", true);
    connect(m_btnCheck, &QPushButton::clicked, this, &TrainingTab::onCheckEnv);
    connect(m_btnTrain, &QPushButton::clicked, this, &TrainingTab::onTrain);
    connect(m_btnVal, &QPushButton::clicked, this, &TrainingTab::onValidate);
    connect(m_btnExport, &QPushButton::clicked, this, &TrainingTab::onExport);
    connect(m_btnStop, &QPushButton::clicked, this, &TrainingTab::onStop);
    buttons->addWidget(m_btnCheck);
    buttons->addWidget(m_btnTrain);
    buttons->addWidget(m_btnVal);
    buttons->addWidget(m_btnExport);
    buttons->addWidget(m_btnStop);
    v->addLayout(buttons);
    return card;
}

QWidget* TrainingTab::buildEnvCard()
{
    auto* card = makeCard();
    card->setMinimumWidth(280);
    card->setMinimumHeight(120);
    card->setMaximumHeight(180);
    card->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* outer = new QVBoxLayout(card);
    outer->setContentsMargins(16, 16, 16, 16);
    outer->setSpacing(8);
    outer->addWidget(makeTitle("03 / 环境与产物"));

    auto* content = new QWidget();
    auto* v = new QVBoxLayout(content);
    v->setContentsMargins(0, 0, 6, 0);
    v->setSpacing(6);
    v->setSizeConstraint(QLayout::SetMinAndMaxSize);
    auto* scroll = new QScrollArea();
    applyPanelScrollStyle(scroll, content);
    scroll->setWidget(content);
    outer->addWidget(scroll, 1);

    m_lblEnv = new QLabel("尚未检测 Python 环境");
    m_lblEnv->setObjectName("fieldLabel");
    keepInfoLabelReadable(m_lblEnv);
    v->addWidget(m_lblEnv);
    m_lblMetric = new QLabel("指标：等待训练或验证");
    m_lblMetric->setObjectName("fieldLabel");
    keepInfoLabelReadable(m_lblMetric);
    v->addWidget(m_lblMetric);
    m_lblArtifact = new QLabel("产物：best.pt / ONNX / transd.onnx 会显示在这里");
    m_lblArtifact->setObjectName("fieldLabel");
    keepInfoLabelReadable(m_lblArtifact);
    v->addWidget(m_lblArtifact);
    v->addStretch(1);
    return card;
}

QWidget* TrainingTab::buildLogCard()
{
    auto* card = makeCard();
    card->setMinimumHeight(180);
    card->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(8);
    v->addWidget(makeTitle("QTextBrowser 运行日志 / stdout / stderr"));
    m_log = new QTextBrowser();
    m_log->setObjectName("logView");
    m_log->setMinimumHeight(130);
    m_log->setOpenExternalLinks(false);
    m_log->setPlaceholderText("Python 训练、验证/测试、ONNX 导出、metadata 写出和报错信息会实时显示在这里。");
    v->addWidget(m_log, 1);
    return card;
}

void TrainingTab::onBrowsePython()
{
    const QString path = QFileDialog::getOpenFileName(this, "选择 python.exe", m_edPython->text(),
                                                      "Python (python.exe);;所有文件 (*.*)");
    if (!path.isEmpty()) {
        m_edPython->setText(QDir::toNativeSeparators(path));
        saveSettings();
    }
}

void TrainingTab::onBrowseModel()
{
    const QString path = QFileDialog::getOpenFileName(this, "选择模型或权重", m_edModel->text(),
                                                      "YOLO 模型 (*.pt *.yaml *.yml);;所有文件 (*.*)");
    if (!path.isEmpty()) {
        m_edModel->setText(QDir::toNativeSeparators(path));
        saveSettings();
    }
}

void TrainingTab::onBrowseData()
{
    if (isAnomalibMode()) {
        const QString path = QFileDialog::getExistingDirectory(this, "选择 Anomalib Folder root", m_edData->text());
        if (!path.isEmpty()) {
            m_edData->setText(QDir::toNativeSeparators(path));
            saveSettings();
        }
        return;
    }
    const QString path = QFileDialog::getOpenFileName(this, "选择 dataset.yaml", m_edData->text(),
                                                      "Dataset YAML (*.yaml *.yml);;所有文件 (*.*)");
    if (!path.isEmpty()) {
        m_edData->setText(QDir::toNativeSeparators(path));
        saveSettings();
    }
}

void TrainingTab::onBrowseAnomalibNormalDir()
{
    const QString path = QFileDialog::getExistingDirectory(this, "选择正常训练目录", m_edData->text());
    if (!path.isEmpty()) {
        m_edAnomNormalDir->setText(QDir::toNativeSeparators(path));
        saveSettings();
    }
}

void TrainingTab::onBrowseAnomalibAbnormalDir()
{
    const QString path = QFileDialog::getExistingDirectory(this, "选择异常测试目录", m_edData->text());
    if (!path.isEmpty()) {
        m_edAnomAbnormalDir->setText(QDir::toNativeSeparators(path));
        saveSettings();
    }
}

void TrainingTab::onBrowseAnomalibNormalTestDir()
{
    const QString path = QFileDialog::getExistingDirectory(this, "选择正常测试目录", m_edData->text());
    if (!path.isEmpty()) {
        m_edAnomNormalTestDir->setText(QDir::toNativeSeparators(path));
        saveSettings();
    }
}

void TrainingTab::onBrowseAnomalibImagenetDir()
{
    const QString path = QFileDialog::getExistingDirectory(this, "选择 EfficientAd imagenet_dir", m_edAnomImagenetDir->text());
    if (!path.isEmpty()) {
        m_edAnomImagenetDir->setText(QDir::toNativeSeparators(path));
        saveSettings();
    }
}

void TrainingTab::onBrowseProject()
{
    const QString path = QFileDialog::getExistingDirectory(this, "选择训练输出目录", m_edProject->text());
    if (!path.isEmpty()) {
        m_edProject->setText(QDir::toNativeSeparators(path));
        saveSettings();
    }
}

void TrainingTab::onCheckEnv()
{
    saveSettings();
    m_controller->setPythonPath(m_edPython->text());
    m_controller->setLauncherKind(isAnomalibMode()
        ? YoloTrainController::LauncherKind::Anomalib
        : YoloTrainController::LauncherKind::Yolo);
    m_controller->start({ "--mode", "check-env" });
}

void TrainingTab::onTrain()
{
    if (isAnomalibMode()) {
        if (m_edData->text().trimmed().isEmpty()) {
            QMessageBox::warning(this, "提示", "请先选择 Anomalib Folder 数据集 root。");
            return;
        }
        saveSettings();
        resetRunArtifacts("anomalib-train-export");
        QStringList args = baseAnomalibArgs();
        m_controller->setPythonPath(m_edPython->text());
        m_controller->setLauncherKind(YoloTrainController::LauncherKind::Anomalib);
        m_controller->start(args);
        return;
    }
    if (m_edData->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, "提示", "请先选择 dataset.yaml。");
        return;
    }
    prepareTrainInitialization();
    saveSettings();
    resetRunArtifacts("train");
    QStringList args = baseYoloArgs("train");
    args << "--epochs" << QString::number(m_spEpochs->value())
         << "--patience" << QString::number(m_spPatience->value())
         << "--workers" << QString::number(m_spWorkers->value())
         << "--project" << m_edProject->text().trimmed()
         << "--name" << m_edName->text().trimmed();

    const QString mode = m_cbInitMode->currentData().toString();
    if (mode == "scratch") {
        args << "--pretrained" << "false";
    } else if (mode == "resume") {
        args << "--resume";
    } else if (!m_edPretrained->text().trimmed().isEmpty()) {
        args << "--pretrained" << m_edPretrained->text().trimmed();
    }
    m_controller->setPythonPath(m_edPython->text());
    m_controller->setLauncherKind(YoloTrainController::LauncherKind::Yolo);
    m_controller->start(args);
}

void TrainingTab::onValidate()
{
    saveSettings();
    resetRunArtifacts("val");
    QStringList args = baseYoloArgs("val");
    args << "--split" << "val";
    if (m_chkHalf->isChecked()) args << "--half";
    m_controller->setPythonPath(m_edPython->text());
    m_controller->setLauncherKind(YoloTrainController::LauncherKind::Yolo);
    m_controller->start(args);
}

void TrainingTab::onExport()
{
    saveSettings();
    resetRunArtifacts("export");
    QStringList args = baseYoloArgs("export");
    args << "--format" << "onnx"
         << "--export-batch" << "1"
         << "--make-transd" << (m_chkMakeTransd->isChecked() ? "true" : "false");
    if (m_chkHalf->isChecked()) args << "--half";
    if (m_chkDynamic->isChecked()) args << "--dynamic";
    m_controller->setPythonPath(m_edPython->text());
    m_controller->setLauncherKind(YoloTrainController::LauncherKind::Yolo);
    m_controller->start(args);
}

void TrainingTab::onStop()
{
    saveSettings();
    m_controller->stop();
}

void TrainingTab::onTaskChanged()
{
    if (m_loadingSettings) return;
    QSettings settings;
    settings.beginGroup("YoloTraining");
    if (!m_currentTask.isEmpty()) {
        settings.setValue(QString("pythonPathByTask/%1").arg(m_currentTask), m_edPython->text().trimmed());
    }
    const QString nextTask = m_cbTask->currentText();
    const QString taskPython = settings.value(QString("pythonPathByTask/%1").arg(nextTask)).toString();
    settings.endGroup();
    m_currentTask = nextTask;
    if (!taskPython.isEmpty()) {
        m_edPython->setText(taskPython);
    }
    if (m_edModel) {
        const QString task = nextTask;
        if (task == "detect") m_edModel->setText("yolov8n.pt");
        else if (task == "segment") m_edModel->setText("yolov8n-seg.pt");
        else if (task == "pose") m_edModel->setText("yolov8n-pose.pt");
        else if (task == "obb") m_edModel->setText("yolov8n-obb.pt");
    }
    saveSettings();
}

void TrainingTab::onTrainTypeChanged()
{
    if (m_loadingSettings) return;
    updateModeUi();
    saveSettings();
}

void TrainingTab::onAnomalibModelChanged()
{
    if (!m_cbAnomalibModel) return;
    const QString model = m_cbAnomalibModel->currentData().toString();
    if (model == "efficientad") {
        if (m_edAnomMaxEpochs && m_edAnomMaxEpochs->text().trimmed().isEmpty()) {
            m_edAnomMaxEpochs->setText("auto");
        }
        if (m_edAnomTrainBatch) m_edAnomTrainBatch->setText("1");
    } else if (m_edAnomTrainBatch && m_edAnomTrainBatch->text().trimmed() == "1") {
        m_edAnomTrainBatch->setText("4");
    }
    updateModeUi();
    saveSettings();
}

void TrainingTab::onLogLine(QString text, bool isError)
{
    const AppLogEvent event {
        isError ? AppLogLevel::Error : AppLogLevel::Info,
        AppEventSource::Training,
        text
    };
    appendLog(event.message, AppEventFormatter::logColor(event.level), event.level);
}

void TrainingTab::onStructuredEvent(QJsonObject event)
{
    const QString type = event.value("type").toString();
    if (type == "env") {
        QStringList lines;
        lines << "Python: " + event.value("python").toString();
        lines << "Env: " + event.value("env_name").toString();
        lines << "Prefix: " + event.value("sys_prefix").toString();
        lines << "Version: " + event.value("python_version").toString();
        for (const QString& key : {
                 "torch", "ultralytics", "anomalib", "lightning", "onnx", "onnxruntime", "PIL", "numpy", "polars"
             }) {
            if (!event.contains(key)) continue;
            const QJsonObject obj = event.value(key).toObject();
            const bool ok = obj.value("ok").toBool();
            QString detail = ok ? obj.value("version").toString() : obj.value("error").toString();
            lines << QString("%1: %2 %3").arg(key, ok ? "OK" : "缺失", detail);
        }
        const QJsonObject cuda = event.value("cuda").toObject();
        lines << QString("CUDA: %1 %2").arg(cuda.value("available").toBool() ? "可用" : "不可用",
                                           jsonValueToText(cuda.value("devices")));
        const QJsonArray missing = event.value("missing").toArray();
        if (!missing.isEmpty()) {
            lines << "缺失库安装命令：";
            for (const QJsonValue& item : missing) {
                const QJsonObject obj = item.toObject();
                lines << "  " + obj.value("install").toString();
            }
        }
        m_lblEnv->setText(lines.join('\n'));
    } else if (type == "metric") {
        m_lblMetric->setText(QString("%1 = %2").arg(event.value("name").toString(),
                                                   jsonValueToText(event.value("value"))));
    } else if (type == "epoch_summary") {
        appendEpochSummary(event);
    } else if (type == "gpu_status") {
        updateGpuStatus(event);
    } else if (type == "plot") {
        updatePlotPreview(event);
    } else if (type == "artifact") {
        const QString kind = event.value("kind").toString();
        const QString name = event.value("name").toString();
        const QString path = event.value("path").toString();
        const QString line = m_artifactModel.addArtifact(kind, name, path);
        m_lblArtifact->setText(line);
        if (kind == "labels_json" && m_lblLabelsJson) {
            const QString message = event.value("message").toString();
            m_lblLabelsJson->setText(m_artifactModel.labelsJsonStatusText(name, path, message));
        } else if (kind == "plot") {
            updatePlotPreview(event);
        }
    } else if (type == "anomaly_model") {
        m_lblMetric->setText(QString("Anomalib 模型：%1\n输入尺寸：%2")
            .arg(event.value("model").toString(), jsonValueToText(event.value("input_size"))));
    } else if (type == "anomaly_export_summary") {
        QStringList lines;
        lines << "Anomalib 训练导出完成";
        lines << "model: " + event.value("model_type").toString();
        lines << "contract: " + event.value("output_contract").toString();
        lines << "normalization: " + event.value("normalization").toString();
        lines << "image_threshold: " + jsonValueToText(event.value("image_threshold"));
        lines << "pixel_threshold: " + jsonValueToText(event.value("pixel_threshold"));
        m_lblMetric->setText(lines.join('\n'));
    } else if (type == "error") {
        const QString message = event.value("message").toString();
        appendLog("ERROR: " + message, "#FF4D6D", AppLogLevel::Error);
        if (event.value("code").toString() == "LABEL_JSON_WRITE_FAILED" && m_lblLabelsJson) {
            m_lblLabelsJson->setText("标签 JSON：写入失败\n" + message + "\n" + event.value("error").toString());
        }
        QStringList details;
        details << message;
        if (!event.value("python").toString().isEmpty()) {
            details << "Python: " + event.value("python").toString();
        }
        if (!event.value("env_name").toString().isEmpty()) {
            details << "虚拟环境: " + event.value("env_name").toString();
        }
        const QJsonArray missing = event.value("missing").toArray();
        if (!missing.isEmpty()) {
            details << "缺失库和安装命令：";
            for (const QJsonValue& item : missing) {
                const QJsonObject obj = item.toObject();
                details << QString(" - %1").arg(obj.value("package").toString());
                details << QString("   %1").arg(obj.value("install").toString());
            }
        } else if (!event.value("install").toString().isEmpty()) {
            details << "安装命令: " + event.value("install").toString();
        }
        QMessageBox::warning(this, isAnomalibMode() ? "Anomalib 任务错误" : "YOLO 任务错误", details.join('\n'));
    } else if (type == "progress") {
        const AppProgressEvent progress {
            AppEventSource::Training,
            event.value("stage").toString(),
            qRound(event.value("pct").toDouble()),
            event
        };
        const AppLogEvent logEvent {
            AppLogLevel::Stage,
            AppEventSource::Training,
            QString("进度：epoch %1/%2，%3%")
                .arg(jsonValueToText(progress.detail.value("epoch")),
                     jsonValueToText(progress.detail.value("total")),
                     QString::number(progress.detail.value("pct").toDouble(), 'f', 1))
        };
        appendLog(logEvent.message, AppEventFormatter::logColor(logEvent.level), logEvent.level);
    }
}

void TrainingTab::onProcessStarted(QString program, QStringList arguments)
{
    setRunning(true);
    if (!program.trimmed().isEmpty()) {
        m_edPython->setText(program);
        savePythonPath(program);
    }
    const AppLogEvent event { AppLogLevel::Stage, AppEventSource::Training,
        QString("启动：%1 %2").arg(program, arguments.join(' ')) };
    appendLog(event.message, AppEventFormatter::logColor(event.level), event.level);
}

void TrainingTab::onProcessFinished(int exitCode, QString reason)
{
    setRunning(false);
    const AppLogEvent event {
        exitCode == 0 ? AppLogLevel::Success : AppLogLevel::Error,
        AppEventSource::Training,
        QString("进程结束：exit=%1 reason=%2").arg(exitCode).arg(reason)
    };
    appendLog(event.message, AppEventFormatter::logColor(event.level), event.level);
    if (exitCode == 0 && (m_activeMode == "train" || m_activeMode == "anomalib-train-export")) {
        showTrainingArtifactsDialog();
    }
    m_activeMode.clear();
}

void TrainingTab::onProcessError(QString message)
{
    setRunning(false);
    appendLog("ERROR: " + message, "#FF4D6D", AppLogLevel::Error);
    QMessageBox::warning(this, isAnomalibMode() ? "Anomalib 任务错误" : "YOLO 任务错误", message);
}

bool TrainingTab::isAnomalibMode() const
{
    return m_cbTrainType && m_cbTrainType->currentData().toString() == "anomalib";
}

QStringList TrainingTab::baseYoloArgs(const QString& mode) const
{
    QStringList args;
    args << "--mode" << mode
         << "--task" << m_cbTask->currentText()
         << "--model" << m_edModel->text().trimmed()
         << "--data" << m_edData->text().trimmed()
         << "--imgsz" << QString::number(m_spImgsz->value())
         << "--batch" << m_edBatch->text().trimmed();
    if (!m_edDevice->text().trimmed().isEmpty()) {
        args << "--device" << m_edDevice->text().trimmed();
    }
    return args;
}

QStringList TrainingTab::baseAnomalibArgs() const
{
    QStringList args;
    args << "--mode" << "train-export"
         << "--model" << m_cbAnomalibModel->currentData().toString()
         << "--root" << m_edData->text().trimmed()
         << "--normal-dir" << m_edAnomNormalDir->text().trimmed()
         << "--abnormal-dir" << m_edAnomAbnormalDir->text().trimmed()
         << "--normal-test-dir" << m_edAnomNormalTestDir->text().trimmed()
         << "--project" << m_edProject->text().trimmed()
         << "--name" << m_edName->text().trimmed()
         << "--input-size" << QString::number(m_spAnomInputSize->value())
         << "--max-epochs" << m_edAnomMaxEpochs->text().trimmed()
         << "--train-batch-size" << m_edAnomTrainBatch->text().trimmed()
         << "--eval-batch-size" << m_edAnomEvalBatch->text().trimmed()
         << "--num-workers" << QString::number(m_spAnomWorkers->value())
         << "--val-split-ratio" << "0.2"
         << "--calibrate-metadata" << (m_chkAnomCalibrate->isChecked() ? "true" : "false")
         << "--calib-limit" << QString::number(m_spAnomCalibLimit->value());
    if (!m_edAnomImagenetDir->text().trimmed().isEmpty()) {
        args << "--imagenet-dir" << m_edAnomImagenetDir->text().trimmed();
    }
    if (!m_edAnomCalibNormalDir->text().trimmed().isEmpty()) {
        args << "--calib-normal-dir" << m_edAnomCalibNormalDir->text().trimmed();
    }
    if (!m_edAnomCalibAbnormalDir->text().trimmed().isEmpty()) {
        args << "--calib-abnormal-dir" << m_edAnomCalibAbnormalDir->text().trimmed();
    }
    return args;
}

void TrainingTab::clearMonitorData()
{
    if (m_metricTable) {
        m_metricTable->setRowCount(0);
    }
    if (m_lblGpu) {
        m_lblGpu->setText("GPU：等待训练采集");
    }
    if (m_barGpuMemory) {
        m_barGpuMemory->setValue(0);
        m_barGpuMemory->setFormat("等待采集");
    }
    if (m_barGpuUtil) {
        m_barGpuUtil->setValue(0);
        m_barGpuUtil->setFormat("等待采集");
    }
    if (m_lblLabelsJson) {
        m_lblLabelsJson->setText("标签 JSON：等待训练结束生成 labels_<实验名称>.json");
    }
    restoreChartGrid();
    for (auto it = m_chartWidgets.begin(); it != m_chartWidgets.end(); ++it) {
        if (!it.value()) continue;
        it.value()->clear();
    }
}

void TrainingTab::appendEpochSummary(const QJsonObject& event)
{
    if (!m_metricTable) return;
    const QJsonObject losses = event.value("losses").toObject();
    const QJsonObject metrics = event.value("metrics").toObject();
    const QJsonObject lr = event.value("lr").toObject();
    const QJsonObject gpu = event.value("gpu").toObject();

    auto pick = [](const QJsonObject& obj, const QStringList& keys) -> QString {
        for (const QString& key : keys) {
            if (!obj.contains(key)) continue;
            const QJsonValue value = obj.value(key);
            if (value.isDouble()) return QString::number(value.toDouble(), 'f', 4);
            if (value.isString()) return value.toString();
        }
        return "-";
    };

    const int row = m_metricTable->rowCount();
    m_metricTable->insertRow(row);
    const QString gpuText = gpu.value("available").toBool()
        ? QString("%1/%2MB %3%")
              .arg(QString::number(gpu.value("memory_used_mb").toDouble(), 'f', 0),
                   QString::number(gpu.value("memory_total_mb").toDouble(), 'f', 0),
                   gpu.contains("utilization_gpu_pct") ? QString::number(gpu.value("utilization_gpu_pct").toDouble(), 'f', 0) : "-")
        : "-";
    const QStringList values = {
        QString("%1/%2").arg(jsonValueToText(event.value("epoch")), jsonValueToText(event.value("total"))),
        pick(losses, {"box_loss", "train/box_loss", "loss/0"}),
        pick(losses, {"cls_loss", "train/cls_loss", "loss/1"}),
        pick(losses, {"dfl_loss", "train/dfl_loss", "loss/2"}),
        pick(metrics, {"metrics/precision(B)", "metrics/precision"}),
        pick(metrics, {"metrics/recall(B)", "metrics/recall"}),
        pick(metrics, {"metrics/mAP50(B)", "metrics/mAP50"}),
        pick(metrics, {"metrics/mAP50-95(B)", "metrics/mAP50-95"}),
        pick(lr, {"lr/pg0", "lr/0"}),
        gpuText,
    };
    for (int col = 0; col < values.size(); ++col) {
        auto* item = new QTableWidgetItem(values[col]);
        item->setForeground(QColor("#D6F3FF"));
        item->setBackground((row % 2 == 0) ? QColor("#020617") : QColor("#0B2238"));
        m_metricTable->setItem(row, col, item);
    }
    updateMonitorCharts(event);
    updateGpuStatus(gpu);
}

void TrainingTab::updateGpuStatus(const QJsonObject& gpu)
{
    if (!m_lblGpu) return;
    if (!gpu.value("available").toBool()) {
        m_lblGpu->setText("GPU：未采集到实时占用\n可继续训练，但无法辅助 batch 调整。");
        if (m_barGpuMemory) {
            m_barGpuMemory->setValue(0);
            m_barGpuMemory->setFormat("未采集到显存");
        }
        if (m_barGpuUtil) {
            m_barGpuUtil->setValue(0);
            m_barGpuUtil->setFormat("未采集到利用率");
        }
        return;
    }
    QStringList lines;
    lines << "GPU：" + gpu.value("name").toString();
    const double usedMb = gpu.value("memory_used_mb").toDouble();
    const double totalMb = gpu.value("memory_total_mb").toDouble();
    const int memoryPct = totalMb > 0.0 ? qBound(0, qRound(usedMb * 100.0 / totalMb), 100) : 0;
    lines << QString("显存：%1 / %2 MB（%3%）")
                 .arg(QString::number(usedMb, 'f', 0),
                      QString::number(totalMb, 'f', 0),
                      QString::number(memoryPct));
    if (gpu.contains("utilization_gpu_pct")) {
        lines << QString("利用率：%1%").arg(QString::number(gpu.value("utilization_gpu_pct").toDouble(), 'f', 0));
    }
    if (m_barGpuMemory) {
        m_barGpuMemory->setValue(memoryPct);
        m_barGpuMemory->setFormat(QString("%1%  %2 / %3 MB")
                                      .arg(memoryPct)
                                      .arg(QString::number(usedMb, 'f', 0),
                                           QString::number(totalMb, 'f', 0)));
    }
    if (m_barGpuUtil) {
        if (gpu.contains("utilization_gpu_pct")) {
            const int utilPct = qBound(0, qRound(gpu.value("utilization_gpu_pct").toDouble()), 100);
            m_barGpuUtil->setValue(utilPct);
            m_barGpuUtil->setFormat(QString("%1%").arg(utilPct));
        } else {
            m_barGpuUtil->setValue(0);
            m_barGpuUtil->setFormat("nvidia-smi 未返回");
        }
    }
    if (totalMb > 0) {
        const double ratio = usedMb / totalMb;
        lines << (ratio < 0.70 ? "建议：显存余量较多，可尝试提高 batch。" : "建议：显存占用较高，谨慎提高 batch。");
    }
    m_lblGpu->setText(lines.join('\n'));
}

void TrainingTab::updatePlotPreview(const QJsonObject& event)
{
    const QString name = event.value("name").toString();
    const QString path = event.value("path").toString();
    MonitorChartWidget* chart = m_chartWidgets.value(name, nullptr);
    if (chart && !path.isEmpty()) {
        chart->setArtifactPath(path);
    }
}

void TrainingTab::addPlotCard(QGridLayout* grid, int row, int col, const QString& key, const QString& title)
{
    auto* card = makeCard();
    card->setStyleSheet("QFrame#card { background:#06101E; border:1px solid #16324A; border-radius:10px; }");
    auto* layout = new QVBoxLayout(card);
    layout->setContentsMargins(10, 10, 10, 10);
    layout->setSpacing(6);
    auto* titleLabel = makeField(title + " · 实时控件");
    layout->addWidget(titleLabel);
    static const QList<QColor> colors = {
        QColor("#22D3EE"), QColor("#A3E635"), QColor("#60A5FA"), QColor("#FBBF24"),
        QColor("#F472B6"), QColor("#34D399"), QColor("#F97316"), QColor("#C084FC")
    };
    const bool imageMode = key == "confusion_matrix" || key == "confusion_matrix_normalized";
    auto* chart = new MonitorChartWidget(title,
                                         colors[m_chartWidgets.size() % colors.size()],
                                         imageMode ? MonitorChartWidget::Mode::Image : MonitorChartWidget::Mode::Line,
                                         card);
    if (!imageMode && (key.startsWith("Box") || key == "results")) {
        chart->setFixedYRange(0.0, 1.0);
    } else if (key == "labels") {
        chart->setFixedYRange(0.0, 100.0, "%");
    }
    chart->setDoubleClickHandler([this, key]() { toggleChartZoom(key); });
    layout->addWidget(chart, 1);
    m_chartWidgets.insert(key, chart);
    m_chartCards.insert(key, card);
    m_chartPositions.insert(key, QPoint(row, col));
    grid->addWidget(card, row, col);
}

void TrainingTab::updateMonitorCharts(const QJsonObject& event)
{
    const int epoch = event.value("epoch").toInt();
    if (epoch <= 0) return;

    const QJsonObject losses = event.value("losses").toObject();
    const QJsonObject metrics = event.value("metrics").toObject();
    const QJsonObject gpu = event.value("gpu").toObject();

    auto pickDouble = [](const QJsonObject& obj, const QStringList& keys, double* out) -> bool {
        for (const QString& key : keys) {
            if (!obj.contains(key)) continue;
            const QJsonValue value = obj.value(key);
            if (value.isDouble()) {
                *out = value.toDouble();
                return true;
            }
            if (value.isString()) {
                bool ok = false;
                const double parsed = value.toString().toDouble(&ok);
                if (ok) {
                    *out = parsed;
                    return true;
                }
            }
        }
        return false;
    };

    auto add = [&](const QString& key, double value) {
        if (auto* chart = m_chartWidgets.value(key, nullptr)) {
            chart->addPoint(epoch, value);
        }
    };

    double precision = 0.0;
    double recall = 0.0;
    double value = 0.0;
    const bool hasPrecision = pickDouble(metrics, {"metrics/precision(B)", "metrics/precision"}, &precision);
    const bool hasRecall = pickDouble(metrics, {"metrics/recall(B)", "metrics/recall"}, &recall);
    if (hasPrecision) add("BoxP_curve", precision);
    if (hasRecall) add("BoxR_curve", recall);
    if (hasPrecision && hasRecall && (precision + recall) > 0.0) {
        add("BoxF1_curve", 2.0 * precision * recall / (precision + recall));
    }
    if (pickDouble(metrics, {"metrics/mAP50(B)", "metrics/mAP50"}, &value)) add("BoxPR_curve", value);
    if (pickDouble(metrics, {"metrics/mAP50-95(B)", "metrics/mAP50-95"}, &value)) add("results", value);
    if (gpu.value("available").toBool() && gpu.value("memory_total_mb").toDouble() > 0.0) {
        add("labels", gpu.value("memory_used_mb").toDouble() / gpu.value("memory_total_mb").toDouble() * 100.0);
    }
}

void TrainingTab::toggleChartZoom(const QString& key)
{
    if (!m_chartGrid || !m_chartCards.contains(key)) return;
    if (m_zoomedChartKey == key) {
        restoreChartGrid();
        return;
    }
    QWidget* selected = m_chartCards.value(key);
    for (auto it = m_chartCards.begin(); it != m_chartCards.end(); ++it) {
        m_chartGrid->removeWidget(it.value());
        it.value()->setVisible(it.key() == key);
    }
    m_chartGrid->addWidget(selected, 0, 0, 4, 2);
    m_zoomedChartKey = key;
}

void TrainingTab::restoreChartGrid()
{
    if (!m_chartGrid) return;
    for (auto it = m_chartCards.begin(); it != m_chartCards.end(); ++it) {
        QWidget* card = it.value();
        if (!card) continue;
        m_chartGrid->removeWidget(card);
        const QPoint pos = m_chartPositions.value(it.key());
        m_chartGrid->addWidget(card, pos.x(), pos.y());
        card->setVisible(true);
    }
    m_zoomedChartKey.clear();
}

void TrainingTab::prepareTrainInitialization()
{
    if (!m_edModel || !m_edPretrained || !m_cbInitMode) return;
    if (m_cbInitMode->currentData().toString() != "finetune") return;

    const QString model = m_edModel->text().trimmed();
    const QString pretrained = m_edPretrained->text().trimmed();
    if (pretrained.isEmpty()) return;

    const QString lower = pretrained.toLower();
    if (lower == "true" || lower == "false") return;
    if (!lower.endsWith(".pt")) return;

    const QFileInfo pretrainedInfo(pretrained);
    if (!pretrainedInfo.exists()) {
        appendLog("WARNING: 预训练权重路径不存在，将按原参数交给 Ultralytics 处理：" + pretrained,
                  "#FFB020",
                  AppLogLevel::Warn);
        return;
    }

    if (model.isEmpty() || isDefaultModelAlias(model)) {
        const QString nativePath = QDir::toNativeSeparators(pretrainedInfo.absoluteFilePath());
        m_edModel->setText(nativePath);
        m_edPretrained->setText("true");
        appendLog("已将本地预训练权重作为模型入口，避免下载默认模型：" + nativePath,
                  "#22E27F",
                  AppLogLevel::Success);
    }
}

bool TrainingTab::isDefaultModelAlias(const QString& model) const
{
    const QString trimmed = model.trimmed();
    if (trimmed.isEmpty()) return true;
    const QString expected = defaultModelAlias(m_cbTask ? m_cbTask->currentText() : QString());
    if (trimmed.compare(expected, Qt::CaseInsensitive) == 0) return true;

    QFileInfo info(trimmed);
    return !info.exists() && trimmed.compare(info.fileName(), Qt::CaseInsensitive) == 0
        && trimmed.startsWith("yolov", Qt::CaseInsensitive) && trimmed.endsWith(".pt", Qt::CaseInsensitive);
}

void TrainingTab::resetRunArtifacts(const QString& mode)
{
    m_activeMode = mode;
    m_artifactModel.reset(mode);
    clearMonitorData();
    if (m_lblArtifact) {
        m_lblArtifact->setText("产物：等待当前任务输出");
    }
    if (m_log) {
        m_log->clear();
    }
}

void TrainingTab::showTrainingArtifactsDialog()
{
    QStringList lines;
    const bool anomalyRun = m_activeMode == "anomalib-train-export";
    lines << (anomalyRun ? "Anomalib 训练导出已完成。" : "训练已完成。");
    const QStringList artifacts = m_artifactModel.artifacts();
    if (artifacts.isEmpty()) {
        lines << (anomalyRun ? "未收到 Python 返回的 ckpt/onnx/metadata 路径，请查看 QTextBrowser 日志。"
                             : "未收到 Python 返回的产物路径，请查看日志中的 Ultralytics 保存目录。");
    } else {
        lines << "本次训练模型和附属文件：";
        for (const QString& artifact : artifacts) {
            lines << " - " + artifact;
        }
    }
    QMessageBox::information(this, anomalyRun ? "Anomalib 训练导出完成" : "YOLO 训练完成", lines.join('\n'));
}

void TrainingTab::loadSettings()
{
    m_loadingSettings = true;
    QSettings settings;
    settings.beginGroup("YoloTraining");
    const QString trainType = settings.value("trainingMode", "yolo").toString();
    const int trainTypeIndex = m_cbTrainType->findData(trainType);
    if (trainTypeIndex >= 0) m_cbTrainType->setCurrentIndex(trainTypeIndex);
    m_edPython->setText(settings.value("pythonPath").toString());
    const QString task = settings.value("task", "detect").toString();
    const int taskIndex = m_cbTask->findText(task);
    if (taskIndex >= 0) m_cbTask->setCurrentIndex(taskIndex);
    m_currentTask = m_cbTask->currentText();
    const QString initMode = settings.value("initMode", "finetune").toString();
    const int initIndex = m_cbInitMode->findData(initMode);
    if (initIndex >= 0) m_cbInitMode->setCurrentIndex(initIndex);
    m_edModel->setText(settings.value("model", m_edModel->text()).toString());
    m_edPretrained->setText(settings.value("pretrained", m_edPretrained->text()).toString());
    m_edData->setText(settings.value("data").toString());
    m_edProject->setText(settings.value("project", m_edProject->text()).toString());
    m_edName->setText(settings.value("name", m_edName->text()).toString());
    m_edDevice->setText(settings.value("device", m_edDevice->text()).toString());
    m_edBatch->setText(settings.value("batch", m_edBatch->text()).toString());
    m_spEpochs->setValue(settings.value("epochs", m_spEpochs->value()).toInt());
    m_spImgsz->setValue(settings.value("imgsz", m_spImgsz->value()).toInt());
    m_spPatience->setValue(settings.value("patience", m_spPatience->value()).toInt());
    m_spWorkers->setValue(settings.value("workers", m_spWorkers->value()).toInt());
    m_chkHalf->setChecked(settings.value("half", m_chkHalf->isChecked()).toBool());
    m_chkDynamic->setChecked(settings.value("dynamic", m_chkDynamic->isChecked()).toBool());
    m_chkMakeTransd->setChecked(settings.value("makeTransd", m_chkMakeTransd->isChecked()).toBool());
    const QString anomModel = settings.value("anomalib/model", "patchcore").toString();
    const int anomModelIndex = m_cbAnomalibModel->findData(anomModel);
    if (anomModelIndex >= 0) m_cbAnomalibModel->setCurrentIndex(anomModelIndex);
    m_edAnomNormalDir->setText(settings.value("anomalib/normalDir", m_edAnomNormalDir->text()).toString());
    m_edAnomAbnormalDir->setText(settings.value("anomalib/abnormalDir", m_edAnomAbnormalDir->text()).toString());
    m_edAnomNormalTestDir->setText(settings.value("anomalib/normalTestDir", m_edAnomNormalTestDir->text()).toString());
    m_edAnomImagenetDir->setText(settings.value("anomalib/imagenetDir", m_edAnomImagenetDir->text()).toString());
    m_edAnomMaxEpochs->setText(settings.value("anomalib/maxEpochs", m_edAnomMaxEpochs->text()).toString());
    m_edAnomTrainBatch->setText(settings.value("anomalib/trainBatchSize", m_edAnomTrainBatch->text()).toString());
    m_edAnomEvalBatch->setText(settings.value("anomalib/evalBatchSize", m_edAnomEvalBatch->text()).toString());
    m_edAnomCalibNormalDir->setText(settings.value("anomalib/calibNormalDir", m_edAnomCalibNormalDir->text()).toString());
    m_edAnomCalibAbnormalDir->setText(settings.value("anomalib/calibAbnormalDir", m_edAnomCalibAbnormalDir->text()).toString());
    m_spAnomInputSize->setValue(settings.value("anomalib/inputSize", m_spAnomInputSize->value()).toInt());
    m_spAnomWorkers->setValue(settings.value("anomalib/numWorkers", m_spAnomWorkers->value()).toInt());
    m_spAnomCalibLimit->setValue(settings.value("anomalib/calibLimit", m_spAnomCalibLimit->value()).toInt());
    m_chkAnomCalibrate->setChecked(settings.value("anomalib/calibrateMetadata", m_chkAnomCalibrate->isChecked()).toBool());
    settings.endGroup();
    m_loadingSettings = false;
}

void TrainingTab::saveSettings() const
{
    if (m_loadingSettings) return;
    QSettings settings;
    settings.beginGroup("YoloTraining");
    settings.setValue("trainingMode", m_cbTrainType->currentData().toString());
    settings.setValue("pythonPath", m_edPython->text().trimmed());
    settings.setValue(QString("pythonPathByTask/%1").arg(m_cbTask->currentText()), m_edPython->text().trimmed());
    settings.setValue("task", m_cbTask->currentText());
    settings.setValue("initMode", m_cbInitMode->currentData().toString());
    settings.setValue("model", m_edModel->text().trimmed());
    settings.setValue("pretrained", m_edPretrained->text().trimmed());
    settings.setValue("data", m_edData->text().trimmed());
    settings.setValue("project", m_edProject->text().trimmed());
    settings.setValue("name", m_edName->text().trimmed());
    settings.setValue("device", m_edDevice->text().trimmed());
    settings.setValue("batch", m_edBatch->text().trimmed());
    settings.setValue("epochs", m_spEpochs->value());
    settings.setValue("imgsz", m_spImgsz->value());
    settings.setValue("patience", m_spPatience->value());
    settings.setValue("workers", m_spWorkers->value());
    settings.setValue("half", m_chkHalf->isChecked());
    settings.setValue("dynamic", m_chkDynamic->isChecked());
    settings.setValue("makeTransd", m_chkMakeTransd->isChecked());
    settings.setValue("anomalib/model", m_cbAnomalibModel->currentData().toString());
    settings.setValue("anomalib/normalDir", m_edAnomNormalDir->text().trimmed());
    settings.setValue("anomalib/abnormalDir", m_edAnomAbnormalDir->text().trimmed());
    settings.setValue("anomalib/normalTestDir", m_edAnomNormalTestDir->text().trimmed());
    settings.setValue("anomalib/imagenetDir", m_edAnomImagenetDir->text().trimmed());
    settings.setValue("anomalib/maxEpochs", m_edAnomMaxEpochs->text().trimmed());
    settings.setValue("anomalib/trainBatchSize", m_edAnomTrainBatch->text().trimmed());
    settings.setValue("anomalib/evalBatchSize", m_edAnomEvalBatch->text().trimmed());
    settings.setValue("anomalib/calibNormalDir", m_edAnomCalibNormalDir->text().trimmed());
    settings.setValue("anomalib/calibAbnormalDir", m_edAnomCalibAbnormalDir->text().trimmed());
    settings.setValue("anomalib/inputSize", m_spAnomInputSize->value());
    settings.setValue("anomalib/numWorkers", m_spAnomWorkers->value());
    settings.setValue("anomalib/calibLimit", m_spAnomCalibLimit->value());
    settings.setValue("anomalib/calibrateMetadata", m_chkAnomCalibrate->isChecked());
    settings.endGroup();
}

void TrainingTab::savePythonPath(const QString& path) const
{
    if (m_loadingSettings) return;
    QSettings settings;
    settings.beginGroup("YoloTraining");
    settings.setValue("pythonPath", path.trimmed());
    if (!m_cbTask->currentText().isEmpty()) {
        settings.setValue(QString("pythonPathByTask/%1").arg(m_cbTask->currentText()), path.trimmed());
    }
    settings.endGroup();
}

void TrainingTab::appendLog(const QString& text, const QString& color, AppLogLevel level)
{
    AppFileLogger::append({ level, AppEventSource::Training, text });
    if (!m_log) return;
    const QString prefix = QDateTime::currentDateTime().toString("HH:mm:ss ");
    const QString safe = (prefix + text).toHtmlEscaped();
    const QString html = color.isEmpty()
        ? QString("<span style=\"color:#D6F3FF;\">%1</span>").arg(safe)
        : QString("<span style=\"color:%1;\">%2</span>").arg(color, safe);
    m_log->append(html);
}

void TrainingTab::setRunning(bool running)
{
    const bool anom = isAnomalibMode();
    m_btnCheck->setEnabled(!running);
    m_btnTrain->setEnabled(!running);
    m_btnVal->setEnabled(!running && !anom);
    m_btnExport->setEnabled(!running && !anom);
    m_btnStop->setEnabled(running);
}

void TrainingTab::updateModeUi()
{
    const bool anom = isAnomalibMode();
    if (m_titleLabel) {
        m_titleLabel->setText(anom
            ? "ANOMALIB TRAINING LAB · 异常检测训练 / ONNX 导出 / metadata"
            : "YOLO TRAINING LAB · 训练 / 验证 / ONNX 导出");
    }
    if (m_yoloTaskBox) m_yoloTaskBox->setVisible(!anom);
    if (m_anomalibTaskBox) m_anomalibTaskBox->setVisible(anom);
    if (m_yoloParamBox) m_yoloParamBox->setVisible(!anom);
    if (m_anomalibParamBox) m_anomalibParamBox->setVisible(anom);
    if (m_btnTrain) m_btnTrain->setText(anom ? "开始训练并导出" : "开始训练");
    if (m_btnVal) m_btnVal->setVisible(!anom);
    if (m_btnExport) m_btnExport->setVisible(!anom);
    if (m_lblArtifact) {
        m_lblArtifact->setText(anom
            ? "产物：model.ckpt / model.onnx / metadata.json 会显示在这里"
            : "产物：best.pt / ONNX / transd.onnx 会显示在这里");
    }
    if (m_lblLabelsJson) {
        m_lblLabelsJson->setVisible(!anom);
    }
    setRunning(m_controller && m_controller->isRunning());
}

void TrainingTab::updateInitHint()
{
    if (!m_lblInitHint) return;
    const QString mode = m_cbInitMode->currentData().toString();
    if (mode == "scratch") {
        m_lblInitHint->setText("从零训练：请使用 .yaml 模型结构，pretrained=false，适合大数据集。");
        m_edPretrained->setText("false");
    } else if (mode == "resume") {
        m_lblInitHint->setText("断点恢复：请选择 last.pt，脚本会追加 resume=true。");
    } else {
        m_lblInitHint->setText("推荐：本地 .pt 请优先放到“模型入口”。如果误填到 pretrained，开始训练时会自动同步，避免下载默认模型。");
        if (m_edPretrained && m_edPretrained->text().trimmed().isEmpty()) {
            m_edPretrained->setText("true");
        }
    }
}

} // namespace trtforge
