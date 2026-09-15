#include "engine/EngineBuilderTab.h"

#include "BuilderWorker.h"
#include "TRTLogger.h"
#include "TRTProgressMonitor.h"
#include "common/AppEventFormatter.h"
#include "common/AppEvents.h"
#include "common/AppFileLogger.h"
#include "engine/OnnxInputInspector.h"

#include <QAbstractSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProgressBar>
#include <QProgressDialog>
#include <QPushButton>
#include <QRegularExpression>
#include <QRegularExpressionMatch>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStyle>
#include <QTextBrowser>
#include <QTimer>
#include <QVBoxLayout>

#include <cuda_runtime_api.h>

namespace trtforge {

namespace {

QFrame* makeCard(QWidget* parent = nullptr)
{
    auto* f = new QFrame(parent);
    f->setObjectName("card");
    f->setFrameShape(QFrame::NoFrame);
    return f;
}

QLabel* makeCardTitle(const QString& text)
{
    auto* l = new QLabel(text);
    l->setObjectName("cardTitle");
    return l;
}

QLabel* makeFieldLabel(const QString& text)
{
    auto* l = new QLabel(text);
    l->setObjectName("fieldLabel");
    return l;
}

QStringList splitShape(const QString& text)
{
    return text.trimmed().split(QRegularExpression(QStringLiteral("[xX,]")), Qt::SkipEmptyParts);
}

} // namespace

EngineBuilderTab::EngineBuilderTab(QWidget* parent)
    : QWidget(parent)
{
    auto* hb = new QHBoxLayout(this);
    hb->setContentsMargins(16, 16, 16, 16);
    hb->setSpacing(16);

    auto* leftCol = new QWidget(this);
    leftCol->setFixedWidth(800);
    auto* leftColV = new QVBoxLayout(leftCol);
    leftColV->setContentsMargins(0, 0, 0, 0);
    leftColV->setSpacing(12);

    auto* leftWrap = new QWidget(leftCol);
    auto* leftScroll = new QScrollArea(leftCol);
    leftScroll->setWidgetResizable(true);
    leftScroll->setFrameShape(QFrame::NoFrame);
    leftScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* leftV = new QVBoxLayout(leftWrap);
    leftV->setContentsMargins(0, 0, 0, 0);
    leftV->setSpacing(14);
    leftV->addWidget(buildIoCard());
    leftV->addWidget(buildPrecisionCard());
    leftV->addWidget(buildBuildParamsCard());
    leftV->addWidget(buildShapesCard());
    leftV->addWidget(buildExtraCard());
    leftV->addWidget(buildTestCard());
    leftV->addStretch(1);
    leftScroll->setWidget(leftWrap);

    leftColV->addWidget(leftScroll, 1);
    leftColV->addWidget(buildActionRow(), 0);
    hb->addWidget(leftCol, 0);

    auto* rightV = new QVBoxLayout();
    rightV->setSpacing(14);
    rightV->addWidget(buildConsoleCard(), 1);
    rightV->addWidget(buildProgressCard(), 0);
    rightV->addWidget(buildTestReportCard(), 0);
    hb->addLayout(rightV, 1);

    m_worker = new BuilderWorker();
    m_thread = new QThread(this);
    m_worker->moveToThread(m_thread);
    connect(m_thread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(this, &EngineBuilderTab::destroyed, m_thread, [t = m_thread] {
        t->quit();
        t->wait();
    });
    m_thread->start();

    connect(m_worker, &BuilderWorker::log, this, &EngineBuilderTab::onLog, Qt::QueuedConnection);
    connect(m_worker, &BuilderWorker::finished, this, &EngineBuilderTab::onFinished, Qt::QueuedConnection);
    connect(m_worker, &BuilderWorker::testFinished, this, &EngineBuilderTab::onTestFinished, Qt::QueuedConnection);

    m_monitor = new TRTProgressMonitor(this);
    connect(m_monitor, &TRTProgressMonitor::progressChanged, this, &EngineBuilderTab::onProgress);
    connect(m_monitor, &TRTProgressMonitor::stageChanged, this, &EngineBuilderTab::onStage);
    connect(m_monitor, &TRTProgressMonitor::etaChanged, this, &EngineBuilderTab::onEta);

    m_uiTimer = new QTimer(this);
    m_uiTimer->setInterval(500);
    connect(m_uiTimer, &QTimer::timeout, this, &EngineBuilderTab::onElapsedTick);

    appendColored(static_cast<int>(LogLevel::Info),
        "DefectVision AI Studio 已就绪。请选择功能页开始标注、训练、部署或推理。");
}

EngineBuilderTab::~EngineBuilderTab()
{
    cancelOnnxInputInspection();
    if (m_onnxInspectThread && m_onnxInspectThread->isRunning()) {
        m_onnxInspectThread->requestInterruption();
        m_onnxInspectThread->wait(2000);
    }
    if (m_thread) {
        m_thread->quit();
        m_thread->wait(2000);
    }
}

QWidget* EngineBuilderTab::buildIoCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(12);
    v->addWidget(makeCardTitle("01  /  模型输入输出"));

    {
        auto* row = new QVBoxLayout();
        row->setSpacing(6);
        row->addWidget(makeFieldLabel("ONNX 模型路径"));
        auto* h = new QHBoxLayout();
        h->setSpacing(8);
        m_edOnnx = new QLineEdit();
        m_edOnnx->setPlaceholderText("D:/models/yolov8n.onnx");
        m_btnBrowse = new QPushButton("浏览");
        m_btnBrowse->setProperty("primary", true);
        connect(m_btnBrowse, &QPushButton::clicked, this, &EngineBuilderTab::onBrowseOnnx);
        h->addWidget(m_edOnnx, 1);
        h->addWidget(m_btnBrowse, 0);
        row->addLayout(h);
        v->addLayout(row);
    }
    {
        auto* row = new QVBoxLayout();
        row->setSpacing(6);
        row->addWidget(makeFieldLabel("输出引擎文件 (.trt)"));
        auto* h = new QHBoxLayout();
        h->setSpacing(8);
        m_edEngine = new QLineEdit();
        m_edEngine->setPlaceholderText("D:/models/yolov8n_fp16.trt");
        m_btnSaveAs = new QPushButton("另存为...");
        connect(m_btnSaveAs, &QPushButton::clicked, this, &EngineBuilderTab::onSaveAs);
        h->addWidget(m_edEngine, 1);
        h->addWidget(m_btnSaveAs, 0);
        row->addLayout(h);
        v->addLayout(row);
    }
    return card;
}

QWidget* EngineBuilderTab::buildPrecisionCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(12);
    v->addWidget(makeCardTitle("02  /  精度与运行时"));

    auto* h = new QHBoxLayout();
    h->setSpacing(12);
    {
        auto* col = new QVBoxLayout();
        col->setSpacing(6);
        col->addWidget(makeFieldLabel("精度模式"));
        m_cbPrec = new QComboBox();
        m_cbPrec->addItem("FP32 - 单精度", static_cast<int>(Precision::FP32));
        m_cbPrec->addItem("FP16 - 半精度 (推荐)", static_cast<int>(Precision::FP16));
        m_cbPrec->addItem("TF32 - Tensor Float32", static_cast<int>(Precision::TF32));
        m_cbPrec->setCurrentIndex(1);
        col->addWidget(m_cbPrec);
        h->addLayout(col, 1);
    }
    {
        auto* col = new QVBoxLayout();
        col->setSpacing(6);
        col->addWidget(makeFieldLabel("运行时模式"));
        m_cbRuntime = new QComboBox();
        m_cbRuntime->addItems({ "full (默认)", "lean", "dispatch" });
        col->addWidget(m_cbRuntime);
        h->addLayout(col, 1);
    }
    v->addLayout(h);

    auto* hint = new QLabel("提示：本工具不支持 INT8 标定，已移除 INT8 / BEST。如需 INT8 请使用 trtexec.exe 完成标定后再部署。");
    hint->setObjectName("fieldLabel");
    v->addWidget(hint);
    return card;
}

QWidget* EngineBuilderTab::buildBuildParamsCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(12);
    v->addWidget(makeCardTitle("03  /  构建参数"));

    auto* row = new QHBoxLayout();
    row->setSpacing(12);
    auto addInt = [&](const QString& label, QSpinBox*& sp, int minV, int maxV, int defV) {
        auto* col = new QVBoxLayout();
        col->setSpacing(6);
        col->addWidget(makeFieldLabel(label));
        sp = new QSpinBox();
        sp->setRange(minV, maxV);
        sp->setValue(defV);
        sp->setButtonSymbols(QAbstractSpinBox::NoButtons);
        col->addWidget(sp);
        row->addLayout(col, 1);
    };
    addInt("工作空间 (MiB)", m_spWorkspace, 64, 65536, 4096);
    addInt("最大 Batch", m_spMaxBatch, 1, 1024, 1);
    addInt("DLA 核心", m_spDla, -1, 3, -1);
    addInt("目标设备", m_spDevice, 0, 16, 0);
    v->addLayout(row);

    auto* chips = new QHBoxLayout();
    chips->setSpacing(8);
    auto addChip = [&](const QString& text, QCheckBox*& cb, bool def) {
        cb = new QCheckBox(text);
        cb->setChecked(def);
        chips->addWidget(cb);
    };
    addChip("稀疏化", m_chkSparsity, false);
    addChip("可重拟合", m_chkRefit, false);
    addChip("版本兼容", m_chkVerCompat, false);
    addChip("允许GPU回退", m_chkGpuFb, true);
    addChip("TF32", m_chkTf32, false);
    chips->addStretch(1);
    v->addLayout(chips);

    return card;
}

QWidget* EngineBuilderTab::buildShapesCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(12);
    v->addWidget(makeCardTitle("04  /  动态形状"));

    auto* h = new QHBoxLayout();
    h->setSpacing(12);
    {
        auto* col = new QVBoxLayout();
        col->setSpacing(6);
        col->addWidget(makeFieldLabel("输入名称"));
        m_edInputName = new QLineEdit("images");
        col->addWidget(m_edInputName);
        h->addLayout(col, 2);
    }
    auto addHw = [&](const QString& label, QSpinBox*& sp) {
        auto* col = new QVBoxLayout();
        col->setSpacing(6);
        col->addWidget(makeFieldLabel(label));
        sp = new QSpinBox();
        sp->setRange(1, 8192);
        sp->setValue(640);
        sp->setButtonSymbols(QAbstractSpinBox::NoButtons);
        col->addWidget(sp);
        h->addLayout(col, 1);
    };
    addHw("统一 H", m_spShapeH);
    addHw("统一 W", m_spShapeW);
    v->addLayout(h);

    m_chkSyncHw = new QCheckBox("同步 H/W 到 min / opt / max（保留各自 batch 和 channel）");
    m_chkSyncHw->setChecked(true);
    v->addWidget(m_chkSyncHw);
    connect(m_spShapeH, qOverload<int>(&QSpinBox::valueChanged), this, [this] { syncShapesFromHw(); });
    connect(m_spShapeW, qOverload<int>(&QSpinBox::valueChanged), this, [this] { syncShapesFromHw(); });
    connect(m_chkSyncHw, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) {
            syncShapesFromHw();
        }
    });

    auto* row = new QHBoxLayout();
    row->setSpacing(12);
    auto addShape = [&](const QString& label, QLineEdit*& ed, const QString& def, bool important) {
        auto* shapeCol = new QVBoxLayout();
        shapeCol->setSpacing(6);
        shapeCol->addWidget(makeFieldLabel(label));
        ed = new QLineEdit(def);
        if (important) {
            ed->setProperty("important", true);
        }
        shapeCol->addWidget(ed);
        row->addLayout(shapeCol, 1);
    };
    addShape("最小形状", m_edMin, "1x3x640x640", false);
    addShape("最优形状", m_edOpt, "4x3x640x640", true);
    addShape("最大形状", m_edMax, "8x3x640x640", false);
    v->addLayout(row);
    return card;
}

QWidget* EngineBuilderTab::buildExtraCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(8);
    v->addWidget(makeCardTitle("额外参数  (trtexec --help 风格 token，留空即可)"));
    m_edExtra = new QPlainTextEdit();
    m_edExtra->setPlaceholderText("--useCudaGraph --calib=D:/calib.cache --timingCacheFile=cache.bin ...");
    m_edExtra->setFixedHeight(70);
    v->addWidget(m_edExtra);
    return card;
}

QWidget* EngineBuilderTab::buildTestCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(10);
    v->addWidget(makeCardTitle("05  /  引擎自检  (随机数据 + 性能基线)"));

    auto* h = new QHBoxLayout();
    h->setSpacing(12);
    m_chkTestAuto = new QCheckBox("构建后自动自检");
    m_chkTestAuto->setChecked(true);
    h->addWidget(m_chkTestAuto, 0);

    {
        auto* col = new QVBoxLayout();
        col->setSpacing(4);
        col->addWidget(makeFieldLabel("Warmup"));
        m_spTestWarmup = new QSpinBox();
        m_spTestWarmup->setRange(0, 1000);
        m_spTestWarmup->setValue(10);
        m_spTestWarmup->setButtonSymbols(QAbstractSpinBox::NoButtons);
        col->addWidget(m_spTestWarmup);
        h->addLayout(col, 1);
    }
    {
        auto* col = new QVBoxLayout();
        col->setSpacing(4);
        col->addWidget(makeFieldLabel("Iterations"));
        m_spTestIter = new QSpinBox();
        m_spTestIter->setRange(1, 100000);
        m_spTestIter->setValue(100);
        m_spTestIter->setButtonSymbols(QAbstractSpinBox::NoButtons);
        col->addWidget(m_spTestIter);
        h->addLayout(col, 1);
    }
    m_btnTestExist = new QPushButton("对已有 .trt 自检");
    m_btnTestExist->setProperty("ghost", true);
    connect(m_btnTestExist, &QPushButton::clicked, this, &EngineBuilderTab::onTestExistingClicked);
    h->addWidget(m_btnTestExist, 0);
    v->addLayout(h);

    auto* hint = new QLabel("说明：使用随机张量数据进行冒烟测试与时延/吞吐统计；不做精度对比。");
    hint->setObjectName("fieldLabel");
    v->addWidget(hint);
    return card;
}

QWidget* EngineBuilderTab::buildTestReportCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(8);

    auto* head = new QHBoxLayout();
    head->addWidget(makeCardTitle("自检报告"));
    head->addStretch(1);
    m_lblTestSummary = new QLabel("尚未自检");
    m_lblTestSummary->setStyleSheet("color:#9FC8E8; font-size:12px; letter-spacing:1px;");
    head->addWidget(m_lblTestSummary);
    v->addLayout(head);

    m_edTestDetail = new QPlainTextEdit();
    m_edTestDetail->setReadOnly(true);
    m_edTestDetail->setFixedHeight(120);
    m_edTestDetail->setPlaceholderText("张量列表 / 性能 / 输出预览");
    v->addWidget(m_edTestDetail);
    return card;
}

QWidget* EngineBuilderTab::buildActionRow()
{
    auto* w = new QWidget(this);
    auto* h = new QHBoxLayout(w);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(12);
    m_btnReset = new QPushButton("重置");
    m_btnReset->setProperty("ghost", true);
    m_btnBuild = new QPushButton("▶  构建引擎");
    m_btnBuild->setProperty("primary", true);
    connect(m_btnReset, &QPushButton::clicked, this, &EngineBuilderTab::onResetClicked);
    connect(m_btnBuild, &QPushButton::clicked, this, &EngineBuilderTab::onBuildClicked);
    h->addWidget(m_btnReset, 0);
    h->addWidget(m_btnBuild, 1);
    return w;
}

QWidget* EngineBuilderTab::buildConsoleCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);

    auto* head = new QFrame();
    head->setStyleSheet("background-color:#0F1A2E; border-bottom:1px solid #16334D;");
    auto* hh = new QHBoxLayout(head);
    hh->setContentsMargins(16, 10, 16, 10);
    hh->addWidget(makeCardTitle("实时日志  /  trtexec 输出"));
    hh->addStretch(1);
    m_btnClear = new QPushButton("清空");
    m_btnClear->setProperty("ghost", true);
    connect(m_btnClear, &QPushButton::clicked, this, &EngineBuilderTab::onClearLog);
    hh->addWidget(m_btnClear);
    v->addWidget(head);

    m_logView = new QTextBrowser();
    m_logView->setObjectName("logView");
    m_logView->setOpenExternalLinks(false);
    v->addWidget(m_logView, 1);
    return card;
}

QWidget* EngineBuilderTab::buildProgressCard()
{
    auto* card = makeCard(this);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(16, 16, 16, 16);
    v->setSpacing(14);

    auto* head = new QHBoxLayout();
    head->addWidget(makeCardTitle("构建进度"));
    head->addStretch(1);
    m_lblPercent = new QLabel("0%");
    m_lblPercent->setStyleSheet("color:#1FB6FF; font-size:22px; font-weight:700; letter-spacing:2px;");
    head->addWidget(m_lblPercent);
    v->addLayout(head);

    m_progress = new QProgressBar();
    m_progress->setRange(0, 1000);
    m_progress->setValue(0);
    m_progress->setTextVisible(false);
    v->addWidget(m_progress);

    auto* stages = new QHBoxLayout();
    const QStringList names = { "解析", "优化", "调优", "序列化", "保存" };
    for (int i = 0; i < 5; ++i) {
        auto* col = new QVBoxLayout();
        col->setAlignment(Qt::AlignHCenter);
        col->setSpacing(4);
        auto* dot = new QLabel();
        dot->setObjectName("stageDot");
        dot->setProperty("stage", "pending");
        dot->setAlignment(Qt::AlignHCenter);
        auto* tx = new QLabel(names[i]);
        tx->setProperty("stage", "pending");
        tx->setAlignment(Qt::AlignHCenter);
        col->addWidget(dot, 0, Qt::AlignHCenter);
        col->addWidget(tx, 0, Qt::AlignHCenter);
        m_stageDots[i] = dot;
        m_stageTexts[i] = tx;
        auto* w = new QWidget();
        w->setLayout(col);
        stages->addWidget(w, 1);
    }
    v->addLayout(stages);

    auto* metrics = new QHBoxLayout();
    metrics->setSpacing(10);
    auto addMetric = [&](const QString& label, QLabel*& valLbl, const QString& accent = "") {
        auto* mf = new QFrame();
        mf->setObjectName("metric");
        auto* mv = new QVBoxLayout(mf);
        mv->setContentsMargins(10, 8, 10, 8);
        mv->setSpacing(2);
        auto* l = new QLabel(label);
        l->setObjectName("metricLabel");
        valLbl = new QLabel("--");
        valLbl->setObjectName("metricValue");
        if (!accent.isEmpty()) {
            valLbl->setProperty("accent", accent);
        }
        mv->addWidget(l);
        mv->addWidget(valLbl);
        metrics->addWidget(mf, 1);
    };
    addMetric("已用时", m_lblElapsed);
    addMetric("预计剩余", m_lblEta, "cyan");
    addMetric("层", m_lblLayer);
    addMetric("显存", m_lblVram, "green");
    v->addLayout(metrics);

    m_lblStageNote = new QLabel("当前阶段：等待开始");
    m_lblStageNote->setStyleSheet("color:#9FC8E8; letter-spacing:1px;");
    v->addWidget(m_lblStageNote);
    return card;
}

void EngineBuilderTab::onBrowseOnnx()
{
    QString f = QFileDialog::getOpenFileName(this, "选择 ONNX 模型",
        m_edOnnx->text(), "ONNX 模型 (*.onnx);;所有文件 (*.*)");
    if (f.isEmpty()) {
        return;
    }
    m_edOnnx->setText(QDir::toNativeSeparators(f));
    if (m_edEngine->text().isEmpty()) {
        QFileInfo fi(f);
        QString def = fi.absolutePath() + "/" + fi.completeBaseName() + ".trt";
        m_edEngine->setText(QDir::toNativeSeparators(def));
    }
    startOnnxInputInspection(f);
}

void EngineBuilderTab::onSaveAs()
{
    QString f = QFileDialog::getSaveFileName(this, "保存 TensorRT 引擎",
        m_edEngine->text(), "TensorRT 引擎 (*.trt *.engine);;所有文件 (*.*)");
    if (f.isEmpty()) {
        return;
    }
    m_edEngine->setText(QDir::toNativeSeparators(f));
}

void EngineBuilderTab::onResetClicked()
{
    BuildOptions def;
    applyOptions(def);
    appendColored(static_cast<int>(LogLevel::Info), "已重置为默认参数。");
}

void EngineBuilderTab::onBuildClicked()
{
    BuildOptions opt;
    collectOptions(opt);
    if (opt.onnxPath.isEmpty()) {
        QMessageBox::warning(this, "提示", "请先选择 ONNX 模型文件");
        return;
    }
    if (opt.enginePath.isEmpty()) {
        QMessageBox::warning(this, "提示", "请指定输出引擎路径");
        return;
    }
    m_logView->clear();
    appendColored(static_cast<int>(LogLevel::Stage),
        QString("开始构建：%1  →  %2").arg(opt.onnxPath, opt.enginePath));

    m_startMs = QDateTime::currentMSecsSinceEpoch();
    m_etaMs = -1;
    m_layerCount = 0;
    m_lblLayer->setText("--");
    setBuildingUi(true);
    m_monitor->start();
    m_uiTimer->start();
    QMetaObject::invokeMethod(m_worker, "start", Qt::QueuedConnection,
        Q_ARG(trtforge::BuildOptions, opt));
}

void EngineBuilderTab::onClearLog()
{
    if (m_logView) {
        m_logView->clear();
    }
}

void EngineBuilderTab::onTestExistingClicked()
{
    QString path = QFileDialog::getOpenFileName(this, "选择 TensorRT 引擎",
        m_edEngine->text(), "TensorRT 引擎 (*.trt *.engine);;所有文件 (*.*)");
    if (path.isEmpty()) {
        return;
    }
    appendColored(static_cast<int>(LogLevel::Stage),
        QString("对已有引擎执行自检：%1").arg(path));
    if (m_lblTestSummary) {
        m_lblTestSummary->setText("自检中...");
    }
    if (m_edTestDetail) {
        m_edTestDetail->clear();
    }
    QMetaObject::invokeMethod(m_worker, "testEngineOnly", Qt::QueuedConnection,
        Q_ARG(QString, path),
        Q_ARG(int, m_spTestWarmup ? m_spTestWarmup->value() : 10),
        Q_ARG(int, m_spTestIter ? m_spTestIter->value() : 100),
        Q_ARG(int, m_spDevice ? m_spDevice->value() : 0),
        Q_ARG(QString, m_edInputName ? m_edInputName->text() : QString("images")),
        Q_ARG(QString, m_edOpt ? m_edOpt->text() : QString()));
}

void EngineBuilderTab::onTestFinished(trtforge::TestReport report)
{
    if (m_lblTestSummary) {
        if (report.ok) {
            m_lblTestSummary->setText(QString("✔ avg %1 ms | %2 QPS | VRAM %3 GB")
                .arg(report.avgMs, 0, 'f', 3)
                .arg(report.qps, 0, 'f', 1)
                .arg(report.vramUsedGB, 0, 'f', 2));
            m_lblTestSummary->setStyleSheet("color:#22E27F; font-size:12px; font-weight:700; letter-spacing:1px;");
        } else {
            m_lblTestSummary->setText(QString("✘ %1").arg(report.message));
            m_lblTestSummary->setStyleSheet("color:#FF4D6D; font-size:12px; font-weight:700; letter-spacing:1px;");
        }
    }
    if (m_edTestDetail) {
        QStringList lines;
        lines << QString("引擎：%1").arg(report.enginePath);
        if (report.ok) {
            lines << QString("迭代：warmup=%1, iters=%2").arg(report.warmup).arg(report.iterations);
            lines << QString("时延 min/avg/max = %1 / %2 / %3 ms")
                         .arg(report.minMs, 0, 'f', 3).arg(report.avgMs, 0, 'f', 3).arg(report.maxMs, 0, 'f', 3);
            lines << QString("吞吐：%1 QPS    显存：%2 GB").arg(report.qps, 0, 'f', 1).arg(report.vramUsedGB, 0, 'f', 2);
            lines << QString("张量：");
            for (const QString& t : report.tensorLines) {
                lines << QString("  %1").arg(t);
            }
            if (!report.firstOutputPreview.isEmpty()) {
                lines << QString("输出 %1 前 %2 个值：%3")
                             .arg(report.firstOutputName)
                             .arg(report.firstOutputPreview.size())
                             .arg(report.firstOutputPreview.join(", "));
            }
        } else {
            lines << QString("失败原因：%1").arg(report.message);
        }
        m_edTestDetail->setPlainText(lines.join('\n'));
    }
}

void EngineBuilderTab::startOnnxInputInspection(const QString& onnxPath)
{
    cancelOnnxInputInspection();

    const int requestId = ++m_onnxInspectRequestId;
    const QString nativePath = QDir::toNativeSeparators(onnxPath);
    showOnnxLoadingProgress(nativePath);
    setOnnxInspectUi(true);
    appendColored(static_cast<int>(LogLevel::Stage),
        QString("开始识别 ONNX 输入：%1").arg(nativePath));

    QPointer<EngineBuilderTab> self(this);
    QThread* thread = QThread::create([self, requestId, nativePath]() {
        const OnnxInputInfo input = OnnxInputInspector::inspect(nativePath);
        if (!self) {
            return;
        }
        QMetaObject::invokeMethod(self, [self, requestId, nativePath, input]() {
            if (!self) {
                return;
            }
            self->finishOnnxInputInspection(requestId, nativePath, input);
        }, Qt::QueuedConnection);
    });
    m_onnxInspectThread = thread;
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    connect(thread, &QThread::finished, this, [this, thread]() {
        if (m_onnxInspectThread == thread) {
            m_onnxInspectThread = nullptr;
        }
    });
    thread->start();
}

void EngineBuilderTab::finishOnnxInputInspection(int requestId,
                                                const QString& onnxPath,
                                                const OnnxInputInfo& input)
{
    if (requestId != m_onnxInspectRequestId || !m_onnxInspecting) {
        return;
    }

    m_onnxInspecting = false;
    hideOnnxLoadingProgress();
    setOnnxInspectUi(false);

    if (!input.ok) {
        appendColored(static_cast<int>(LogLevel::Warning),
            QString("ONNX 输入自动识别失败：%1。请手动填写输入名称和动态 Shape。").arg(input.error));
        return;
    }

    applyOnnxInputInfo(input);
    QStringList dimText;
    for (int dim : input.dims) {
        dimText << QString::number(dim);
    }
    appendColored(static_cast<int>(LogLevel::Info),
        QString("已识别 ONNX 输入：%1 [%2]，来源：%3").arg(input.name, dimText.join('x'), onnxPath));
}

void EngineBuilderTab::cancelOnnxInputInspection()
{
    if (!m_onnxInspecting && !m_onnxLoadingProgress) {
        return;
    }
    ++m_onnxInspectRequestId;
    m_onnxInspecting = false;
    hideOnnxLoadingProgress();
    setOnnxInspectUi(false);
}

void EngineBuilderTab::showOnnxLoadingProgress(const QString& onnxPath)
{
    hideOnnxLoadingProgress();
    m_onnxLoadingProgress = new QProgressDialog(
        QStringLiteral("正在加载 ONNX 模型并识别输入，请稍候...\n%1").arg(onnxPath),
        QStringLiteral("取消"),
        0,
        0,
        this);
    m_onnxLoadingProgress->setWindowTitle(QStringLiteral("ONNX 加载中"));
    m_onnxLoadingProgress->setWindowModality(Qt::WindowModal);
    m_onnxLoadingProgress->setMinimumDuration(0);
    m_onnxLoadingProgress->setAutoClose(false);
    m_onnxLoadingProgress->setAutoReset(false);
    connect(m_onnxLoadingProgress, &QProgressDialog::canceled,
            this, &EngineBuilderTab::cancelOnnxInputInspection);
    m_onnxLoadingProgress->show();
}

void EngineBuilderTab::hideOnnxLoadingProgress()
{
    if (!m_onnxLoadingProgress) {
        return;
    }
    disconnect(m_onnxLoadingProgress, nullptr, this, nullptr);
    m_onnxLoadingProgress->hide();
    m_onnxLoadingProgress->deleteLater();
    m_onnxLoadingProgress = nullptr;
}

void EngineBuilderTab::setOnnxInspectUi(bool inspecting)
{
    m_onnxInspecting = inspecting;
    if (m_btnBrowse) m_btnBrowse->setEnabled(!inspecting);
    if (m_btnBuild) m_btnBuild->setEnabled(!inspecting);
    if (m_btnSaveAs) m_btnSaveAs->setEnabled(!inspecting);
    if (m_btnReset) m_btnReset->setEnabled(!inspecting);
    if (m_btnBuild) m_btnBuild->setText(inspecting ? "识别 ONNX 中..." : "▶  构建引擎");
}

void EngineBuilderTab::applyOnnxInputInfo(const OnnxInputInfo& input)
{
    if (!input.name.trimmed().isEmpty()) {
        m_edInputName->setText(input.name.trimmed());
    }

    const int channel = input.channel(3);
    const int height = input.height(m_spShapeH ? m_spShapeH->value() : 640);
    const int width = input.width(m_spShapeW ? m_spShapeW->value() : 640);
    if (m_spShapeH && m_spShapeW) {
        const QSignalBlocker hBlocker(m_spShapeH);
        const QSignalBlocker wBlocker(m_spShapeW);
        m_spShapeH->setValue(height);
        m_spShapeW->setValue(width);
    }

    if (m_edMin && m_edOpt && m_edMax && m_chkSyncHw && m_chkSyncHw->isChecked()) {
        m_edMin->setText(QStringLiteral("1x%1x%2x%3").arg(channel).arg(height).arg(width));
        m_edOpt->setText(QStringLiteral("4x%1x%2x%3").arg(channel).arg(height).arg(width));
        m_edMax->setText(QStringLiteral("8x%1x%2x%3").arg(channel).arg(height).arg(width));
    }

}

QString EngineBuilderTab::shapeWithHw(const QString& currentShape, int fallbackBatch) const
{
    QStringList parts = splitShape(currentShape);
    const QString fallbackChannel = QStringLiteral("3");
    while (parts.size() < 4) {
        if (parts.isEmpty()) {
            parts << QString::number(fallbackBatch);
        } else if (parts.size() == 1) {
            parts << fallbackChannel;
        } else {
            parts << QStringLiteral("640");
        }
    }
    parts[0] = parts.value(0).trimmed().isEmpty() ? QString::number(fallbackBatch) : parts.value(0).trimmed();
    parts[1] = parts.value(1).trimmed().isEmpty() ? fallbackChannel : parts.value(1).trimmed();
    parts[2] = QString::number(m_spShapeH ? m_spShapeH->value() : 640);
    parts[3] = QString::number(m_spShapeW ? m_spShapeW->value() : 640);
    return parts.mid(0, 4).join('x');
}

void EngineBuilderTab::syncShapesFromHw()
{
    if (!m_chkSyncHw || !m_chkSyncHw->isChecked() || !m_edMin || !m_edOpt || !m_edMax) {
        return;
    }
    m_edMin->setText(shapeWithHw(m_edMin->text(), 1));
    m_edOpt->setText(shapeWithHw(m_edOpt->text(), 4));
    m_edMax->setText(shapeWithHw(m_edMax->text(), 8));
}

void EngineBuilderTab::onLog(int level, QString text)
{
    const AppLogEvent logEvent { AppEventFormatter::fromTrtLogLevel(level), AppEventSource::Engine, text };
    appendColored(AppEventFormatter::toTrtLogLevel(logEvent.level), logEvent.message);
    if (m_monitor) {
        m_monitor->onLog(level, logEvent.message);
    }

    static const QRegularExpression rxLayers(R"(([0-9]+)\s+layers)", QRegularExpression::CaseInsensitiveOption);
    auto m = rxLayers.match(text);
    if (m.hasMatch()) {
        m_layerCount = m.captured(1).toInt();
        if (m_lblLayer) {
            m_lblLayer->setText(QString::number(m_layerCount));
        }
    }
}

void EngineBuilderTab::onFinished(bool ok, QString enginePath, qint64 elapsedMs)
{
    m_monitor->finish(ok);
    m_uiTimer->stop();
    setBuildingUi(false);
    if (ok) {
        appendColored(static_cast<int>(LogLevel::Success),
            QString("✔ 引擎已生成：%1（耗时 %2 ms）").arg(enginePath).arg(elapsedMs));
        QMessageBox::information(this, "完成", QString("引擎已生成：\n%1\n耗时 %2 ms").arg(enginePath).arg(elapsedMs));
    } else {
        appendColored(static_cast<int>(LogLevel::Error),
            QString("✘ 构建失败（耗时 %1 ms）").arg(elapsedMs));
    }
}

void EngineBuilderTab::onProgress(int permille)
{
    if (!m_progress) {
        return;
    }
    m_progress->setValue(permille);
    m_lblPercent->setText(QString::number(permille / 10) + "%");
}

void EngineBuilderTab::onStage(int stageInt)
{
    using S = TRTProgressMonitor::Stage;
    auto s = static_cast<S>(stageInt);
    auto setS = [&](int idx, const char* state) {
        if (idx < 0 || idx >= 5) {
            return;
        }
        m_stageDots[idx]->setProperty("stage", state);
        m_stageTexts[idx]->setProperty("stage", state);
        m_stageDots[idx]->style()->unpolish(m_stageDots[idx]);
        m_stageDots[idx]->style()->polish(m_stageDots[idx]);
        m_stageTexts[idx]->style()->unpolish(m_stageTexts[idx]);
        m_stageTexts[idx]->style()->polish(m_stageTexts[idx]);
    };
    int active = -1;
    switch (s) {
        case S::Parse: active = 0; break;
        case S::Optimize: active = 1; break;
        case S::Tune: active = 2; break;
        case S::Serialize: active = 3; break;
        case S::Save: active = 4; break;
        case S::Done: active = 5; break;
        default: break;
    }
    for (int i = 0; i < 5; ++i) {
        if (i < active) {
            setS(i, "done");
        } else if (i == active) {
            setS(i, "active");
        } else {
            setS(i, "pending");
        }
    }
    if (s == S::Done) {
        for (int i = 0; i < 5; ++i) {
            setS(i, "done");
        }
    }

    static const QStringList names = { "解析", "优化", "调优", "序列化", "保存", "完成", "失败" };
    int idx = qBound(0, stageInt - 1, names.size() - 1);
    if (s == S::Done) {
        idx = 5;
    }
    if (s == S::Failed) {
        idx = 6;
    }
    if (m_lblStageNote) {
        m_lblStageNote->setText(QString("当前阶段：%1").arg(names.value(idx, "--")));
    }
}

void EngineBuilderTab::onEta(qint64 etaMs)
{
    m_etaMs = etaMs;
    if (etaMs < 0) {
        m_lblEta->setText("--");
    } else {
        m_lblEta->setText(formatHms(etaMs));
    }
}

void EngineBuilderTab::onElapsedTick()
{
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    qint64 dt = now - m_startMs;
    if (m_lblElapsed) {
        m_lblElapsed->setText(formatHms(dt));
    }

    size_t freeB = 0;
    size_t totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess && totalB > 0) {
        double usedGB = double(totalB - freeB) / 1024.0 / 1024.0 / 1024.0;
        m_lblVram->setText(QString::asprintf("%.2f GB", usedGB));
    }
}

void EngineBuilderTab::collectOptions(BuildOptions& opt) const
{
    opt.onnxPath = m_edOnnx->text().trimmed();
    opt.enginePath = m_edEngine->text().trimmed();
    opt.precision = static_cast<Precision>(m_cbPrec->currentData().toInt());
    opt.runtime = static_cast<RuntimeMode>(m_cbRuntime->currentIndex());
    opt.workspaceMiB = m_spWorkspace->value();
    opt.maxBatch = m_spMaxBatch->value();
    opt.dlaCore = m_spDla->value();
    opt.device = m_spDevice->value();
    opt.sparsity = m_chkSparsity->isChecked();
    opt.refit = m_chkRefit->isChecked();
    opt.versionCompatible = m_chkVerCompat->isChecked();
    opt.gpuFallback = m_chkGpuFb->isChecked();
    opt.tf32 = m_chkTf32->isChecked();
    opt.shapeInputName = m_edInputName->text().trimmed();
    opt.minShape = m_edMin->text().trimmed();
    opt.optShape = m_edOpt->text().trimmed();
    opt.maxShape = m_edMax->text().trimmed();
    opt.extraArgs = m_edExtra->toPlainText();
    opt.testEnabled = m_chkTestAuto ? m_chkTestAuto->isChecked() : true;
    opt.testWarmup = m_spTestWarmup ? m_spTestWarmup->value() : 10;
    opt.testIterations = m_spTestIter ? m_spTestIter->value() : 100;
}

void EngineBuilderTab::applyOptions(const BuildOptions& opt)
{
    int idx = m_cbPrec->findData(static_cast<int>(opt.precision));
    if (idx < 0) {
        idx = m_cbPrec->findData(static_cast<int>(Precision::FP16));
    }
    if (idx < 0) {
        idx = 0;
    }
    m_cbPrec->setCurrentIndex(idx);
    m_cbRuntime->setCurrentIndex(static_cast<int>(opt.runtime));
    m_spWorkspace->setValue(opt.workspaceMiB);
    m_spMaxBatch->setValue(opt.maxBatch);
    m_spDla->setValue(opt.dlaCore);
    m_spDevice->setValue(opt.device);
    m_chkSparsity->setChecked(opt.sparsity);
    m_chkRefit->setChecked(opt.refit);
    m_chkVerCompat->setChecked(opt.versionCompatible);
    m_chkGpuFb->setChecked(opt.gpuFallback);
    m_chkTf32->setChecked(opt.tf32);
    m_edInputName->setText(opt.shapeInputName);
    m_edMin->setText(opt.minShape);
    m_edOpt->setText(opt.optShape);
    m_edMax->setText(opt.maxShape);
    m_edExtra->setPlainText(opt.extraArgs);
    if (m_chkTestAuto) {
        m_chkTestAuto->setChecked(opt.testEnabled);
    }
    if (m_spTestWarmup) {
        m_spTestWarmup->setValue(opt.testWarmup);
    }
    if (m_spTestIter) {
        m_spTestIter->setValue(opt.testIterations);
    }
}

void EngineBuilderTab::setBuildingUi(bool building)
{
    m_btnBuild->setEnabled(!building);
    m_btnBrowse->setEnabled(!building);
    m_btnSaveAs->setEnabled(!building);
    m_btnReset->setEnabled(!building);
    m_btnBuild->setText(building ? "构建中..." : "▶  构建引擎");
}

void EngineBuilderTab::appendColored(int level, const QString& text)
{
    const AppLogLevel appLevel = AppEventFormatter::fromTrtLogLevel(level);
    AppFileLogger::append({ appLevel, AppEventSource::Engine, text });
    if (!m_logView) {
        return;
    }
    const QString color = AppEventFormatter::logColor(appLevel, QStringLiteral("#9FC8E8"));
    const QString prefix = AppEventFormatter::logPrefix(appLevel);
    QString safe = text.toHtmlEscaped();
    QString ts = QDateTime::currentDateTime().toString("HH:mm:ss");
    QString html = QString("<span style='color:%1'>%2 %3 %4</span>").arg(color, ts, prefix, safe);
    m_logView->append(html);
}

QString EngineBuilderTab::formatHms(qint64 ms) const
{
    if (ms < 0) {
        return "--";
    }
    qint64 s = ms / 1000;
    qint64 hh = s / 3600;
    qint64 mm = (s / 60) % 60;
    qint64 ss = s % 60;
    return QString::asprintf("%02lld : %02lld : %02lld", hh, mm, ss);
}

} // namespace trtforge
