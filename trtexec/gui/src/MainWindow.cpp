#include "MainWindow.h"

#include "AnnotationTab.h"
#include "InferenceTab.h"
#include "TrainingTab.h"
#include "engine/EngineBuilderTab.h"

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QTabWidget>
#include <QVBoxLayout>

#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

namespace trtforge {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("DefectVision AI Studio - Industrial AI Defect Inspection System");
    resize(1320, 880);
    setMinimumSize(1100, 640);

    auto* central = new QWidget(this);
    central->setObjectName("centralRoot");
    setCentralWidget(central);

    auto* root = new QVBoxLayout(central);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);
    root->addWidget(buildHeader());

    auto* tabs = new QTabWidget(central);
    tabs->setObjectName("mainTabs");
    tabs->setDocumentMode(true);
    root->addWidget(tabs, 1);

    tabs->addTab(new EngineBuilderTab(tabs), "ENGINE BUILDER · 引擎构建");
    tabs->addTab(new AnnotationTab(tabs), "ANNOTATION DATASET · 标注数据");
    tabs->addTab(new TrainingTab(tabs), "YOLO TRAINING LAB · 训练导出");
    tabs->addTab(new InferenceTab(tabs), "INFERENCE LAB · 推理演示");

    refreshGpuStatus();
}

QWidget* MainWindow::buildHeader()
{
    auto* bar = new QFrame(this);
    bar->setObjectName("headerBar");
    bar->setFixedHeight(64);

    auto* hl = new QHBoxLayout(bar);
    hl->setContentsMargins(20, 0, 20, 0);
    hl->setSpacing(16);

    auto* logo = new QLabel("AI");
    logo->setObjectName("logoBadge");
    hl->addWidget(logo);

    auto* titleCol = new QVBoxLayout();
    titleCol->setSpacing(2);
    auto* tMain = new QLabel("DEFECTVISION AI");
    tMain->setObjectName("titleMain");
    auto* tSub = new QLabel("工业 AI 缺陷检测 · 标注 / 训练 / 部署 / 推理");
    tSub->setObjectName("titleSub");
    titleCol->addWidget(tMain);
    titleCol->addWidget(tSub);
    hl->addLayout(titleCol);

    hl->addStretch(1);

    m_lblGpuStatus = new QLabel("GPU 检测中...");
    m_lblGpuStatus->setObjectName("statusPill");
    hl->addWidget(m_lblGpuStatus);

    return bar;
}

void MainWindow::refreshGpuStatus()
{
    int devCount = 0;
    QString text;
    if (cudaGetDeviceCount(&devCount) == cudaSuccess && devCount > 0) {
        cudaDeviceProp prop {};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            text = QString("GPU 就绪  |  %1  |  CUDA %2.%3  |  TRT %4.%5.%6")
                       .arg(prop.name)
                       .arg(CUDART_VERSION / 1000)
                       .arg((CUDART_VERSION % 1000) / 10)
                       .arg(NV_TENSORRT_MAJOR)
                       .arg(NV_TENSORRT_MINOR)
                       .arg(NV_TENSORRT_PATCH);
        }
    }
    if (text.isEmpty()) {
        text = "未检测到 CUDA 设备";
    }
    m_lblGpuStatus->setText(text);
}

} // namespace trtforge
