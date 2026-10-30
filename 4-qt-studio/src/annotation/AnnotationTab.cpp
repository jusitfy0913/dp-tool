#include "AnnotationTab.h"

#include "AnnotationConfirmDialog.h"
#include "DatasetExporter.h"
#include "common/AppSettings.h"
#include "dataset/DatasetSplitPolicy.h"

#include <QAbstractSpinBox>
#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QImageReader>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListView>
#include <QMessageBox>
#include <QPushButton>
#include <QRandomGenerator>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTextEdit>
#include <QVBoxLayout>
#include <algorithm>

namespace trtforge {

namespace {

QFrame* makeCard()
{
    auto* f = new QFrame();
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

QPushButton* smallButton(const QString& text, bool primary = false)
{
    auto* b = new QPushButton(text);
    b->setProperty(primary ? "primary" : "ghost", true);
    return b;
}

QSpinBox* makePercentSpin(int value)
{
    auto* spin = new QSpinBox();
    spin->setRange(0, 100);
    spin->setValue(value);
    spin->setSuffix("%");
    spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    spin->setFixedWidth(58);
    return spin;
}

QString shapeLabel(AnnotationShape shape)
{
    switch (shape) {
    case AnnotationShape::Rectangle: return QStringLiteral("Rect");
    case AnnotationShape::RotatedRect: return QStringLiteral("OBB");
    case AnnotationShape::Polygon: return QStringLiteral("Polygon");
    case AnnotationShape::Line: return QStringLiteral("Line");
    case AnnotationShape::Point: return QStringLiteral("Point");
    case AnnotationShape::Keypoints: return QStringLiteral("COCO17");
    }
    return QStringLiteral("Object");
}

} // namespace

AnnotationTab::AnnotationTab(QWidget* parent) : QWidget(parent)
{
    m_project.clear();

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(12);
    root->addWidget(buildToolbar(), 0);

    auto* center = new QWidget(this);
    auto* centerLayout = new QHBoxLayout(center);
    centerLayout->setContentsMargins(0, 0, 0, 0);
    centerLayout->setSpacing(12);
    centerLayout->addWidget(buildLeftTools(), 0);

    m_canvas = new AnnotationCanvas();
    m_canvas->setObjectName("annotationCanvas");
    m_canvas->setClasses(&m_project.classes);
    connect(m_canvas, &AnnotationCanvas::draftCompleted, this, &AnnotationTab::onDraftCompleted);
    connect(m_canvas, &AnnotationCanvas::objectSelected, this, &AnnotationTab::onCanvasObjectSelected);
    connect(m_canvas, &AnnotationCanvas::annotationsEdited, this, &AnnotationTab::onCanvasAnnotationsEdited);
    connect(m_canvas, &AnnotationCanvas::editStarted, this, &AnnotationTab::onCanvasEditStarted);
    centerLayout->addWidget(m_canvas, 1);
    setToolButton(m_toolButtons.value(2), AnnotationTool::Rectangle);
    centerLayout->addWidget(buildRightPanel(), 0);

    root->addWidget(center, 1);
    root->addWidget(buildBottomPanel(), 0);

    loadSplitRatioSettings();
    refreshAll();
}

QWidget* AnnotationTab::buildToolbar()
{
    auto* bar = makeCard();
    auto* h = new QHBoxLayout(bar);
    h->setContentsMargins(12, 10, 12, 10);
    h->setSpacing(8);

    m_projectInfo = new QLabel(QStringLiteral("TRTForge 标注项目 · .tfanno"));
    m_projectInfo->setObjectName("cardTitle");
    h->addWidget(m_projectInfo, 0);

    m_cbTask = new QComboBox();
    m_cbTask->addItem("Detect 目标检测", static_cast<int>(AnnotationTask::Detect));
    m_cbTask->addItem("Segment 实例分割", static_cast<int>(AnnotationTask::Segment));
    m_cbTask->addItem("Pose COCO17 姿态", static_cast<int>(AnnotationTask::Pose));
    m_cbTask->addItem("OBB 旋转框", static_cast<int>(AnnotationTask::Obb));
    connect(m_cbTask, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] {
        if (!m_project.images.isEmpty()) {
            auto ret = QMessageBox::question(this, QStringLiteral("切换任务"),
                QStringLiteral("切换任务类型可能导致现有标注和导出格式不兼容，是否继续？"));
            if (ret != QMessageBox::Yes) {
                m_cbTask->blockSignals(true);
                m_cbTask->setCurrentIndex(static_cast<int>(m_project.task));
                m_cbTask->blockSignals(false);
                return;
            }
        }
        m_project.task = static_cast<AnnotationTask>(m_cbTask->currentData().toInt());
        m_project.dirty = true;
        refreshAll();
    });
    h->addWidget(m_cbTask, 0);
    h->addStretch(1);

    auto add = [&](const QString& text, auto slot, bool primary = false) {
        auto* b = smallButton(text, primary);
        connect(b, &QPushButton::clicked, this, slot);
        h->addWidget(b);
    };
    add("新建", &AnnotationTab::onNewProject);
    add("打开 .tfanno", &AnnotationTab::onOpenProject);
    add("保存", &AnnotationTab::onSaveProject);
    add("导入图片", &AnnotationTab::onImportImages);
    add("导入标注", &AnnotationTab::onImportAnnotations);
    add("导入类别", &AnnotationTab::onImportLabels);
    add("自动划分", &AnnotationTab::onAutoSplit);
    add("导出 YOLO", &AnnotationTab::onExportYolo, true);
    add("COCO", &AnnotationTab::onExportCoco);
    add("VOC", &AnnotationTab::onExportVoc);

    return bar;
}

QWidget* AnnotationTab::buildLeftTools()
{
    auto* panel = makeCard();
    panel->setFixedWidth(86);
    auto* v = new QVBoxLayout(panel);
    v->setContentsMargins(10, 12, 10, 12);
    v->setSpacing(7);

    auto addTool = [&](const QString& icon, const QString& label, AnnotationTool tool) {
        auto* b = new QPushButton(QStringLiteral("%1\n%2").arg(icon, label));
        b->setCheckable(true);
        b->setProperty("toolButton", true);
        b->setProperty("tool", static_cast<int>(tool));
        b->setToolTip(label);
        b->setFixedSize(64, 52);
        connect(b, &QPushButton::clicked, this, &AnnotationTab::onToolClicked);
        m_toolButtons.append(b);
        v->addWidget(b);
    };
    addTool("↖", "选择", AnnotationTool::Select);
    addTool("✋", "移动", AnnotationTool::Pan);
    addTool("□", "矩形", AnnotationTool::Rectangle);
    addTool("◇", "旋转", AnnotationTool::RotatedRect);
    addTool("⬟", "多边形", AnnotationTool::Polygon);
    addTool("╱", "线段", AnnotationTool::Line);
    addTool("●", "点", AnnotationTool::Point);
    addTool("☷", "姿态", AnnotationTool::Pose);
    v->addStretch(1);
    m_btnUndo = new QPushButton(QStringLiteral("↶\n撤销"));
    m_btnUndo->setProperty("toolButton", true);
    m_btnUndo->setToolTip(QStringLiteral("撤销"));
    m_btnUndo->setFixedSize(64, 52);
    connect(m_btnUndo, &QPushButton::clicked, this, &AnnotationTab::onUndo);
    v->addWidget(m_btnUndo);
    return panel;
}

QWidget* AnnotationTab::buildRightPanel()
{
    auto* panel = new QWidget();
    panel->setFixedWidth(330);
    auto* v = new QVBoxLayout(panel);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(12);

    auto* classCard = makeCard();
    auto* cv = new QVBoxLayout(classCard);
    cv->setContentsMargins(12, 12, 12, 12);
    cv->addWidget(makeCardTitle("标签"));
    m_classList = new QListWidget();
    cv->addWidget(m_classList, 1);
    auto* classActions = new QHBoxLayout();
    auto* addClass = smallButton("+ 新增标签");
    connect(addClass, &QPushButton::clicked, this, [this] {
        bool ok = false;
        QString name = QInputDialog::getText(this, QStringLiteral("新增标签"), QStringLiteral("标签名"), QLineEdit::Normal, {}, &ok);
        if (ok && !name.trimmed().isEmpty()) {
            m_project.addClass(name);
            refreshClassList();
        }
    });
    auto* deleteClass = smallButton("删除标签");
    connect(deleteClass, &QPushButton::clicked, this, [this] {
        const int row = m_classList->currentRow();
        if (row < 0 || row >= m_project.classes.size()) {
            QMessageBox::information(this, QStringLiteral("删除标签"), QStringLiteral("请先在标签列表中选择要删除的标签。"));
            return;
        }
        const AnnotationClass cls = m_project.classes[row];
        for (const auto& image : m_project.images) {
            for (const auto& ann : image.annotations) {
                if (ann.classId == cls.id) {
                    QMessageBox::warning(this, QStringLiteral("删除标签"),
                        QStringLiteral("标签“%1”已被对象使用，请先编辑或删除相关标注对象。").arg(cls.name));
                    return;
                }
            }
        }
        auto ret = QMessageBox::question(this, QStringLiteral("删除标签"),
            QStringLiteral("确认删除标签“%1”？").arg(cls.name));
        if (ret != QMessageBox::Yes)
            return;
        m_project.classes.removeAt(row);
        m_project.dirty = true;
        refreshClassList();
        if (m_canvas)
            m_canvas->update();
    });
    classActions->addWidget(addClass);
    classActions->addWidget(deleteClass);
    cv->addLayout(classActions);
    v->addWidget(classCard, 1);

    auto* objectCard = makeCard();
    auto* ov = new QVBoxLayout(objectCard);
    ov->setContentsMargins(12, 12, 12, 12);
    ov->addWidget(makeCardTitle("对象"));
    m_objectList = new QListWidget();
    connect(m_objectList, &QListWidget::currentRowChanged, this, [this](int row) {
        m_selectedObject = row;
        if (m_canvas)
            m_canvas->setSelectedObject(row);
        refreshPropertyPanel();
    });
    ov->addWidget(m_objectList, 1);
    auto* objActions = new QHBoxLayout();
    auto* editObj = smallButton("编辑");
    auto* delObj = smallButton("删除");
    connect(editObj, &QPushButton::clicked, this, &AnnotationTab::onEditSelectedObject);
    connect(delObj, &QPushButton::clicked, this, &AnnotationTab::onDeleteSelectedObject);
    objActions->addWidget(editObj);
    objActions->addWidget(delObj);
    ov->addLayout(objActions);
    v->addWidget(objectCard, 1);

    auto* propCard = makeCard();
    auto* pv = new QVBoxLayout(propCard);
    pv->setContentsMargins(12, 12, 12, 12);
    pv->addWidget(makeCardTitle("属性与描述"));
    m_propertyText = new QTextEdit();
    m_propertyText->setReadOnly(true);
    pv->addWidget(m_propertyText, 1);
    v->addWidget(propCard, 1);

    return panel;
}

QWidget* AnnotationTab::buildBottomPanel()
{
    auto* panel = makeCard();
    panel->setFixedHeight(150);
    auto* h = new QHBoxLayout(panel);
    h->setContentsMargins(12, 12, 12, 12);
    h->setSpacing(12);

    auto* splitBox = new QWidget();
    splitBox->setFixedWidth(360);
    auto* sv = new QVBoxLayout(splitBox);
    sv->setContentsMargins(0, 0, 0, 0);
    sv->setSpacing(6);
    sv->addWidget(makeCardTitle("数据集划分"));
    m_splitInfo = new QLabel();
    m_splitInfo->setObjectName("fieldLabel");
    sv->addWidget(m_splitInfo);
    auto* ratioRow = new QHBoxLayout();
    ratioRow->setSpacing(6);
    ratioRow->addWidget(new QLabel("train"));
    m_spSplitTrain = makePercentSpin(80);
    ratioRow->addWidget(m_spSplitTrain);
    ratioRow->addWidget(new QLabel("val"));
    m_spSplitVal = makePercentSpin(10);
    ratioRow->addWidget(m_spSplitVal);
    ratioRow->addWidget(new QLabel("test"));
    m_spSplitTest = makePercentSpin(10);
    ratioRow->addWidget(m_spSplitTest);
    ratioRow->addStretch(1);
    sv->addLayout(ratioRow);
    m_splitRatioHint = new QLabel();
    m_splitRatioHint->setObjectName("fieldLabel");
    sv->addWidget(m_splitRatioHint);
    connect(m_spSplitTrain, QOverload<int>::of(&QSpinBox::valueChanged), this, &AnnotationTab::onSplitRatioChanged);
    connect(m_spSplitVal, QOverload<int>::of(&QSpinBox::valueChanged), this, &AnnotationTab::onSplitRatioChanged);
    connect(m_spSplitTest, QOverload<int>::of(&QSpinBox::valueChanged), this, &AnnotationTab::onSplitRatioChanged);
    auto* splitButtons = new QHBoxLayout();
    auto* trainBtn = smallButton("train");
    auto* valBtn = smallButton("val");
    auto* testBtn = smallButton("test");
    connect(trainBtn, &QPushButton::clicked, this, &AnnotationTab::onSetCurrentSplitTrain);
    connect(valBtn, &QPushButton::clicked, this, &AnnotationTab::onSetCurrentSplitVal);
    connect(testBtn, &QPushButton::clicked, this, &AnnotationTab::onSetCurrentSplitTest);
    splitButtons->addWidget(trainBtn);
    splitButtons->addWidget(valBtn);
    splitButtons->addWidget(testBtn);
    sv->addLayout(splitButtons);
    h->addWidget(splitBox);

    m_imageList = new QListWidget();
    m_imageList->setViewMode(QListView::ListMode);
    connect(m_imageList, &QListWidget::currentRowChanged, this, &AnnotationTab::onImageChanged);
    h->addWidget(m_imageList, 1);
    return panel;
}

void AnnotationTab::onNewProject()
{
    m_project.clear();
    m_currentImage = -1;
    m_selectedObject = -1;
    m_undoStack.clear();
    m_cbTask->setCurrentIndex(0);
    refreshAll();
}

void AnnotationTab::onOpenProject()
{
    QString path = QFileDialog::getOpenFileName(this, QStringLiteral("打开 TRTForge 标注项目"),
        AppSettings::annotationProjectDir(), QStringLiteral("TRTForge Annotation (*.tfanno)"));
    if (path.isEmpty())
        return;
    AppSettings::setAnnotationProjectDir(path);
    QString error;
    if (!m_project.load(path, &error)) {
        QMessageBox::warning(this, QStringLiteral("打开失败"), error);
        return;
    }
    m_cbTask->setCurrentIndex(static_cast<int>(m_project.task));
    m_currentImage = m_project.images.isEmpty() ? -1 : 0;
    m_selectedObject = -1;
    m_undoStack.clear();
    refreshAll();
}

void AnnotationTab::onSaveProject()
{
    if (m_project.projectPath.isEmpty()) {
        onSaveProjectAs();
        return;
    }
    QString error;
    if (!m_project.save(m_project.projectPath, &error))
        QMessageBox::warning(this, QStringLiteral("保存失败"), error);
    else
        QMessageBox::information(this, QStringLiteral("保存完成"), QStringLiteral("项目已保存：%1").arg(m_project.projectPath));
}

void AnnotationTab::onSaveProjectAs()
{
    QString path = QFileDialog::getSaveFileName(this, QStringLiteral("保存 TRTForge 标注项目"),
        AppSettings::annotationProjectDir(), QStringLiteral("TRTForge Annotation (*.tfanno)"));
    if (path.isEmpty())
        return;
    if (!path.endsWith(".tfanno", Qt::CaseInsensitive))
        path += ".tfanno";
    AppSettings::setAnnotationProjectDir(path);
    m_project.projectPath = path;
    onSaveProject();
}

void AnnotationTab::onImportImages()
{
    QString dirPath = QFileDialog::getExistingDirectory(this, QStringLiteral("选择图片目录"),
        AppSettings::annotationImageImportDir());
    if (dirPath.isEmpty())
        return;
    AppSettings::setAnnotationImageImportDir(dirPath);
    QDir dir(dirPath);
    const auto files = dir.entryInfoList({"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"}, QDir::Files, QDir::Name);
    for (const QFileInfo& info : files) {
        AnnotationImage image;
        image.id = info.completeBaseName();
        image.path = info.absoluteFilePath();
        QImageReader reader(image.path);
        QSize size = reader.size();
        image.width = size.width();
        image.height = size.height();
        image.split = DatasetSplit::Train;
        m_project.images.append(image);
    }
    if (m_currentImage < 0 && !m_project.images.isEmpty())
        m_currentImage = 0;
    m_project.dirty = true;
    refreshAll();
}

void AnnotationTab::onImportLabels()
{
    const QString labelDir = AppSettings::annotationLabelImportDir().isEmpty()
        ? qApp->applicationDirPath()
        : AppSettings::annotationLabelImportDir();
    QString path = QFileDialog::getOpenFileName(this, QStringLiteral("导入 labels_*.json"), labelDir, QStringLiteral("JSON (*.json)"));
    if (path.isEmpty())
        return;
    AppSettings::setAnnotationLabelImportDir(path);
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("导入失败"), QStringLiteral("无法读取：%1").arg(path));
        return;
    }
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();
    QJsonArray classes = root.value("classes").toArray();
    QJsonArray colors = root.value("colors").toArray();
    m_project.classes.clear();
    for (int i = 0; i < classes.size(); ++i) {
        QColor color = QColor::fromHsv((i * 47) % 360, 210, 240);
        QJsonArray c = (i < colors.size()) ? colors.at(i).toArray() : QJsonArray();
        if (c.size() >= 3)
            color = QColor(c[0].toInt(), c[1].toInt(), c[2].toInt());
        m_project.classes.append({i, classes[i].toString(QStringLiteral("class_%1").arg(i)), color});
    }
    m_project.ensureDefaultClass();
    m_project.dirty = true;
    refreshClassList();
}

void AnnotationTab::onImportAnnotations()
{
    QString path = QFileDialog::getOpenFileName(this, QStringLiteral("导入 COCO/VOC/YOLO 标注"),
        AppSettings::annotationDatasetImportDir(), QStringLiteral("Dataset (*.yaml *.json *.xml);;All (*.*)"));
    if (path.isEmpty())
        return;
    AppSettings::setAnnotationDatasetImportDir(path);

    DatasetIoResult result;
    QFileInfo info(path);
    if (info.suffix().compare("json", Qt::CaseInsensitive) == 0) {
        result = DatasetExporter::importCoco(m_project, path, static_cast<AnnotationTask>(m_cbTask->currentData().toInt()));
    } else if (info.suffix().compare("xml", Qt::CaseInsensitive) == 0) {
        result = DatasetExporter::importVoc(m_project, info.absolutePath());
    } else {
        result = DatasetExporter::importYolo(m_project, path, static_cast<AnnotationTask>(m_cbTask->currentData().toInt()));
    }
    m_currentImage = m_project.images.isEmpty() ? -1 : 0;
    refreshAll();
    showIoResult(QStringLiteral("导入标注"), result.ok, result.message);
}

void AnnotationTab::onExportYolo()
{
    QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("选择 YOLO 导出目录"),
        AppSettings::annotationExportDir());
    if (dir.isEmpty())
        return;
    AppSettings::setAnnotationExportDir(dir);
    auto result = DatasetExporter::exportYolo(m_project, dir);
    showIoResult(QStringLiteral("导出 YOLO"), result.ok, result.message);
}

void AnnotationTab::onExportCoco()
{
    QString path = QFileDialog::getSaveFileName(this, QStringLiteral("导出 COCO JSON"),
        AppSettings::annotationExportDir(), QStringLiteral("COCO JSON (*.json)"));
    if (path.isEmpty())
        return;
    AppSettings::setAnnotationExportDir(path);
    auto result = DatasetExporter::exportCoco(m_project, path);
    showIoResult(QStringLiteral("导出 COCO"), result.ok, result.message);
}

void AnnotationTab::onExportVoc()
{
    QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("选择 VOC 导出目录"),
        AppSettings::annotationExportDir());
    if (dir.isEmpty())
        return;
    AppSettings::setAnnotationExportDir(dir);
    auto result = DatasetExporter::exportVoc(m_project, dir);
    showIoResult(QStringLiteral("导出 VOC"), result.ok, result.message);
}

void AnnotationTab::loadSplitRatioSettings()
{
    if (!m_spSplitTrain || !m_spSplitVal || !m_spSplitTest)
        return;

    const SplitRatio ratio = AppSettings::annotationSplitRatio();

    const QSignalBlocker trainBlocker(m_spSplitTrain);
    const QSignalBlocker valBlocker(m_spSplitVal);
    const QSignalBlocker testBlocker(m_spSplitTest);
    m_spSplitTrain->setValue(ratio.train);
    m_spSplitVal->setValue(ratio.val);
    m_spSplitTest->setValue(ratio.test);
    onSplitRatioChanged();
}

void AnnotationTab::saveSplitRatioSettings() const
{
    AppSettings::setAnnotationSplitRatio({ splitTrainPct(), splitValPct(), splitTestPct() });
}

int AnnotationTab::splitTrainPct() const
{
    return m_spSplitTrain ? m_spSplitTrain->value() : 80;
}

int AnnotationTab::splitValPct() const
{
    return m_spSplitVal ? m_spSplitVal->value() : 10;
}

int AnnotationTab::splitTestPct() const
{
    return m_spSplitTest ? m_spSplitTest->value() : 10;
}

QString AnnotationTab::splitRatioText() const
{
    return QStringLiteral("%1/%2/%3").arg(splitTrainPct()).arg(splitValPct()).arg(splitTestPct());
}

void AnnotationTab::onSplitRatioChanged()
{
    const int total = splitTrainPct() + splitValPct() + splitTestPct();
    if (m_splitRatioHint) {
        m_splitRatioHint->setText(total == 100
            ? QStringLiteral("自动比例 %1，已保存").arg(splitRatioText())
            : QStringLiteral("自动比例 %1，合计 %2%，请调整为 100%").arg(splitRatioText()).arg(total));
    }
    if (total == 100)
        saveSplitRatioSettings();
    if (m_splitInfo)
        refreshImageList();
}

void AnnotationTab::onAutoSplit()
{
    if (m_project.images.isEmpty())
        return;
    const int trainPct = splitTrainPct();
    const int valPct = splitValPct();
    const int testPct = splitTestPct();
    const int totalPct = trainPct + valPct + testPct;
    if (totalPct != 100) {
        QMessageBox::warning(this, QStringLiteral("自动划分比例无效"),
            QStringLiteral("train/val/test 百分比合计必须为 100%，当前为 %1%。").arg(totalPct));
        return;
    }
    DatasetSplitPolicy::applyToProject(m_project, { trainPct, valPct, testPct });
    refreshAll();
}

void AnnotationTab::onEditSelectedObject()
{
    AnnotationImage* image = currentImage();
    int row = m_objectList->currentRow();
    if (!image || row < 0 || row >= image->annotations.size())
        return;
    AnnotationConfirmDialog dialog(m_project, image->annotations[row], image->split, this);
    if (dialog.exec() != QDialog::Accepted)
        return;
    pushUndoForCurrentImage();
    image->annotations[row] = dialog.annotation();
    image->split = dialog.split();
    m_project.dirty = true;
    refreshAll();
    m_objectList->setCurrentRow(row);
}

void AnnotationTab::onDeleteSelectedObject()
{
    AnnotationImage* image = currentImage();
    int row = m_objectList->currentRow();
    if (!image || row < 0 || row >= image->annotations.size())
        return;
    auto ret = QMessageBox::question(this, QStringLiteral("删除对象"), QStringLiteral("确认删除当前标注对象？"));
    if (ret != QMessageBox::Yes)
        return;
    pushUndoForCurrentImage();
    image->annotations.removeAt(row);
    m_selectedObject = -1;
    m_project.dirty = true;
    refreshAll();
}

void AnnotationTab::onSetCurrentSplitTrain()
{
    if (auto* image = currentImage()) {
        image->split = DatasetSplit::Train;
        m_project.dirty = true;
        refreshAll();
    }
}

void AnnotationTab::onSetCurrentSplitVal()
{
    if (auto* image = currentImage()) {
        image->split = DatasetSplit::Val;
        m_project.dirty = true;
        refreshAll();
    }
}

void AnnotationTab::onSetCurrentSplitTest()
{
    if (auto* image = currentImage()) {
        image->split = DatasetSplit::Test;
        m_project.dirty = true;
        refreshAll();
    }
}

void AnnotationTab::onUndo()
{
    if (m_undoStack.isEmpty())
        return;
    UndoState state = m_undoStack.takeLast();
    if (state.imageIndex < 0 || state.imageIndex >= m_project.images.size())
        return;
    m_currentImage = state.imageIndex;
    m_project.images[state.imageIndex].annotations = state.annotations;
    m_selectedObject = -1;
    m_project.dirty = true;
    refreshAll();
}

void AnnotationTab::onCanvasObjectSelected(int row)
{
    m_selectedObject = row;
    m_objectList->blockSignals(true);
    m_objectList->setCurrentRow(row);
    m_objectList->blockSignals(false);
    refreshPropertyPanel();
}

void AnnotationTab::onCanvasAnnotationsEdited()
{
    m_project.dirty = true;
    refreshPropertyPanel();
}

void AnnotationTab::onCanvasEditStarted()
{
    pushUndoForCurrentImage();
}

void AnnotationTab::onImageChanged(int row)
{
    m_currentImage = row;
    m_selectedObject = -1;
    loadCurrentImage();
    refreshObjectList();
    refreshPropertyPanel();
}

void AnnotationTab::onDraftCompleted(AnnotationObject draft)
{
    AnnotationImage* image = currentImage();
    if (!image)
        return;
    if (m_project.classes.isEmpty()) {
        bool ok = false;
        QString label = QInputDialog::getText(this, QStringLiteral("新增标签"),
            QStringLiteral("当前项目没有标签，请先输入第一个标签名称"), QLineEdit::Normal, {}, &ok);
        if (!ok || label.trimmed().isEmpty()) {
            m_canvas->update();
            return;
        }
        m_project.addClass(label.trimmed());
        refreshClassList();
    }
    draft.id = QStringLiteral("ann_%1").arg(image->annotations.size() + 1);
    draft.classId = m_project.currentClassId();
    draft.label = m_project.classById(draft.classId).name;
    AnnotationConfirmDialog dialog(m_project, draft, image->split, this);
    if (dialog.exec() != QDialog::Accepted) {
        m_canvas->update();
        return;
    }
    AnnotationObject ann = dialog.annotation();
    pushUndoForCurrentImage();
    image->split = dialog.split();
    image->annotations.append(ann);
    m_selectedObject = image->annotations.size() - 1;
    m_project.dirty = true;
    refreshAll();
    m_objectList->setCurrentRow(m_selectedObject);
}

void AnnotationTab::onToolClicked()
{
    auto* button = qobject_cast<QPushButton*>(sender());
    if (!button)
        return;
    setToolButton(button, static_cast<AnnotationTool>(button->property("tool").toInt()));
}

void AnnotationTab::refreshAll()
{
    m_cbTask->blockSignals(true);
    m_cbTask->setCurrentIndex(static_cast<int>(m_project.task));
    m_cbTask->blockSignals(false);
    QString name = m_project.projectPath.isEmpty() ? QStringLiteral("未保存项目") : QFileInfo(m_project.projectPath).fileName();
    m_projectInfo->setText(QStringLiteral("%1 · %2 · %3 张图片")
        .arg(name, AnnotationProject::taskToString(m_project.task))
        .arg(m_project.images.size()));
    refreshClassList();
    refreshImageList();
    loadCurrentImage();
    refreshObjectList();
    refreshPropertyPanel();
}

void AnnotationTab::refreshClassList()
{
    m_classList->clear();
    for (const auto& cls : m_project.classes) {
        auto* item = new QListWidgetItem(QStringLiteral("%1  %2").arg(cls.id, 2, 10, QChar('0')).arg(cls.name));
        item->setForeground(cls.color);
        m_classList->addItem(item);
    }
    if (m_canvas)
        m_canvas->setClasses(&m_project.classes);
}

void AnnotationTab::refreshImageList()
{
    m_imageList->blockSignals(true);
    m_imageList->clear();
    int train = 0, val = 0, test = 0;
    for (const auto& image : m_project.images) {
        if (image.split == DatasetSplit::Train) ++train;
        else if (image.split == DatasetSplit::Val) ++val;
        else ++test;
        m_imageList->addItem(QStringLiteral("%1  [%2]  %3 objects")
            .arg(QFileInfo(image.path).fileName(), AnnotationProject::splitToString(image.split))
            .arg(image.annotations.size()));
    }
    m_splitInfo->setText(QStringLiteral("train: %1\nval: %2\ntest: %3\n自动比例 %4，支持手动调整")
        .arg(train)
        .arg(val)
        .arg(test)
        .arg(splitRatioText()));
    if (m_currentImage >= 0 && m_currentImage < m_imageList->count())
        m_imageList->setCurrentRow(m_currentImage);
    m_imageList->blockSignals(false);
}

void AnnotationTab::refreshObjectList()
{
    m_objectList->blockSignals(true);
    m_objectList->clear();
    const AnnotationImage* image = currentImage();
    if (!image) {
        m_objectList->blockSignals(false);
        return;
    }
    for (const auto& ann : image->annotations) {
        m_objectList->addItem(QStringLiteral("%1 · %2 · %3")
            .arg(ann.id, shapeLabel(ann.shape), m_project.classById(ann.classId).name));
    }
    if (m_selectedObject >= 0 && m_selectedObject < m_objectList->count())
        m_objectList->setCurrentRow(m_selectedObject);
    else
        m_selectedObject = -1;
    m_objectList->blockSignals(false);
    if (m_canvas)
        m_canvas->setSelectedObject(m_selectedObject);
}

void AnnotationTab::refreshPropertyPanel()
{
    const AnnotationImage* image = currentImage();
    if (!image) {
        m_propertyText->setPlainText(QStringLiteral("尚未选择图片"));
        return;
    }
    int row = m_objectList->currentRow();
    if (row < 0 || row >= image->annotations.size()) {
        m_propertyText->setPlainText(QStringLiteral("图片: %1\n尺寸: %2 x %3\n集合: %4\n对象数: %5")
            .arg(QFileInfo(image->path).fileName())
            .arg(image->width)
            .arg(image->height)
            .arg(AnnotationProject::splitToString(image->split))
            .arg(image->annotations.size()));
        return;
    }
    const auto& ann = image->annotations[row];
    m_propertyText->setPlainText(QStringLiteral("标签: %1\n描述: %2\n对象: %3\n分组: %4\n集合: %5\n格式: %6\n点数: %7\n状态: 已确认")
        .arg(m_project.classById(ann.classId).name,
             ann.description.isEmpty() ? QStringLiteral("-") : ann.description,
             ann.id)
        .arg(ann.groupId)
        .arg(AnnotationProject::splitToString(image->split),
             AnnotationProject::taskToString(m_project.task))
        .arg(ann.points.size()));
}

void AnnotationTab::loadCurrentImage()
{
    AnnotationImage* image = currentImage();
    if (!image) {
        m_canvas->setAnnotations(nullptr);
        m_canvas->setImage(QString());
        return;
    }
    m_canvas->setImage(image->path);
    m_canvas->setAnnotations(&image->annotations);
    m_canvas->setSelectedObject(m_selectedObject);
}

void AnnotationTab::setToolButton(QPushButton* button, AnnotationTool tool)
{
    if (!button)
        return;
    for (auto* b : m_toolButtons)
        b->setChecked(b == button);
    if (m_canvas)
        m_canvas->setTool(tool);
}

AnnotationImage* AnnotationTab::currentImage()
{
    return m_project.imageAt(m_currentImage);
}

const AnnotationImage* AnnotationTab::currentImage() const
{
    return m_project.imageAt(m_currentImage);
}

void AnnotationTab::showIoResult(const QString& title, bool ok, const QString& message)
{
    if (ok)
        QMessageBox::information(this, title, message);
    else
        QMessageBox::warning(this, title, message);
}

void AnnotationTab::pushUndoForCurrentImage()
{
    const AnnotationImage* image = currentImage();
    if (!image)
        return;
    UndoState state;
    state.imageIndex = m_currentImage;
    state.annotations = image->annotations;
    m_undoStack.append(state);
    if (m_undoStack.size() > 50)
        m_undoStack.removeFirst();
}

} // namespace trtforge

