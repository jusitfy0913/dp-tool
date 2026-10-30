#include "AnnotationConfirmDialog.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

namespace trtforge {

AnnotationConfirmDialog::AnnotationConfirmDialog(const AnnotationProject& project,
                                                 const AnnotationObject& draft,
                                                 DatasetSplit currentSplit,
                                                 QWidget* parent)
    : QDialog(parent), m_annotation(draft)
{
    setWindowTitle(QStringLiteral("标注确认"));
    setModal(true);
    resize(440, 320);

    auto* root = new QVBoxLayout(this);
    root->setSpacing(12);

    auto* title = new QLabel(QStringLiteral("绘制完成，请确认标签、描述和数据集归属"));
    title->setObjectName("cardTitle");
    root->addWidget(title);

    auto* form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignRight);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(10);

    m_cbClass = new QComboBox();
    for (const auto& cls : project.classes)
        m_cbClass->addItem(QStringLiteral("%1 · %2").arg(cls.id).arg(cls.name), cls.id);
    int idx = m_cbClass->findData(draft.classId);
    if (idx >= 0)
        m_cbClass->setCurrentIndex(idx);
    form->addRow(QStringLiteral("标签"), m_cbClass);

    m_edDescription = new QPlainTextEdit();
    m_edDescription->setFixedHeight(72);
    m_edDescription->setPlaceholderText(QStringLiteral("请输入标注描述，可为空"));
    m_edDescription->setPlainText(draft.description);
    form->addRow(QStringLiteral("描述"), m_edDescription);

    m_spGroup = new QSpinBox();
    m_spGroup->setRange(0, 999999);
    m_spGroup->setValue(qMax(0, draft.groupId));
    form->addRow(QStringLiteral("Group"), m_spGroup);

    m_cbSplit = new QComboBox();
    m_cbSplit->addItem("train", static_cast<int>(DatasetSplit::Train));
    m_cbSplit->addItem("val", static_cast<int>(DatasetSplit::Val));
    m_cbSplit->addItem("test", static_cast<int>(DatasetSplit::Test));
    m_cbSplit->setCurrentIndex(static_cast<int>(currentSplit));
    form->addRow(QStringLiteral("集合"), m_cbSplit);

    m_spScore = new QDoubleSpinBox();
    m_spScore->setRange(-1.0, 1.0);
    m_spScore->setSingleStep(0.01);
    m_spScore->setDecimals(3);
    m_spScore->setSpecialValueText(QStringLiteral("无"));
    m_spScore->setValue(draft.score);
    form->addRow(QStringLiteral("置信度"), m_spScore);

    root->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok);
    buttons->button(QDialogButtonBox::Ok)->setText(QStringLiteral("确定并更新"));
    buttons->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("取消"));
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    root->addWidget(buttons);
}

AnnotationObject AnnotationConfirmDialog::annotation() const
{
    AnnotationObject out = m_annotation;
    out.classId = m_cbClass->currentData().toInt();
    out.label = m_cbClass->currentText().section(QStringLiteral(" · "), 1);
    out.description = m_edDescription->toPlainText().trimmed();
    out.groupId = m_spGroup->value();
    out.score = m_spScore->value();
    return out;
}

DatasetSplit AnnotationConfirmDialog::split() const
{
    return static_cast<DatasetSplit>(m_cbSplit->currentData().toInt());
}

} // namespace trtforge

