#include "common/AppSettings.h"

#include <QFileInfo>
#include <QSettings>
#include <QVariant>
#include <QtGlobal>

namespace trtforge {

namespace {

const QString kAnnotationDatasetGroup = QStringLiteral("AnnotationDataset");
const QString kSplitTrainPct = QStringLiteral("splitTrainPct");
const QString kSplitValPct = QStringLiteral("splitValPct");
const QString kSplitTestPct = QStringLiteral("splitTestPct");
const QString kProjectDir = QStringLiteral("projectDir");
const QString kImageImportDir = QStringLiteral("imageImportDir");
const QString kLabelImportDir = QStringLiteral("labelImportDir");
const QString kDatasetImportDir = QStringLiteral("datasetImportDir");
const QString kExportDir = QStringLiteral("exportDir");

SplitRatio normalizedSplitRatio(SplitRatio ratio)
{
    ratio.train = qBound(0, ratio.train, 100);
    ratio.val = qBound(0, ratio.val, 100);
    ratio.test = qBound(0, ratio.test, 100);
    return ratio.isValid() ? ratio : SplitRatio::defaults();
}

QString directoryFromPath(const QString& path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty()) {
        return {};
    }
    const QFileInfo info(trimmed);
    if (info.isDir()) {
        return info.absoluteFilePath();
    }
    return info.absolutePath();
}

QString annotationPathValue(const QString& key)
{
    QSettings settings;
    settings.beginGroup(kAnnotationDatasetGroup);
    const QString path = settings.value(key).toString();
    settings.endGroup();
    return path;
}

void setAnnotationPathValue(const QString& key, const QString& path)
{
    const QString directory = directoryFromPath(path);
    if (directory.isEmpty()) {
        return;
    }

    QSettings settings;
    settings.beginGroup(kAnnotationDatasetGroup);
    settings.setValue(key, QVariant(directory));
    settings.endGroup();
}

} // namespace

int SplitRatio::total() const
{
    return train + val + test;
}

bool SplitRatio::isValid() const
{
    return train >= 0 && val >= 0 && test >= 0
        && train <= 100 && val <= 100 && test <= 100
        && total() == 100;
}

SplitRatio SplitRatio::defaults()
{
    return {};
}

SplitRatio AppSettings::annotationSplitRatio()
{
    QSettings settings;
    settings.beginGroup(kAnnotationDatasetGroup);
    SplitRatio ratio;
    ratio.train = settings.value(kSplitTrainPct, ratio.train).toInt();
    ratio.val = settings.value(kSplitValPct, ratio.val).toInt();
    ratio.test = settings.value(kSplitTestPct, ratio.test).toInt();
    settings.endGroup();
    return normalizedSplitRatio(ratio);
}

void AppSettings::setAnnotationSplitRatio(const SplitRatio& ratio)
{
    if (!ratio.isValid()) {
        return;
    }

    QSettings settings;
    settings.beginGroup(kAnnotationDatasetGroup);
    settings.setValue(kSplitTrainPct, ratio.train);
    settings.setValue(kSplitValPct, ratio.val);
    settings.setValue(kSplitTestPct, ratio.test);
    settings.endGroup();
}

QString AppSettings::annotationProjectDir()
{
    return annotationPathValue(kProjectDir);
}

void AppSettings::setAnnotationProjectDir(const QString& path)
{
    setAnnotationPathValue(kProjectDir, path);
}

QString AppSettings::annotationImageImportDir()
{
    return annotationPathValue(kImageImportDir);
}

void AppSettings::setAnnotationImageImportDir(const QString& path)
{
    setAnnotationPathValue(kImageImportDir, path);
}

QString AppSettings::annotationLabelImportDir()
{
    return annotationPathValue(kLabelImportDir);
}

void AppSettings::setAnnotationLabelImportDir(const QString& path)
{
    setAnnotationPathValue(kLabelImportDir, path);
}

QString AppSettings::annotationDatasetImportDir()
{
    return annotationPathValue(kDatasetImportDir);
}

void AppSettings::setAnnotationDatasetImportDir(const QString& path)
{
    setAnnotationPathValue(kDatasetImportDir, path);
}

QString AppSettings::annotationExportDir()
{
    return annotationPathValue(kExportDir);
}

void AppSettings::setAnnotationExportDir(const QString& path)
{
    setAnnotationPathValue(kExportDir, path);
}

} // namespace trtforge
