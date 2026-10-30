#include "dataset/DatasetYamlWriter.h"

#include <QDir>
#include <QFile>
#include <QStringConverter>
#include <QTextStream>

namespace trtforge {

DatasetIoResult DatasetYamlWriter::writeYoloDatasetYaml(const AnnotationProject& project, const QString& outputDir)
{
    QDir root(outputDir);
    QFile yaml(root.filePath("dataset.yaml"));
    if (!yaml.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return {false, QStringLiteral("无法写入 dataset.yaml")};
    }

    QTextStream ys(&yaml);
    ys.setEncoding(QStringConverter::Utf8);
    ys << "path: " << QDir::toNativeSeparators(root.absolutePath()).replace("\\", "/") << "\n";
    ys << "train: images/train\n";
    ys << "val: images/val\n";
    ys << "test: images/test\n";
    if (project.task == AnnotationTask::Pose) {
        ys << "kpt_shape: [17, 3]\n";
        ys << "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n";
    }
    ys << "names:\n";
    for (const auto& cls : project.classes) {
        ys << "  " << cls.id << ": " << cls.name << "\n";
    }

    return {true, QString()};
}

} // namespace trtforge
