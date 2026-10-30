#pragma once

#include "AnnotationProject.h"

#include <QString>

namespace trtforge {

struct DatasetIoResult {
    bool ok = false;
    QString message;
};

class DatasetExporter
{
public:
    static DatasetIoResult exportYolo(const AnnotationProject& project, const QString& outputDir);
    static DatasetIoResult exportCoco(const AnnotationProject& project, const QString& outputFile);
    static DatasetIoResult exportVoc(const AnnotationProject& project, const QString& outputDir);

    static DatasetIoResult importYolo(AnnotationProject& project, const QString& datasetYamlOrRoot, AnnotationTask task);
    static DatasetIoResult importCoco(AnnotationProject& project, const QString& annotationFile, AnnotationTask task);
    static DatasetIoResult importVoc(AnnotationProject& project, const QString& xmlDir);
};

} // namespace trtforge

