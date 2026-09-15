#pragma once

#include "AnnotationProject.h"
#include "DatasetExporter.h"

#include <QString>

namespace trtforge {

class DatasetYamlWriter
{
public:
    static DatasetIoResult writeYoloDatasetYaml(const AnnotationProject& project, const QString& outputDir);
};

} // namespace trtforge
