#pragma once

#include <QString>

namespace trtforge {

struct SplitRatio
{
    int train { 80 };
    int val { 10 };
    int test { 10 };

    int total() const;
    bool isValid() const;
    static SplitRatio defaults();
};

class AppSettings
{
public:
    static SplitRatio annotationSplitRatio();
    static void setAnnotationSplitRatio(const SplitRatio& ratio);

    static QString annotationProjectDir();
    static void setAnnotationProjectDir(const QString& path);
    static QString annotationImageImportDir();
    static void setAnnotationImageImportDir(const QString& path);
    static QString annotationLabelImportDir();
    static void setAnnotationLabelImportDir(const QString& path);
    static QString annotationDatasetImportDir();
    static void setAnnotationDatasetImportDir(const QString& path);
    static QString annotationExportDir();
    static void setAnnotationExportDir(const QString& path);
};

} // namespace trtforge
