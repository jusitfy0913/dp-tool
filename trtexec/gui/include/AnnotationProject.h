#pragma once

#include <QColor>
#include <QJsonObject>
#include <QPointF>
#include <QString>
#include <QStringList>
#include <QVector>

namespace trtforge {

enum class AnnotationTask {
    Detect,
    Segment,
    Pose,
    Obb
};

enum class AnnotationShape {
    Rectangle,
    RotatedRect,
    Polygon,
    Line,
    Point,
    Keypoints
};

enum class DatasetSplit {
    Train,
    Val,
    Test
};

struct AnnotationClass {
    int id = 0;
    QString name;
    QColor color = QColor(31, 182, 255);
};

struct AnnotationObject {
    QString id;
    int classId = 0;
    QString label;
    QString description;
    int groupId = 0;
    AnnotationShape shape = AnnotationShape::Rectangle;
    QVector<QPointF> points;
    QVector<int> keypointVisibility;
    double rotation = 0.0;
    double score = -1.0;
};

struct AnnotationImage {
    QString id;
    QString path;
    int width = 0;
    int height = 0;
    DatasetSplit split = DatasetSplit::Train;
    QVector<AnnotationObject> annotations;
};

class AnnotationProject
{
public:
    static constexpr int CurrentVersion = 1;

    int version = CurrentVersion;
    AnnotationTask task = AnnotationTask::Detect;
    QVector<AnnotationClass> classes;
    QVector<AnnotationImage> images;
    QString projectPath;
    bool dirty = false;

    void clear();
    void ensureDefaultClass();
    QStringList classNames() const;
    AnnotationClass classById(int id) const;
    int addClass(const QString& name, const QColor& color = QColor());
    AnnotationImage* imageAt(int index);
    const AnnotationImage* imageAt(int index) const;
    int currentClassId() const;

    bool save(const QString& path, QString* error = nullptr) const;
    bool load(const QString& path, QString* error = nullptr);

    static QString taskToString(AnnotationTask task);
    static AnnotationTask taskFromString(const QString& text);
    static QString shapeToString(AnnotationShape shape);
    static AnnotationShape shapeFromString(const QString& text);
    static QString splitToString(DatasetSplit split);
    static DatasetSplit splitFromString(const QString& text);

    static QJsonObject objectToJson(const AnnotationObject& obj);
    static AnnotationObject objectFromJson(const QJsonObject& obj);
};

} // namespace trtforge

