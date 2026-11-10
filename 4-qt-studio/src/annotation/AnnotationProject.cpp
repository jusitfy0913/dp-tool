#include "AnnotationProject.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>

namespace trtforge {

namespace {

QJsonArray pointArray(const QVector<QPointF>& points)
{
    QJsonArray out;
    for (const QPointF& p : points) {
        QJsonArray pair;
        pair.append(p.x());
        pair.append(p.y());
        out.append(pair);
    }
    return out;
}

QVector<QPointF> pointsFromJson(const QJsonArray& arr)
{
    QVector<QPointF> points;
    points.reserve(arr.size());
    for (const auto& value : arr) {
        const auto pair = value.toArray();
        if (pair.size() >= 2)
            points.append(QPointF(pair[0].toDouble(), pair[1].toDouble()));
    }
    return points;
}

QJsonArray colorArray(const QColor& color)
{
    QJsonArray arr;
    arr.append(color.red());
    arr.append(color.green());
    arr.append(color.blue());
    return arr;
}

QColor colorFromJson(const QJsonArray& arr, int seed)
{
    if (arr.size() >= 3)
        return QColor(arr[0].toInt(), arr[1].toInt(), arr[2].toInt());
    return QColor::fromHsv((seed * 47) % 360, 210, 240);
}

} // namespace

void AnnotationProject::clear()
{
    version = CurrentVersion;
    task = AnnotationTask::Detect;
    classes.clear();
    images.clear();
    projectPath.clear();
    dirty = false;
}

void AnnotationProject::ensureDefaultClass()
{
    if (!classes.isEmpty())
        return;
    classes.append({0, QStringLiteral("object"), QColor(31, 182, 255)});
}

QStringList AnnotationProject::classNames() const
{
    QStringList names;
    for (const auto& cls : classes)
        names << cls.name;
    return names;
}

AnnotationClass AnnotationProject::classById(int id) const
{
    for (const auto& cls : classes) {
        if (cls.id == id)
            return cls;
    }
    if (!classes.isEmpty())
        return classes.first();
    return {0, QStringLiteral("object"), QColor(31, 182, 255)};
}

int AnnotationProject::addClass(const QString& name, const QColor& color)
{
    int nextId = 0;
    for (const auto& cls : classes)
        nextId = qMax(nextId, cls.id + 1);
    QColor resolved = color.isValid() ? color : QColor::fromHsv((nextId * 47) % 360, 210, 240);
    classes.append({nextId, name.trimmed().isEmpty() ? QStringLiteral("class_%1").arg(nextId) : name.trimmed(), resolved});
    dirty = true;
    return nextId;
}

AnnotationImage* AnnotationProject::imageAt(int index)
{
    if (index < 0 || index >= images.size())
        return nullptr;
    return &images[index];
}

const AnnotationImage* AnnotationProject::imageAt(int index) const
{
    if (index < 0 || index >= images.size())
        return nullptr;
    return &images[index];
}

int AnnotationProject::currentClassId() const
{
    return classes.isEmpty() ? -1 : classes.first().id;
}

bool AnnotationProject::save(const QString& path, QString* error) const
{
    QJsonObject root;
    root["version"] = version;
    root["task"] = taskToString(task);

    QJsonArray classArray;
    for (const auto& cls : classes) {
        QJsonObject obj;
        obj["id"] = cls.id;
        obj["name"] = cls.name;
        obj["color"] = colorArray(cls.color);
        classArray.append(obj);
    }
    root["classes"] = classArray;

    QJsonArray imageArray;
    for (const auto& image : images) {
        QJsonObject obj;
        obj["id"] = image.id;
        obj["path"] = image.path;
        obj["width"] = image.width;
        obj["height"] = image.height;
        obj["split"] = splitToString(image.split);

        QJsonArray annArray;
        for (const auto& ann : image.annotations)
            annArray.append(objectToJson(ann));
        obj["annotations"] = annArray;
        imageArray.append(obj);
    }
    root["images"] = imageArray;

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (error) *error = QStringLiteral("无法写入项目文件：%1").arg(path);
        return false;
    }
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    return true;
}

bool AnnotationProject::load(const QString& path, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) *error = QStringLiteral("无法读取项目文件：%1").arg(path);
        return false;
    }

    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        if (error) *error = QStringLiteral("项目文件 JSON 无效：%1").arg(parseError.errorString());
        return false;
    }

    QJsonObject root = doc.object();
    const QString taskText = root.value("task").toString().trimmed().toLower();
    if (taskText == QLatin1String("classify") || taskText == QLatin1String("classification")) {
        if (error) {
            *error = QStringLiteral(
                "这是分类工程（整张图一个类别）。桌面版这一版不能打开，否则保存时会把任务改成检测。请用 Web 标注页打开。");
        }
        return false;
    }
    version = root.value("version").toInt(CurrentVersion);
    task = taskFromString(root.value("task").toString());
    projectPath = path;
    classes.clear();
    images.clear();

    for (const auto& value : root.value("classes").toArray()) {
        QJsonObject obj = value.toObject();
        int id = obj.value("id").toInt(classes.size());
        classes.append({
            id,
            obj.value("name").toString(QStringLiteral("class_%1").arg(id)),
            colorFromJson(obj.value("color").toArray(), id)
        });
    }
    ensureDefaultClass();

    for (const auto& value : root.value("images").toArray()) {
        QJsonObject obj = value.toObject();
        AnnotationImage image;
        image.id = obj.value("id").toString(QFileInfo(obj.value("path").toString()).completeBaseName());
        image.path = obj.value("path").toString();
        image.width = obj.value("width").toInt();
        image.height = obj.value("height").toInt();
        image.split = splitFromString(obj.value("split").toString());
        for (const auto& annValue : obj.value("annotations").toArray())
            image.annotations.append(objectFromJson(annValue.toObject()));
        images.append(image);
    }

    dirty = false;
    return true;
}

QString AnnotationProject::taskToString(AnnotationTask task)
{
    switch (task) {
    case AnnotationTask::Detect: return QStringLiteral("detect");
    case AnnotationTask::Segment: return QStringLiteral("segment");
    case AnnotationTask::Pose: return QStringLiteral("pose");
    case AnnotationTask::Obb: return QStringLiteral("obb");
    }
    return QStringLiteral("detect");
}

AnnotationTask AnnotationProject::taskFromString(const QString& text)
{
    const QString t = text.trimmed().toLower();
    if (t == "segment") return AnnotationTask::Segment;
    if (t == "pose") return AnnotationTask::Pose;
    if (t == "obb") return AnnotationTask::Obb;
    return AnnotationTask::Detect;
}

QString AnnotationProject::shapeToString(AnnotationShape shape)
{
    switch (shape) {
    case AnnotationShape::Rectangle: return QStringLiteral("rectangle");
    case AnnotationShape::RotatedRect: return QStringLiteral("rotated_rect");
    case AnnotationShape::Polygon: return QStringLiteral("polygon");
    case AnnotationShape::Line: return QStringLiteral("line");
    case AnnotationShape::Point: return QStringLiteral("point");
    case AnnotationShape::Keypoints: return QStringLiteral("keypoints");
    }
    return QStringLiteral("rectangle");
}

AnnotationShape AnnotationProject::shapeFromString(const QString& text)
{
    const QString t = text.trimmed().toLower();
    if (t == "rotated_rect") return AnnotationShape::RotatedRect;
    if (t == "polygon") return AnnotationShape::Polygon;
    if (t == "line") return AnnotationShape::Line;
    if (t == "point") return AnnotationShape::Point;
    if (t == "keypoints") return AnnotationShape::Keypoints;
    return AnnotationShape::Rectangle;
}

QString AnnotationProject::splitToString(DatasetSplit split)
{
    switch (split) {
    case DatasetSplit::Train: return QStringLiteral("train");
    case DatasetSplit::Val: return QStringLiteral("val");
    case DatasetSplit::Test: return QStringLiteral("test");
    }
    return QStringLiteral("train");
}

DatasetSplit AnnotationProject::splitFromString(const QString& text)
{
    const QString t = text.trimmed().toLower();
    if (t == "val" || t == "valid" || t == "validation") return DatasetSplit::Val;
    if (t == "test") return DatasetSplit::Test;
    return DatasetSplit::Train;
}

QJsonObject AnnotationProject::objectToJson(const AnnotationObject& obj)
{
    QJsonObject out;
    out["id"] = obj.id;
    out["class_id"] = obj.classId;
    out["label"] = obj.label;
    out["description"] = obj.description;
    out["group_id"] = obj.groupId;
    out["shape_type"] = shapeToString(obj.shape);
    out["points"] = pointArray(obj.points);
    out["rotation"] = obj.rotation;
    if (obj.score >= 0.0)
        out["score"] = obj.score;
    QJsonArray vis;
    for (int v : obj.keypointVisibility)
        vis.append(v);
    out["keypoint_visibility"] = vis;
    return out;
}

AnnotationObject AnnotationProject::objectFromJson(const QJsonObject& obj)
{
    AnnotationObject out;
    out.id = obj.value("id").toString();
    out.classId = obj.value("class_id").toInt();
    out.label = obj.value("label").toString();
    out.description = obj.value("description").toString();
    out.groupId = obj.value("group_id").toInt();
    out.shape = shapeFromString(obj.value("shape_type").toString());
    out.points = pointsFromJson(obj.value("points").toArray());
    out.rotation = obj.value("rotation").toDouble();
    out.score = obj.contains("score") ? obj.value("score").toDouble() : -1.0;
    for (const auto& v : obj.value("keypoint_visibility").toArray())
        out.keypointVisibility.append(v.toInt(2));
    return out;
}

} // namespace trtforge

