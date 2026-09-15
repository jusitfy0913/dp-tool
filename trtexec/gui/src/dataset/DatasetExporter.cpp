#include "DatasetExporter.h"

#include "dataset/DatasetYamlWriter.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QHash>
#include <QImageReader>
#include <QRegularExpression>
#include <QStringConverter>
#include <QTextStream>
#include <QXmlStreamReader>
#include <QXmlStreamWriter>
#include <algorithm>

namespace trtforge {

namespace {

QString norm(double value)
{
    return QString::number(std::clamp(value, 0.0, 1.0), 'f', 6);
}

/** Strip optional YAML quotes around a name token. */
QString stripYamlNameToken(QString t)
{
    t = t.trimmed();
    if (t.size() >= 2) {
        const QChar a = t.front();
        const QChar b = t.back();
        if ((a == QLatin1Char('"') && b == QLatin1Char('"')) ||
            (a == QLatin1Char('\'') && b == QLatin1Char('\''))) {
            t = t.mid(1, t.size() - 2).trimmed();
        }
    }
    return t.trimmed();
}

/**
 * Ultralytics data.yaml uses several "names" shapes. TRTForge previously only parsed "0: cat"
 * per line, so "names: [ a, b ]" never loaded and every box fell back to a single "object" class.
 */
void parseYoloDataYamlClassTable(const QString& yamlText, QVector<AnnotationClass>& outClasses)
{
    outClasses.clear();

    // 1) Inline list: names: [ "a", "b" ]  (possibly multiline between brackets)
    {
        static const QRegularExpression bracketRx(
            QStringLiteral(R"(names:\s*\[([\s\S]*?)\])"),
            QRegularExpression::CaseInsensitiveOption);
        const QRegularExpressionMatch bm = bracketRx.match(yamlText);
        if (bm.hasMatch()) {
            const QString inner = bm.captured(1);
            int cid = 0;
            for (QString part : inner.split(QLatin1Char(','))) {
                const QString name = stripYamlNameToken(part);
                if (name.isEmpty())
                    continue;
                outClasses.append({cid, name, QColor::fromHsv((cid * 47) % 360, 210, 240)});
                ++cid;
            }
            if (!outClasses.isEmpty())
                return;
        }
    }

    // 2) Block list under names:
    //    names:
    //      - missing_hole
    {
        const QStringList lines = yamlText.split(QLatin1Char('\n'));
        bool inNames = false;
        int cid = 0;
        for (const QString& line : lines) {
            const QString t = line.trimmed();
            if (!inNames) {
                if (t.startsWith(QStringLiteral("names:"), Qt::CaseInsensitive)) {
                    if (t.contains(QLatin1Char('[')))
                        continue;
                    inNames = true;
                }
                continue;
            }
            if (t.isEmpty() || t.startsWith(QLatin1Char('#')))
                continue;
            static const QRegularExpression dashName(QStringLiteral(R"(^\s*-\s+(.+)$)"));
            const QRegularExpressionMatch dm = dashName.match(line);
            if (dm.hasMatch()) {
                const QString name = stripYamlNameToken(dm.captured(1));
                if (!name.isEmpty()) {
                    outClasses.append({cid, name, QColor::fromHsv((cid * 47) % 360, 210, 240)});
                    ++cid;
                }
                continue;
            }
            // New top-level key (path:, train:, val:, nc:, ...)
            if (t.contains(QLatin1Char(':')) && !line.startsWith(QLatin1Char(' ')) &&
                !line.startsWith(QLatin1Char('\t'))) {
                break;
            }
        }
        if (!outClasses.isEmpty())
            return;
    }

    // 3) Legacy dict lines anywhere: "  0: missing hole"
    static const QRegularExpression idNameRx(QStringLiteral(R"(^\s*([0-9]+)\s*:\s*['\"]?([^'\"\n#]+)['\"]?\s*(?:#.*)?$)"));
    const QStringList lines = yamlText.split(QLatin1Char('\n'));
    for (const QString& line : lines) {
        const QRegularExpressionMatch m = idNameRx.match(line);
        if (!m.hasMatch())
            continue;
        const int id = m.captured(1).toInt();
        QString name = m.captured(2).trimmed();
        name = stripYamlNameToken(name);
        if (name.isEmpty())
            continue;
        outClasses.append({id, name, QColor::fromHsv((id * 47) % 360, 210, 240)});
    }
    std::sort(outClasses.begin(), outClasses.end(),
              [](const AnnotationClass& a, const AnnotationClass& b) { return a.id < b.id; });
}

QRectF boundsOf(const QVector<QPointF>& points)
{
    if (points.isEmpty())
        return {};
    qreal minX = points.first().x();
    qreal minY = points.first().y();
    qreal maxX = minX;
    qreal maxY = minY;
    for (const auto& p : points) {
        minX = qMin(minX, p.x());
        minY = qMin(minY, p.y());
        maxX = qMax(maxX, p.x());
        maxY = qMax(maxY, p.y());
    }
    return QRectF(QPointF(minX, minY), QPointF(maxX, maxY)).normalized();
}

QString splitDir(DatasetSplit split)
{
    return AnnotationProject::splitToString(split);
}

bool ensureDir(const QString& path, QString* error)
{
    QDir dir;
    if (dir.mkpath(path))
        return true;
    if (error) *error = QStringLiteral("无法创建目录：%1").arg(path);
    return false;
}

bool copyImage(const AnnotationImage& image, const QString& outputImagesDir, QString* copiedName, QString* error)
{
    QFileInfo srcInfo(image.path);
    if (!srcInfo.exists()) {
        if (error) *error = QStringLiteral("图片不存在：%1").arg(image.path);
        return false;
    }
    QString fileName = srcInfo.fileName();
    QString dst = QDir(outputImagesDir).filePath(fileName);
    if (QFile::exists(dst))
        QFile::remove(dst);
    if (!QFile::copy(image.path, dst)) {
        if (error) *error = QStringLiteral("复制图片失败：%1 -> %2").arg(image.path, dst);
        return false;
    }
    if (copiedName) *copiedName = fileName;
    return true;
}

QString yoloLine(const AnnotationObject& ann, AnnotationTask task, int w, int h)
{
    if (w <= 0 || h <= 0)
        return {};
    QStringList parts;
    parts << QString::number(ann.classId);

    if (task == AnnotationTask::Detect) {
        QRectF r = boundsOf(ann.points);
        parts << norm(r.center().x() / w)
              << norm(r.center().y() / h)
              << norm(r.width() / w)
              << norm(r.height() / h);
    } else if (task == AnnotationTask::Segment) {
        if (ann.points.size() < 3)
            return {};
        for (const auto& p : ann.points)
            parts << norm(p.x() / w) << norm(p.y() / h);
    } else if (task == AnnotationTask::Pose) {
        QRectF r = boundsOf(ann.points);
        parts << norm(r.center().x() / w)
              << norm(r.center().y() / h)
              << norm(r.width() / w)
              << norm(r.height() / h);
        for (int i = 0; i < 17; ++i) {
            QPointF p = i < ann.points.size() ? ann.points[i] : QPointF(0, 0);
            int v = i < ann.keypointVisibility.size() ? ann.keypointVisibility[i] : 2;
            parts << norm(p.x() / w) << norm(p.y() / h) << QString::number(v);
        }
    } else if (task == AnnotationTask::Obb) {
        QVector<QPointF> pts = ann.points;
        if (pts.size() < 4) {
            QRectF r = boundsOf(pts);
            pts = {r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()};
        }
        if (pts.size() < 4)
            return {};
        for (int i = 0; i < 4; ++i)
            parts << norm(pts[i].x() / w) << norm(pts[i].y() / h);
    }
    return parts.join(' ');
}

QVector<QPointF> denormPairs(const QStringList& tokens, int start, int count, int w, int h)
{
    QVector<QPointF> points;
    for (int i = 0; i + 1 < count; i += 2) {
        double x = tokens.value(start + i).toDouble() * w;
        double y = tokens.value(start + i + 1).toDouble() * h;
        points.append(QPointF(x, y));
    }
    return points;
}

QJsonArray classJson(const AnnotationProject& project)
{
    QJsonArray categories;
    for (const auto& cls : project.classes) {
        QJsonObject obj;
        obj["id"] = cls.id;
        obj["name"] = cls.name;
        obj["supercategory"] = QStringLiteral("object");
        categories.append(obj);
    }
    return categories;
}

} // namespace

DatasetIoResult DatasetExporter::exportYolo(const AnnotationProject& project, const QString& outputDir)
{
    if (project.images.isEmpty())
        return {false, QStringLiteral("项目中没有图片")};

    QDir root(outputDir);
    QString error;
    for (DatasetSplit split : {DatasetSplit::Train, DatasetSplit::Val, DatasetSplit::Test}) {
        if (!ensureDir(root.filePath(QStringLiteral("images/%1").arg(splitDir(split))), &error) ||
            !ensureDir(root.filePath(QStringLiteral("labels/%1").arg(splitDir(split))), &error))
            return {false, error};
    }

    for (const auto& image : project.images) {
        const QString split = splitDir(image.split);
        QString copiedName;
        if (!copyImage(image, root.filePath(QStringLiteral("images/%1").arg(split)), &copiedName, &error))
            return {false, error};

        QFileInfo info(copiedName);
        QFile labelFile(root.filePath(QStringLiteral("labels/%1/%2.txt").arg(split, info.completeBaseName())));
        if (!labelFile.open(QIODevice::WriteOnly | QIODevice::Text))
            return {false, QStringLiteral("无法写入 YOLO 标签：%1").arg(labelFile.fileName())};
        QTextStream out(&labelFile);
        out.setEncoding(QStringConverter::Utf8);
        for (const auto& ann : image.annotations) {
            QString line = yoloLine(ann, project.task, image.width, image.height);
            if (!line.isEmpty())
                out << line << '\n';
        }
    }

    const DatasetIoResult yamlResult = DatasetYamlWriter::writeYoloDatasetYaml(project, outputDir);
    if (!yamlResult.ok)
        return yamlResult;

    return {true, QStringLiteral("YOLO 数据集导出完成：%1").arg(outputDir)};
}

DatasetIoResult DatasetExporter::exportCoco(const AnnotationProject& project, const QString& outputFile)
{
    if (project.task == AnnotationTask::Obb)
        return {false, QStringLiteral("标准 COCO 不支持 OBB；请导出 YOLO OBB")};

    QJsonObject root;
    QJsonArray images;
    QJsonArray annotations;
    int imageId = 1;
    int annId = 1;

    for (const auto& image : project.images) {
        QJsonObject img;
        img["id"] = imageId;
        img["file_name"] = QFileInfo(image.path).fileName();
        img["width"] = image.width;
        img["height"] = image.height;
        images.append(img);

        for (const auto& ann : image.annotations) {
            QRectF r = boundsOf(ann.points);
            QJsonObject obj;
            obj["id"] = annId++;
            obj["image_id"] = imageId;
            obj["category_id"] = ann.classId;
            obj["iscrowd"] = 0;
            QJsonArray bbox;
            bbox << r.x() << r.y() << r.width() << r.height();
            obj["bbox"] = bbox;
            obj["area"] = r.width() * r.height();

            if (project.task == AnnotationTask::Segment) {
                QJsonArray poly;
                for (const auto& p : ann.points) {
                    poly.append(p.x());
                    poly.append(p.y());
                }
                QJsonArray seg;
                seg.append(poly);
                obj["segmentation"] = seg;
            } else if (project.task == AnnotationTask::Pose) {
                QJsonArray kpts;
                int visible = 0;
                for (int i = 0; i < 17; ++i) {
                    QPointF p = i < ann.points.size() ? ann.points[i] : QPointF(0, 0);
                    int v = i < ann.keypointVisibility.size() ? ann.keypointVisibility[i] : 2;
                    if (v > 0) ++visible;
                    kpts << p.x() << p.y() << v;
                }
                obj["keypoints"] = kpts;
                obj["num_keypoints"] = visible;
            } else {
                obj["segmentation"] = QJsonArray();
            }
            annotations.append(obj);
        }
        ++imageId;
    }

    root["images"] = images;
    root["annotations"] = annotations;
    root["categories"] = classJson(project);

    QFile file(outputFile);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return {false, QStringLiteral("无法写入 COCO 文件：%1").arg(outputFile)};
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    return {true, QStringLiteral("COCO 标注导出完成：%1").arg(outputFile)};
}

DatasetIoResult DatasetExporter::exportVoc(const AnnotationProject& project, const QString& outputDir)
{
    if (project.task != AnnotationTask::Detect)
        return {false, QStringLiteral("VOC 仅支持标准水平矩形检测，不支持 segment/pose/obb")};

    QDir root(outputDir);
    QString error;
    if (!ensureDir(root.filePath("Annotations"), &error) ||
        !ensureDir(root.filePath("JPEGImages"), &error))
        return {false, error};

    for (const auto& image : project.images) {
        QString copiedName;
        if (!copyImage(image, root.filePath("JPEGImages"), &copiedName, &error))
            return {false, error};

        QFileInfo info(copiedName);
        QFile xml(root.filePath(QStringLiteral("Annotations/%1.xml").arg(info.completeBaseName())));
        if (!xml.open(QIODevice::WriteOnly | QIODevice::Text))
            return {false, QStringLiteral("无法写入 VOC XML：%1").arg(xml.fileName())};

        QXmlStreamWriter w(&xml);
        w.setAutoFormatting(true);
        w.writeStartDocument();
        w.writeStartElement("annotation");
        w.writeTextElement("folder", "JPEGImages");
        w.writeTextElement("filename", copiedName);
        w.writeStartElement("size");
        w.writeTextElement("width", QString::number(image.width));
        w.writeTextElement("height", QString::number(image.height));
        w.writeTextElement("depth", "3");
        w.writeEndElement();

        for (const auto& ann : image.annotations) {
            QRectF r = boundsOf(ann.points);
            w.writeStartElement("object");
            w.writeTextElement("name", project.classById(ann.classId).name);
            w.writeTextElement("pose", "Unspecified");
            w.writeTextElement("truncated", "0");
            w.writeTextElement("difficult", "0");
            w.writeStartElement("bndbox");
            w.writeTextElement("xmin", QString::number(qRound(r.left())));
            w.writeTextElement("ymin", QString::number(qRound(r.top())));
            w.writeTextElement("xmax", QString::number(qRound(r.right())));
            w.writeTextElement("ymax", QString::number(qRound(r.bottom())));
            w.writeEndElement();
            w.writeEndElement();
        }
        w.writeEndElement();
        w.writeEndDocument();
    }
    return {true, QStringLiteral("VOC 数据集导出完成：%1").arg(outputDir)};
}

DatasetIoResult DatasetExporter::importYolo(AnnotationProject& project, const QString& datasetYamlOrRoot, AnnotationTask task)
{
    QFileInfo inputInfo(datasetYamlOrRoot);
    QDir root(inputInfo.isDir() ? datasetYamlOrRoot : inputInfo.absolutePath());
    project.clear();
    project.task = task;
    project.projectPath.clear();

    if (inputInfo.isFile()) {
        QFile yaml(inputInfo.absoluteFilePath());
        if (yaml.open(QIODevice::ReadOnly | QIODevice::Text)) {
            project.classes.clear();
            const QString yamlText = QString::fromUtf8(yaml.readAll());
            yaml.close();
            parseYoloDataYamlClassTable(yamlText, project.classes);
            project.ensureDefaultClass();
        }
    }

    const QStringList splitNames = {"train", "val", "test"};
    for (const QString& splitName : splitNames) {
        QDir imgDir(root.filePath(QStringLiteral("images/%1").arg(splitName)));
        QDir labelDir(root.filePath(QStringLiteral("labels/%1").arg(splitName)));
        if (!imgDir.exists())
            continue;
        const auto files = imgDir.entryInfoList({"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"}, QDir::Files);
        for (const QFileInfo& imgInfo : files) {
            AnnotationImage image;
            image.id = imgInfo.completeBaseName();
            image.path = imgInfo.absoluteFilePath();
            image.split = AnnotationProject::splitFromString(splitName);
            QImageReader reader(image.path);
            QSize size = reader.size();
            image.width = size.width();
            image.height = size.height();

            QFile labels(labelDir.filePath(imgInfo.completeBaseName() + ".txt"));
            if (labels.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QTextStream in(&labels);
                while (!in.atEnd()) {
                    QStringList tokens = in.readLine().simplified().split(' ', Qt::SkipEmptyParts);
                    if (tokens.size() < 5)
                        continue;
                    AnnotationObject ann;
                    ann.id = QStringLiteral("ann_%1").arg(image.annotations.size() + 1);
                    ann.classId = tokens[0].toInt();
                    ann.label = project.classById(ann.classId).name;
                    if (task == AnnotationTask::Detect && tokens.size() >= 5) {
                        ann.shape = AnnotationShape::Rectangle;
                        double cx = tokens[1].toDouble() * image.width;
                        double cy = tokens[2].toDouble() * image.height;
                        double bw = tokens[3].toDouble() * image.width;
                        double bh = tokens[4].toDouble() * image.height;
                        ann.points = {QPointF(cx - bw / 2, cy - bh / 2), QPointF(cx + bw / 2, cy + bh / 2)};
                    } else if (task == AnnotationTask::Segment) {
                        ann.shape = AnnotationShape::Polygon;
                        ann.points = denormPairs(tokens, 1, tokens.size() - 1, image.width, image.height);
                    } else if (task == AnnotationTask::Obb && tokens.size() >= 9) {
                        ann.shape = AnnotationShape::RotatedRect;
                        ann.points = denormPairs(tokens, 1, 8, image.width, image.height);
                    } else if (task == AnnotationTask::Pose) {
                        ann.shape = AnnotationShape::Keypoints;
                        for (int i = 5; i + 2 < tokens.size(); i += 3) {
                            ann.points.append(QPointF(tokens[i].toDouble() * image.width, tokens[i + 1].toDouble() * image.height));
                            ann.keypointVisibility.append(tokens[i + 2].toInt());
                        }
                    }
                    image.annotations.append(ann);
                }
            }
            project.images.append(image);
        }
    }
    project.dirty = true;
    return {true, QStringLiteral("YOLO 数据集已加载：%1 张图片").arg(project.images.size())};
}

DatasetIoResult DatasetExporter::importCoco(AnnotationProject& project, const QString& annotationFile, AnnotationTask task)
{
    QFile file(annotationFile);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return {false, QStringLiteral("无法读取 COCO 文件：%1").arg(annotationFile)};
    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject())
        return {false, QStringLiteral("COCO JSON 无效：%1").arg(err.errorString())};

    project.clear();
    project.task = task;
    QJsonObject root = doc.object();
    project.classes.clear();
    for (const auto& v : root.value("categories").toArray()) {
        QJsonObject c = v.toObject();
        int id = c.value("id").toInt(project.classes.size());
        project.classes.append({id, c.value("name").toString(QStringLiteral("class_%1").arg(id)), QColor::fromHsv((id * 47) % 360, 210, 240)});
    }
    project.ensureDefaultClass();

    QHash<int, int> imageIndexById;
    const QString base = QFileInfo(annotationFile).absolutePath();
    for (const auto& v : root.value("images").toArray()) {
        QJsonObject img = v.toObject();
        AnnotationImage image;
        int id = img.value("id").toInt(project.images.size() + 1);
        image.id = QString::number(id);
        image.path = QDir(base).filePath(img.value("file_name").toString());
        image.width = img.value("width").toInt();
        image.height = img.value("height").toInt();
        imageIndexById.insert(id, project.images.size());
        project.images.append(image);
    }

    for (const auto& v : root.value("annotations").toArray()) {
        QJsonObject obj = v.toObject();
        int imageId = obj.value("image_id").toInt();
        if (!imageIndexById.contains(imageId))
            continue;
        AnnotationImage& image = project.images[imageIndexById.value(imageId)];
        AnnotationObject ann;
        ann.id = QStringLiteral("ann_%1").arg(image.annotations.size() + 1);
        ann.classId = obj.value("category_id").toInt();
        ann.label = project.classById(ann.classId).name;
        if (task == AnnotationTask::Segment && obj.value("segmentation").isArray() && !obj.value("segmentation").toArray().isEmpty()) {
            ann.shape = AnnotationShape::Polygon;
            QJsonArray poly = obj.value("segmentation").toArray().first().toArray();
            for (int i = 0; i + 1 < poly.size(); i += 2)
                ann.points.append(QPointF(poly[i].toDouble(), poly[i + 1].toDouble()));
        } else if (task == AnnotationTask::Pose && obj.value("keypoints").isArray()) {
            ann.shape = AnnotationShape::Keypoints;
            QJsonArray k = obj.value("keypoints").toArray();
            for (int i = 0; i + 2 < k.size(); i += 3) {
                ann.points.append(QPointF(k[i].toDouble(), k[i + 1].toDouble()));
                ann.keypointVisibility.append(k[i + 2].toInt());
            }
        } else {
            ann.shape = AnnotationShape::Rectangle;
            QJsonArray b = obj.value("bbox").toArray();
            if (b.size() >= 4)
                ann.points = {QPointF(b[0].toDouble(), b[1].toDouble()), QPointF(b[0].toDouble() + b[2].toDouble(), b[1].toDouble() + b[3].toDouble())};
        }
        image.annotations.append(ann);
    }
    project.dirty = true;
    return {true, QStringLiteral("COCO 标注已加载：%1 张图片").arg(project.images.size())};
}

DatasetIoResult DatasetExporter::importVoc(AnnotationProject& project, const QString& xmlDir)
{
    project.clear();
    project.task = AnnotationTask::Detect;
    project.classes.clear();
    QDir dir(xmlDir);
    const auto files = dir.entryInfoList({"*.xml"}, QDir::Files);
    for (const QFileInfo& xmlInfo : files) {
        QFile file(xmlInfo.absoluteFilePath());
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            continue;
        QXmlStreamReader r(&file);
        AnnotationImage image;
        image.id = xmlInfo.completeBaseName();
        AnnotationObject current;
        QString currentElement;
        while (!r.atEnd()) {
            r.readNext();
            if (r.isStartElement()) {
                currentElement = r.name().toString();
                if (currentElement == "object")
                    current = AnnotationObject{};
            } else if (r.isCharacters() && !r.isWhitespace()) {
                const QString text = r.text().toString();
                if (currentElement == "filename")
                    image.path = dir.filePath(text);
                else if (currentElement == "width")
                    image.width = text.toInt();
                else if (currentElement == "height")
                    image.height = text.toInt();
                else if (currentElement == "name") {
                    int classId = -1;
                    for (const auto& cls : project.classes) {
                        if (cls.name == text) {
                            classId = cls.id;
                            break;
                        }
                    }
                    if (classId < 0)
                        classId = project.addClass(text);
                    current.classId = classId;
                    current.label = text;
                    current.shape = AnnotationShape::Rectangle;
                } else if (currentElement == "xmin") {
                    current.points.resize(2);
                    current.points[0].setX(text.toDouble());
                } else if (currentElement == "ymin") {
                    current.points.resize(2);
                    current.points[0].setY(text.toDouble());
                } else if (currentElement == "xmax") {
                    current.points.resize(2);
                    current.points[1].setX(text.toDouble());
                } else if (currentElement == "ymax") {
                    current.points.resize(2);
                    current.points[1].setY(text.toDouble());
                }
            } else if (r.isEndElement() && r.name() == "object") {
                current.id = QStringLiteral("ann_%1").arg(image.annotations.size() + 1);
                image.annotations.append(current);
            }
        }
        project.images.append(image);
    }
    project.ensureDefaultClass();
    project.dirty = true;
    return {true, QStringLiteral("VOC 标注已加载：%1 张图片").arg(project.images.size())};
}

} // namespace trtforge

