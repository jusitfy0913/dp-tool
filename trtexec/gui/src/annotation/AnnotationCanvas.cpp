#include "AnnotationCanvas.h"

#include <QLineF>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

namespace trtforge {

AnnotationCanvas::AnnotationCanvas(QWidget* parent) : QWidget(parent)
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    setMinimumSize(640, 420);
}

void AnnotationCanvas::setTool(AnnotationTool tool)
{
    m_tool = tool;
    m_dragging = false;
    clearDraft();
    m_polyPoints.clear();
    m_posePoints.clear();
    update();
}

void AnnotationCanvas::setImage(const QString& path)
{
    m_imagePath = path;
    m_image.load(path);
    m_zoom = 1.0;
    m_pan = QPointF();
    clearDraft();
    m_polyPoints.clear();
    m_posePoints.clear();
    update();
}

void AnnotationCanvas::setAnnotations(QVector<AnnotationObject>* annotations)
{
    m_annotations = annotations;
    if (!m_annotations || m_selectedObject >= m_annotations->size())
        m_selectedObject = -1;
    update();
}

void AnnotationCanvas::setClasses(const QVector<AnnotationClass>* classes)
{
    m_classes = classes;
    update();
}

void AnnotationCanvas::setSelectedObject(int index)
{
    m_selectedObject = index;
    update();
}

void AnnotationCanvas::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.fillRect(rect(), QColor("#07111F"));

    QRectF imageRect = imageRectOnWidget();
    p.fillRect(imageRect, QColor("#D7EEF2"));
    if (!m_image.isNull())
        p.drawImage(imageRect, m_image);

    p.setPen(QPen(QColor(31, 182, 255, 80), 1, Qt::DashLine));
    p.drawLine(QPointF(width() / 2.0, 0), QPointF(width() / 2.0, height()));
    p.drawLine(QPointF(0, height() / 2.0), QPointF(width(), height() / 2.0));

    if (m_annotations) {
        for (int i = 0; i < m_annotations->size(); ++i)
            drawObject(p, m_annotations->at(i), false);
    }
    if (m_dragging || !m_draft.points.isEmpty())
        drawObject(p, m_draft, true);
    if (hasObbDraft())
        drawObbControls(p);
    if (hasRectangleDraft())
        drawRectangleControls(p);

    if (!m_polyPoints.isEmpty()) {
        QPolygonF poly;
        for (const auto& point : m_polyPoints)
            poly << imageToWidget(point);
        p.setPen(QPen(QColor("#A855F7"), 2, Qt::DashLine));
        p.setBrush(QColor("#A855F7"));
        p.drawPolyline(poly);
        for (const auto& point : poly)
            p.drawEllipse(point, 4, 4);
    }
    if (!m_posePoints.isEmpty()) {
        AnnotationObject obj;
        obj.classId = 0;
        obj.shape = AnnotationShape::Keypoints;
        obj.points = m_posePoints;
        drawObject(p, obj, true);
    }

    p.setPen(QColor("#9FC8E8"));
    p.drawText(12, 22, m_image.isNull()
        ? QStringLiteral("导入图片目录后开始标注")
        : QStringLiteral("%1 · %2%").arg(m_imagePath).arg(int(m_zoom * 100)));
}

void AnnotationCanvas::mousePressEvent(QMouseEvent* event)
{
    m_lastWidget = event->position();
    QPointF img = widgetToImage(event->position());

    if ((m_tool == AnnotationTool::RotatedRect || m_tool == AnnotationTool::Rectangle) && event->button() == Qt::RightButton) {
        if (hasAdjustableBoxDraft() && isDraftLargeEnough()) {
            finishCurrentDraft();
        } else if (hasAdjustableBoxDraft()) {
            emit statusChanged(QStringLiteral("矩形太小，请重新拖出更大的矩形"));
            clearDraft();
            update();
        }
        return;
    }

    if (m_tool == AnnotationTool::Polygon && event->button() == Qt::RightButton) {
        if (m_polyPoints.size() >= 3) {
            AnnotationObject obj;
            obj.shape = AnnotationShape::Polygon;
            obj.points = m_polyPoints;
            m_polyPoints.clear();
            emit draftCompleted(obj);
            update();
        } else if (!m_polyPoints.isEmpty()) {
            emit statusChanged(QStringLiteral("多边形至少需要 3 个点，继续左键添加点位"));
        }
        return;
    }

    if (m_tool == AnnotationTool::Pan || event->button() == Qt::MiddleButton) {
        m_dragging = true;
        m_movingObject = false;
        return;
    }
    if (event->button() != Qt::LeftButton || m_image.isNull())
        return;

    if (m_tool == AnnotationTool::RotatedRect && hasObbDraft()) {
        const QPointF center = imageToWidget(obbCenterImage());
        const QPointF rotateHandle = obbRotateHandleWidget();
        if (QLineF(event->position(), rotateHandle).length() <= 12.0) {
            m_dragging = true;
            m_movingObject = false;
            m_obbEditMode = ObbEditMode::Rotating;
            m_obbDragStartImage = img;
            m_obbDragStartPoints = m_draft.points;
            emit statusChanged(QStringLiteral("拖动旋转手柄调整角度，右键结束并确认"));
            return;
        }
        if (QLineF(event->position(), center).length() <= 12.0) {
            m_dragging = true;
            m_movingObject = false;
            m_obbEditMode = ObbEditMode::Moving;
            m_obbDragStartImage = img;
            m_obbDragStartPoints = m_draft.points;
            emit statusChanged(QStringLiteral("拖动中心点平移旋转框，右键结束并确认"));
            return;
        }
        emit statusChanged(QStringLiteral("请拖动中心点平移，或拖动右侧旋转手柄旋转；右键结束"));
        return;
    }

    if (m_tool == AnnotationTool::Rectangle && hasRectangleDraft()) {
        const QPointF center = imageToWidget(draftCenterImage());
        if (QLineF(event->position(), center).length() <= 12.0) {
            m_dragging = true;
            m_movingObject = false;
            m_obbEditMode = ObbEditMode::Moving;
            m_obbDragStartImage = img;
            m_obbDragStartPoints = m_draft.points;
            emit statusChanged(QStringLiteral("拖动中心点平移矩形，右键结束并确认"));
            return;
        }
        emit statusChanged(QStringLiteral("请拖动绿色中心点平移矩形；右键结束"));
        return;
    }

    if (m_tool == AnnotationTool::Select) {
        int hit = hitTestObject(event->position());
        m_selectedObject = hit;
        emit objectSelected(hit);
        if (hit >= 0) {
            m_dragging = true;
            m_movingObject = true;
            m_lastImage = img;
            emit editStarted();
        }
        update();
        return;
    }

    if (m_tool == AnnotationTool::Polygon) {
        m_polyPoints.append(img);
        emit statusChanged(QStringLiteral("多边形点数：%1，右键闭合完成").arg(m_polyPoints.size()));
        update();
        return;
    }
    if (m_tool == AnnotationTool::Point) {
        AnnotationObject obj;
        obj.shape = AnnotationShape::Point;
        obj.points = {img};
        emit draftCompleted(obj);
        return;
    }
    if (m_tool == AnnotationTool::Pose) {
        m_posePoints.append(img);
        if (m_posePoints.size() >= 17) {
            AnnotationObject obj;
            obj.shape = AnnotationShape::Keypoints;
            obj.points = m_posePoints;
            obj.keypointVisibility = QVector<int>(m_posePoints.size(), 2);
            m_posePoints.clear();
            emit draftCompleted(obj);
        } else {
            emit statusChanged(QStringLiteral("COCO17 姿态点：%1 / 17").arg(m_posePoints.size()));
        }
        update();
        return;
    }

    m_dragging = true;
    m_movingObject = false;
    m_obbEditMode = (m_tool == AnnotationTool::RotatedRect || m_tool == AnnotationTool::Rectangle)
        ? ObbEditMode::Drawing
        : ObbEditMode::None;
    m_startImage = img;
    m_lastImage = img;
    m_draft = AnnotationObject{};
    m_draft.shape = (m_tool == AnnotationTool::RotatedRect) ? AnnotationShape::RotatedRect :
                    (m_tool == AnnotationTool::Line) ? AnnotationShape::Line : AnnotationShape::Rectangle;
    m_draft.points = {img, img};
}

void AnnotationCanvas::mouseMoveEvent(QMouseEvent* event)
{
    QPointF pos = event->position();
    if (m_dragging && hasAdjustableBoxDraft() && (m_obbEditMode == ObbEditMode::Moving || m_obbEditMode == ObbEditMode::Rotating)) {
        const QPointF img = widgetToImage(pos);
        if (m_obbEditMode == ObbEditMode::Moving) {
            const QPointF delta = img - m_obbDragStartImage;
            m_draft.points = m_obbDragStartPoints;
            for (QPointF& point : m_draft.points)
                point += delta;
        } else {
            const QPointF center = obbCenterImage();
            const double startAngle = std::atan2(m_obbDragStartImage.y() - center.y(),
                                                 m_obbDragStartImage.x() - center.x());
            const double currentAngle = std::atan2(img.y() - center.y(), img.x() - center.x());
            const double delta = currentAngle - startAngle;
            const double c = std::cos(delta);
            const double s = std::sin(delta);
            m_draft.points = m_obbDragStartPoints;
            for (QPointF& point : m_draft.points) {
                const QPointF rel = point - center;
                point = QPointF(center.x() + rel.x() * c - rel.y() * s,
                                center.y() + rel.x() * s + rel.y() * c);
            }
        }
        update();
        return;
    }
    if (m_dragging && (m_tool == AnnotationTool::Pan || event->buttons() & Qt::MiddleButton)) {
        m_pan += pos - m_lastWidget;
        m_lastWidget = pos;
        update();
        return;
    }
    if (m_dragging && m_movingObject && m_annotations && m_selectedObject >= 0 && m_selectedObject < m_annotations->size()) {
        QPointF img = widgetToImage(pos);
        QPointF delta = img - m_lastImage;
        AnnotationObject& obj = (*m_annotations)[m_selectedObject];
        for (QPointF& p : obj.points)
            p += delta;
        m_lastImage = img;
        emit annotationsEdited();
        update();
        return;
    }
    if (m_dragging && !m_draft.points.isEmpty()) {
        QPointF img = widgetToImage(pos);
        if (m_draft.shape == AnnotationShape::RotatedRect) {
            QRectF r(m_startImage, img);
            r = r.normalized();
            m_draft.points = {r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()};
        } else {
            m_draft.points = {m_startImage, img};
        }
        update();
    }
}

void AnnotationCanvas::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton && event->button() != Qt::MiddleButton)
        return;
    if (m_tool == AnnotationTool::Pan || event->button() == Qt::MiddleButton) {
        m_dragging = false;
        m_movingObject = false;
        return;
    }
    if (m_movingObject) {
        m_dragging = false;
        m_movingObject = false;
        return;
    }
    if ((m_tool == AnnotationTool::RotatedRect || m_tool == AnnotationTool::Rectangle) && hasAdjustableBoxDraft()
        && (m_obbEditMode == ObbEditMode::Moving || m_obbEditMode == ObbEditMode::Rotating)) {
        m_dragging = false;
        m_obbEditMode = ObbEditMode::None;
        emit statusChanged(m_tool == AnnotationTool::Rectangle
            ? QStringLiteral("可继续拖动中心点平移矩形；右键结束并确认")
            : QStringLiteral("可继续拖动中心点/旋转手柄调整；右键结束并确认"));
        return;
    }
    if ((m_tool == AnnotationTool::RotatedRect || m_tool == AnnotationTool::Rectangle)
        && m_obbEditMode == ObbEditMode::Drawing && !m_draft.points.isEmpty()) {
        m_dragging = false;
        m_obbEditMode = ObbEditMode::None;
        if (!isDraftLargeEnough()) {
            emit statusChanged(QStringLiteral("矩形太小，请重新拖出更大的矩形"));
            clearDraft();
        } else {
            emit statusChanged(m_tool == AnnotationTool::Rectangle
                ? QStringLiteral("已创建矩形草稿：拖动绿色中心点平移，右键结束并确认")
                : QStringLiteral("已创建旋转框草稿：拖动中心点平移，拖动右侧手柄旋转，右键结束并确认"));
        }
        update();
        return;
    }
    if (m_dragging && !m_draft.points.isEmpty()) {
        m_dragging = false;
        finishCurrentDraft();
    }
}

void AnnotationCanvas::mouseDoubleClickEvent(QMouseEvent*)
{
    if (m_tool == AnnotationTool::Polygon)
        emit statusChanged(QStringLiteral("多边形请使用左键添加点位，右键闭合完成"));
}

void AnnotationCanvas::wheelEvent(QWheelEvent* event)
{
    const double factor = event->angleDelta().y() > 0 ? 1.1 : 0.9;
    m_zoom = std::clamp(m_zoom * factor, 0.1, 8.0);
    update();
}

QRectF AnnotationCanvas::imageRectOnWidget() const
{
    QSizeF imgSize = m_image.isNull() ? QSizeF(1280, 720) : QSizeF(m_image.size());
    double scale = qMin(width() / imgSize.width(), height() / imgSize.height()) * 0.92 * m_zoom;
    QSizeF scaled = imgSize * scale;
    QPointF topLeft((width() - scaled.width()) / 2.0, (height() - scaled.height()) / 2.0);
    return QRectF(topLeft + m_pan, scaled);
}

QPointF AnnotationCanvas::imageToWidget(const QPointF& p) const
{
    QRectF r = imageRectOnWidget();
    QSizeF imgSize = m_image.isNull() ? QSizeF(1, 1) : QSizeF(m_image.size());
    return QPointF(r.left() + p.x() * r.width() / imgSize.width(),
                   r.top() + p.y() * r.height() / imgSize.height());
}

QPointF AnnotationCanvas::widgetToImage(const QPointF& p) const
{
    QRectF r = imageRectOnWidget();
    QSizeF imgSize = m_image.isNull() ? QSizeF(1, 1) : QSizeF(m_image.size());
    return QPointF((p.x() - r.left()) * imgSize.width() / r.width(),
                   (p.y() - r.top()) * imgSize.height() / r.height());
}

void AnnotationCanvas::finishCurrentDraft()
{
    if (!isDraftLargeEnough())
        return;
    emit draftCompleted(m_draft);
    clearDraft();
    update();
}

bool AnnotationCanvas::hasObbDraft() const
{
    return m_tool == AnnotationTool::RotatedRect
        && m_draft.shape == AnnotationShape::RotatedRect
        && m_draft.points.size() >= 4;
}

bool AnnotationCanvas::hasRectangleDraft() const
{
    return m_tool == AnnotationTool::Rectangle
        && m_draft.shape == AnnotationShape::Rectangle
        && m_draft.points.size() >= 2;
}

bool AnnotationCanvas::hasAdjustableBoxDraft() const
{
    return hasObbDraft() || hasRectangleDraft();
}

bool AnnotationCanvas::isDraftLargeEnough() const
{
    if (m_draft.points.isEmpty())
        return false;
    if (m_draft.shape == AnnotationShape::RotatedRect && m_draft.points.size() >= 4) {
        const double w = QLineF(m_draft.points[0], m_draft.points[1]).length();
        const double h = QLineF(m_draft.points[1], m_draft.points[2]).length();
        return w >= 3.0 && h >= 3.0;
    }
    if (m_draft.points.size() >= 2) {
        const QRectF b = QRectF(m_draft.points.first(), m_draft.points.last()).normalized();
        return b.width() >= 3.0 && b.height() >= 3.0;
    }
    return true;
}

QPointF AnnotationCanvas::draftCenterImage() const
{
    if (hasObbDraft())
        return obbCenterImage();
    if (hasRectangleDraft()) {
        const QRectF rect(m_draft.points[0], m_draft.points[1]);
        return rect.normalized().center();
    }
    return {};
}

QPointF AnnotationCanvas::obbCenterImage() const
{
    if (!hasObbDraft())
        return {};
    QPointF center;
    for (int i = 0; i < 4; ++i)
        center += m_draft.points[i];
    return center / 4.0;
}

QPointF AnnotationCanvas::obbRightMidWidget() const
{
    if (!hasObbDraft())
        return {};
    return imageToWidget((m_draft.points[1] + m_draft.points[2]) / 2.0);
}

QPointF AnnotationCanvas::obbRotateHandleWidget() const
{
    if (!hasObbDraft())
        return {};
    const QPointF center = imageToWidget(obbCenterImage());
    const QPointF rightMid = obbRightMidWidget();
    const QLineF axis(center, rightMid);
    if (axis.length() < 1.0)
        return rightMid;
    const QPointF unit((rightMid.x() - center.x()) / axis.length(),
                       (rightMid.y() - center.y()) / axis.length());
    return rightMid + unit * 32.0;
}

void AnnotationCanvas::drawObbControls(QPainter& painter) const
{
    if (!hasObbDraft())
        return;

    const QPointF center = imageToWidget(obbCenterImage());
    const QPointF rightMid = obbRightMidWidget();
    const QPointF rotateHandle = obbRotateHandleWidget();

    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(QColor("#FFD166"), 2, Qt::DashLine));
    painter.drawLine(rightMid, rotateHandle);

    painter.setPen(QPen(QColor("#07111F"), 2));
    painter.setBrush(QColor("#22C55E"));
    painter.drawEllipse(center, 7, 7);

    painter.setBrush(QColor("#FFD166"));
    painter.drawEllipse(rotateHandle, 8, 8);

    painter.setPen(QColor("#EAF6FF"));
    painter.drawText(rotateHandle + QPointF(10, -8), QStringLiteral("旋转"));
    painter.drawText(center + QPointF(10, 18), QStringLiteral("平移"));
    painter.restore();
}

void AnnotationCanvas::drawRectangleControls(QPainter& painter) const
{
    if (!hasRectangleDraft())
        return;

    const QPointF center = imageToWidget(draftCenterImage());

    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(QColor("#07111F"), 2));
    painter.setBrush(QColor("#22C55E"));
    painter.drawEllipse(center, 7, 7);
    painter.setPen(QColor("#EAF6FF"));
    painter.drawText(center + QPointF(10, 18), QStringLiteral("平移"));
    painter.restore();
}

void AnnotationCanvas::clearDraft()
{
    m_draft = AnnotationObject{};
    m_obbEditMode = ObbEditMode::None;
    m_obbDragStartImage = {};
    m_obbDragStartPoints.clear();
}

void AnnotationCanvas::drawObject(QPainter& painter, const AnnotationObject& obj, bool draft) const
{
    QColor c = draft ? QColor("#A855F7") : colorForClass(obj.classId);
    bool selected = false;
    if (!draft && m_annotations) {
        const AnnotationObject* base = m_annotations->constData();
        const AnnotationObject* cur = &obj;
        selected = m_selectedObject >= 0 && m_selectedObject < m_annotations->size() && cur == base + m_selectedObject;
    }
    QPen pen(selected ? QColor("#FFD166") : c, selected ? 4 : 2, draft ? Qt::DashLine : Qt::SolidLine);
    painter.setPen(pen);
    painter.setBrush(QColor(c.red(), c.green(), c.blue(), draft ? 34 : 26));

    if (obj.shape == AnnotationShape::Rectangle && obj.points.size() >= 2) {
        painter.drawRect(QRectF(imageToWidget(obj.points[0]), imageToWidget(obj.points[1])).normalized());
    } else if (obj.shape == AnnotationShape::RotatedRect && obj.points.size() >= 4) {
        QPolygonF poly;
        for (int i = 0; i < 4; ++i)
            poly << imageToWidget(obj.points[i]);
        painter.drawPolygon(poly);
    } else if (obj.shape == AnnotationShape::Polygon && obj.points.size() >= 2) {
        QPolygonF poly;
        for (const auto& p : obj.points)
            poly << imageToWidget(p);
        painter.drawPolygon(poly);
    } else if (obj.shape == AnnotationShape::Line && obj.points.size() >= 2) {
        painter.drawLine(imageToWidget(obj.points[0]), imageToWidget(obj.points[1]));
    } else if (obj.shape == AnnotationShape::Point && !obj.points.isEmpty()) {
        painter.setBrush(c);
        painter.drawEllipse(imageToWidget(obj.points[0]), 4, 4);
    } else if (obj.shape == AnnotationShape::Keypoints) {
        painter.setBrush(c);
        for (const auto& p : obj.points)
            painter.drawEllipse(imageToWidget(p), 4, 4);
        for (int i = 1; i < obj.points.size(); ++i)
            painter.drawLine(imageToWidget(obj.points[i - 1]), imageToWidget(obj.points[i]));
    }

    if (!draft)
        drawObjectLabel(painter, obj, c);
}

void AnnotationCanvas::drawObjectLabel(QPainter& painter, const AnnotationObject& obj, const QColor& color) const
{
    const QString text = labelTextForObject(obj);
    const QRectF bounds = objectBoundsOnWidget(obj);
    if (text.isEmpty() || bounds.isEmpty())
        return;

    painter.save();
    QFont font = painter.font();
    font.setPointSize(9);
    font.setBold(true);
    painter.setFont(font);

    const QFontMetrics fm(font);
    constexpr int padX = 6;
    constexpr int padY = 3;
    const int labelW = fm.horizontalAdvance(text) + padX * 2;
    const int labelH = fm.height() + padY * 2;

    QRectF imageArea = imageRectOnWidget().intersected(rect());
    if (imageArea.isEmpty())
        imageArea = rect();

    double x = bounds.left();
    double y = bounds.top() - labelH - 3.0;
    if (y < imageArea.top() + 2.0)
        y = bounds.top() + 3.0;
    x = std::clamp(x, imageArea.left() + 2.0, imageArea.right() - labelW - 2.0);
    y = std::clamp(y, imageArea.top() + 2.0, imageArea.bottom() - labelH - 2.0);

    const QRectF labelRect(x, y, labelW, labelH);
    painter.setPen(QPen(color, 1));
    painter.setBrush(QColor(2, 4, 7, 215));
    painter.drawRoundedRect(labelRect, 4, 4);
    painter.setPen(QColor("#EAF6FF"));
    painter.drawText(labelRect.adjusted(padX, padY, -padX, -padY), Qt::AlignVCenter | Qt::AlignLeft, text);
    painter.restore();
}

QRectF AnnotationCanvas::objectBoundsOnWidget(const AnnotationObject& obj) const
{
    if (obj.points.isEmpty())
        return {};

    if (obj.shape == AnnotationShape::Rectangle && obj.points.size() >= 2) {
        return QRectF(imageToWidget(obj.points[0]), imageToWidget(obj.points[1]))
            .normalized()
            .adjusted(-1, -1, 1, 1);
    }

    QPointF first = imageToWidget(obj.points.first());
    double minX = first.x();
    double minY = first.y();
    double maxX = first.x();
    double maxY = first.y();
    for (const QPointF& point : obj.points) {
        const QPointF widgetPoint = imageToWidget(point);
        minX = std::min(minX, widgetPoint.x());
        minY = std::min(minY, widgetPoint.y());
        maxX = std::max(maxX, widgetPoint.x());
        maxY = std::max(maxY, widgetPoint.y());
    }
    return QRectF(QPointF(minX, minY), QPointF(maxX, maxY)).normalized().adjusted(-1, -1, 1, 1);
}

QString AnnotationCanvas::labelTextForObject(const AnnotationObject& obj) const
{
    QString name = obj.label.trimmed();
    if (m_classes) {
        for (const auto& cls : *m_classes) {
            if (cls.id == obj.classId) {
                name = cls.name.trimmed();
                break;
            }
        }
    }
    if (name.isEmpty())
        name = QStringLiteral("class_%1").arg(obj.classId);
    return QStringLiteral("%1 %2").arg(obj.classId, 2, 10, QLatin1Char('0')).arg(name);
}

bool AnnotationCanvas::objectContainsWidgetPoint(const AnnotationObject& obj, const QPointF& widgetPoint) const
{
    if (obj.points.isEmpty())
        return false;

    if (obj.shape == AnnotationShape::Rectangle && obj.points.size() >= 2) {
        QRectF r(imageToWidget(obj.points[0]), imageToWidget(obj.points[1]));
        return r.normalized().adjusted(-6, -6, 6, 6).contains(widgetPoint);
    }

    if ((obj.shape == AnnotationShape::RotatedRect || obj.shape == AnnotationShape::Polygon) && obj.points.size() >= 3) {
        QPolygonF poly;
        for (const auto& p : obj.points)
            poly << imageToWidget(p);
        QPainterPath path;
        path.addPolygon(poly);
        if (path.contains(widgetPoint))
            return true;
        QRectF b = poly.boundingRect().adjusted(-6, -6, 6, 6);
        return b.contains(widgetPoint);
    }

    if ((obj.shape == AnnotationShape::Line || obj.shape == AnnotationShape::Point || obj.shape == AnnotationShape::Keypoints)) {
        for (const auto& p : obj.points) {
            if (QLineF(imageToWidget(p), widgetPoint).length() <= 8)
                return true;
        }
        if (obj.shape == AnnotationShape::Line && obj.points.size() >= 2) {
            QPainterPathStroker stroker;
            stroker.setWidth(12);
            QPainterPath path;
            path.moveTo(imageToWidget(obj.points[0]));
            path.lineTo(imageToWidget(obj.points[1]));
            return stroker.createStroke(path).contains(widgetPoint);
        }
    }
    return false;
}

int AnnotationCanvas::hitTestObject(const QPointF& widgetPoint) const
{
    if (!m_annotations)
        return -1;
    for (int i = m_annotations->size() - 1; i >= 0; --i) {
        if (objectContainsWidgetPoint(m_annotations->at(i), widgetPoint))
            return i;
    }
    return -1;
}

QColor AnnotationCanvas::colorForClass(int classId) const
{
    if (m_classes) {
        for (const auto& cls : *m_classes) {
            if (cls.id == classId)
                return cls.color;
        }
    }
    return QColor("#1FB6FF");
}

} // namespace trtforge

