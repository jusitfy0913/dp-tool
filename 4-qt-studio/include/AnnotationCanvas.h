#pragma once

#include "AnnotationProject.h"

#include <QImage>
#include <QWidget>

namespace trtforge {

enum class AnnotationTool {
    Select,
    Pan,
    Rectangle,
    RotatedRect,
    Polygon,
    Line,
    Point,
    Pose
};

class AnnotationCanvas : public QWidget
{
    Q_OBJECT
public:
    explicit AnnotationCanvas(QWidget* parent = nullptr);

    void setTool(AnnotationTool tool);
    AnnotationTool tool() const { return m_tool; }
    void setImage(const QString& path);
    void setAnnotations(QVector<AnnotationObject>* annotations);
    void setClasses(const QVector<AnnotationClass>* classes);
    void setSelectedObject(int index);
    int selectedObject() const { return m_selectedObject; }

signals:
    void draftCompleted(trtforge::AnnotationObject object);
    void statusChanged(QString text);
    void objectSelected(int index);
    void editStarted();
    void annotationsEdited();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    QPointF imageToWidget(const QPointF& p) const;
    QPointF widgetToImage(const QPointF& p) const;
    QRectF imageRectOnWidget() const;
    void finishCurrentDraft();
    bool hasObbDraft() const;
    bool hasRectangleDraft() const;
    bool hasAdjustableBoxDraft() const;
    bool isDraftLargeEnough() const;
    QPointF draftCenterImage() const;
    QPointF obbCenterImage() const;
    QPointF obbRotateHandleWidget() const;
    QPointF obbRightMidWidget() const;
    void drawObbControls(QPainter& painter) const;
    void drawRectangleControls(QPainter& painter) const;
    void clearDraft();
    void drawObject(QPainter& painter, const AnnotationObject& obj, bool draft) const;
    void drawObjectLabel(QPainter& painter, const AnnotationObject& obj, const QColor& color) const;
    QRectF objectBoundsOnWidget(const AnnotationObject& obj) const;
    QString labelTextForObject(const AnnotationObject& obj) const;
    bool objectContainsWidgetPoint(const AnnotationObject& obj, const QPointF& widgetPoint) const;
    int hitTestObject(const QPointF& widgetPoint) const;
    QColor colorForClass(int classId) const;

    AnnotationTool m_tool = AnnotationTool::Rectangle;
    QImage m_image;
    QString m_imagePath;
    QVector<AnnotationObject>* m_annotations = nullptr;
    const QVector<AnnotationClass>* m_classes = nullptr;
    int m_selectedObject = -1;

    double m_zoom = 1.0;
    QPointF m_pan;
    bool m_dragging = false;
    bool m_movingObject = false;
    QPointF m_startImage;
    QPointF m_lastImage;
    QPointF m_lastWidget;
    AnnotationObject m_draft;
    enum class ObbEditMode { None, Drawing, Moving, Rotating };
    ObbEditMode m_obbEditMode = ObbEditMode::None;
    QPointF m_obbDragStartImage;
    QVector<QPointF> m_obbDragStartPoints;
    QVector<QPointF> m_polyPoints;
    QVector<QPointF> m_posePoints;
};

} // namespace trtforge

