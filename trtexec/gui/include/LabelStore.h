#pragma once

#include <QColor>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace trtforge {

class LabelStore : public QObject
{
    Q_OBJECT
public:
    explicit LabelStore(QObject* parent = nullptr);

    bool load(const QString& jsonPath);

    int          count()                       const { return m_classes.size(); }
    QString      className(int id)             const;
    QColor       classColor(int id)            const;
    QStringList  allClasses()                  const { return m_classes; }
    QString      datasetName()                 const { return m_name; }
    QString      filePath()                    const { return m_path; }

signals:
    void loaded(int classCount, const QString& name);

private:
    void generateColors(int n);

    QString      m_name;
    QString      m_path;
    QStringList  m_classes;
    QVector<QColor> m_colors;
};

} // namespace trtforge
