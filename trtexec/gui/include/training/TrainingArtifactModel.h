#pragma once

#include <QString>
#include <QStringList>

namespace trtforge {

class TrainingArtifactModel
{
public:
    void reset(const QString& mode);
    QString mode() const { return m_mode; }

    QString addArtifact(const QString& kind, const QString& name, const QString& path);
    QString labelsJsonStatusText(const QString& name, const QString& path, const QString& message);
    QStringList artifacts() const { return m_artifacts; }

private:
    QString m_mode;
    QStringList m_artifacts;
    QStringList m_labelJsonPaths;
};

} // namespace trtforge
