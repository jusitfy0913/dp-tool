#include "training/TrainingArtifactModel.h"

namespace trtforge {

void TrainingArtifactModel::reset(const QString& mode)
{
    m_mode = mode;
    m_artifacts.clear();
    m_labelJsonPaths.clear();
}

QString TrainingArtifactModel::addArtifact(const QString& kind, const QString& name, const QString& path)
{
    QString line = kind;
    if (!name.isEmpty()) {
        line += "/" + name;
    }
    if (!path.isEmpty()) {
        line += ": " + path;
    }
    if (!line.trimmed().isEmpty() && !m_artifacts.contains(line)) {
        m_artifacts << line;
    }
    return line;
}

QString TrainingArtifactModel::labelsJsonStatusText(const QString& name, const QString& path, const QString& message)
{
    if (!path.isEmpty()) {
        const QString labelLine = name.isEmpty() ? path : QString("%1\n%2").arg(name, path);
        if (!m_labelJsonPaths.contains(labelLine)) {
            m_labelJsonPaths << labelLine;
        }
        return QStringLiteral("标签 JSON：生成成功\n") + m_labelJsonPaths.join('\n');
    }
    if (!message.isEmpty()) {
        return QStringLiteral("标签 JSON：未生成\n") + message;
    }
    return QStringLiteral("标签 JSON：未生成，请检查 dataset.yaml 的 names 字段。");
}

} // namespace trtforge
