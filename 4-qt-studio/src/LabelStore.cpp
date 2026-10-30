#include "LabelStore.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <cmath>

namespace trtforge {

LabelStore::LabelStore(QObject* parent) : QObject(parent) {}

bool LabelStore::load(const QString& jsonPath)
{
    QFile f(jsonPath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    QJsonParseError err;
    auto doc = QJsonDocument::fromJson(f.readAll(), &err);
    f.close();
    if (err.error != QJsonParseError::NoError)
        return false;

    auto obj = doc.object();
    m_name = obj.value("name").toString("unknown");
    m_path = jsonPath;

    m_classes.clear();
    auto arr = obj.value("classes").toArray();
    for (const auto& v : arr)
        m_classes.append(v.toString());

    m_colors.clear();
    auto cArr = obj.value("colors").toArray();
    if (cArr.size() >= m_classes.size()) {
        for (int i = 0; i < m_classes.size(); ++i) {
            auto c = cArr[i].toArray();
            if (c.size() >= 3)
                m_colors.append(QColor(c[0].toInt(), c[1].toInt(), c[2].toInt()));
            else
                m_colors.append(Qt::white);
        }
    } else {
        generateColors(m_classes.size());
    }

    emit loaded(m_classes.size(), m_name);
    return true;
}

QString LabelStore::className(int id) const
{
    if (id >= 0 && id < m_classes.size())
        return m_classes[id];
    return QStringLiteral("class_%1").arg(id);
}

QColor LabelStore::classColor(int id) const
{
    if (id >= 0 && id < m_colors.size())
        return m_colors[id];
    return QColor::fromHsvF(std::fmod(id * 0.618033988749895, 1.0), 0.85, 0.95);
}

void LabelStore::generateColors(int n)
{
    m_colors.clear();
    m_colors.reserve(n);
    for (int i = 0; i < n; ++i) {
        double hue = std::fmod(i * 0.618033988749895, 1.0);
        m_colors.append(QColor::fromHsvF(hue, 0.85, 0.95));
    }
}

} // namespace trtforge
