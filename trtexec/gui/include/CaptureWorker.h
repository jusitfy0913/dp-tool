#pragma once

#include <QObject>
#include <QImage>
#include <QMutex>
#include <QString>
#include <QVector>
#include <atomic>

namespace trtforge {

enum class SourceType { Image, Video, Camera };

struct CameraInfo {
    int     index   = 0;       // OpenCV VideoCapture 索引
    int     backend = 0;       // cv::CAP_DSHOW / cv::CAP_MSMF / cv::CAP_ANY
    QString name;              // 设备 FriendlyName（DShow 枚举得到）
    int     width   = 0;
    int     height  = 0;
    double  fps     = 0.0;
};

class CaptureWorker : public QObject
{
    Q_OBJECT
public:
    explicit CaptureWorker(QObject* parent = nullptr);
    ~CaptureWorker() override;

    // pathOrIndex：图像/视频为路径，相机为整数索引字符串
    // backend：仅相机有效；<=0 表示自动尝试
    void setSource(SourceType type, const QString& pathOrIndex, int backend = 0);

    // 探测某后端下指定索引是否可用，并回填 W/H/FPS
    static bool probeCamera(int index, int backend,
                            int* outW = nullptr, int* outH = nullptr,
                            double* outFps = nullptr);

    // 枚举系统中的视频输入设备（Windows 走 DirectShow + 同步用 MSMF 索引探测）
    // 返回的每一项都已通过对应 backend 实际打开过一次
    static QVector<CameraInfo> enumerateCameras(int maxProbe = 6);

public slots:
    void start();
    void stop();

signals:
    void frameReady(QImage frame, qint64 timestampMs);
    void finished();
    void errorOccurred(QString msg);
    void sourceInfo(int width, int height, double fps);

private:
    SourceType  m_type    = SourceType::Image;
    QString     m_path;
    int         m_backend = 0;
    std::atomic<bool> m_running{false};
};

} // namespace trtforge
