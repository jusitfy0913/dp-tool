#include "CaptureWorker.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QThread>

#include <vector>

#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <dshow.h>
  #include <strmif.h>
  #pragma comment(lib, "strmiids.lib")
  #pragma comment(lib, "ole32.lib")
  #pragma comment(lib, "oleaut32.lib")
#endif

namespace trtforge {

namespace {

cv::Mat imreadUnicode(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};

    const QByteArray bytes = file.readAll();
    if (bytes.isEmpty()) return {};

    const std::vector<uchar> data(bytes.begin(), bytes.end());
    return cv::imdecode(data, cv::IMREAD_COLOR);
}

bool containsNonAscii(const QString& text)
{
    for (const QChar ch : text) {
        if (ch.unicode() > 127) return true;
    }
    return false;
}

QString makeAsciiVideoCopy(const QString& source)
{
    const QFileInfo info(source);
    const QString suffix = info.suffix().isEmpty() ? QStringLiteral("video") : info.suffix();
    const QString target = QDir::temp().absoluteFilePath(
        QStringLiteral("trtforge_video_%1.%2").arg(QDateTime::currentMSecsSinceEpoch()).arg(suffix));
    QFile::remove(target);
    return QFile::copy(source, target) ? target : QString();
}

bool openVideoUnicode(cv::VideoCapture& cap, const QString& path, QString* tempCopyPath = nullptr)
{
    if (!containsNonAscii(path)) {
        cap.open(path.toStdString());
        if (cap.isOpened()) return true;
    }

    const QString copied = makeAsciiVideoCopy(path);
    if (copied.isEmpty()) return false;

    cap.open(copied.toStdString());
    if (!cap.isOpened()) {
        QFile::remove(copied);
        return false;
    }

    if (tempCopyPath) *tempCopyPath = copied;
    return true;
}

} // namespace

CaptureWorker::CaptureWorker(QObject* parent) : QObject(parent) {}

CaptureWorker::~CaptureWorker() { m_running = false; }

void CaptureWorker::setSource(SourceType type, const QString& pathOrIndex, int backend)
{
    m_type    = type;
    m_path    = pathOrIndex;
    m_backend = backend;
}

bool CaptureWorker::probeCamera(int index, int backend,
                                int* outW, int* outH, double* outFps)
{
    // 包一层异常保护：MSMF 后端遇到无效索引可能直接抛 SEH/std::exception
    try {
        cv::VideoCapture cap;
        cap.open(index, backend > 0 ? backend : cv::CAP_ANY);
        if (!cap.isOpened()) return false;

        cv::Mat test;
        bool got = false;
        // 部分 USB 摄像头首帧需要 200~500ms，每次失败 sleep 60ms 再试，最多 ~1.2s
        for (int k = 0; k < 20; ++k) {
            if (cap.read(test) && !test.empty()) { got = true; break; }
            QThread::msleep(60);
        }
        if (!got) { cap.release(); return false; }

        if (outW)   *outW   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        if (outH)   *outH   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (outFps) *outFps = cap.get(cv::CAP_PROP_FPS);
        cap.release();
        return true;
    } catch (...) {
        return false;
    }
}

#ifdef _WIN32
// 通过 DirectShow 枚举系统中的视频输入设备 FriendlyName
// 顺序与 OpenCV CAP_DSHOW 索引一致（0,1,2,...）
static QVector<QString> enumerateDShowNames()
{
    QVector<QString> names;
    HRESULT hrInit = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    // 仅当我们真正完成首次初始化（S_OK）时才在末尾 Uninit；
    // S_FALSE 表示已经被别处（Qt）初始化过，绝不能 Uninit，否则会破坏 Qt 的 COM 计数。
    bool needUninit = (hrInit == S_OK);

    ICreateDevEnum* pDevEnum = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_ICreateDevEnum, reinterpret_cast<void**>(&pDevEnum));
    if (SUCCEEDED(hr) && pDevEnum) {
        IEnumMoniker* pEnum = nullptr;
        hr = pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnum, 0);
        if (hr == S_OK && pEnum) {
            IMoniker* pMoniker = nullptr;
            while (pEnum->Next(1, &pMoniker, nullptr) == S_OK) {
                IPropertyBag* pBag = nullptr;
                hr = pMoniker->BindToStorage(nullptr, nullptr, IID_IPropertyBag,
                                             reinterpret_cast<void**>(&pBag));
                QString name;
                if (SUCCEEDED(hr) && pBag) {
                    VARIANT v; VariantInit(&v);
                    hr = pBag->Read(L"FriendlyName", &v, nullptr);
                    if (SUCCEEDED(hr) && v.vt == VT_BSTR) {
                        name = QString::fromWCharArray(v.bstrVal);
                    }
                    VariantClear(&v);
                    pBag->Release();
                }
                if (name.isEmpty()) name = QStringLiteral("Unknown Camera");
                names.push_back(name);
                pMoniker->Release();
            }
            pEnum->Release();
        }
        pDevEnum->Release();
    }

    if (needUninit) CoUninitialize();
    return names;
}
#endif

QVector<CameraInfo> CaptureWorker::enumerateCameras(int maxProbe)
{
    QVector<CameraInfo> result;

#ifdef _WIN32
    // Windows 上以 DirectShow 枚举为基线（能拿到真实 FriendlyName）；
    // 对每个真实存在的设备，先试 DSHOW，再用 MSMF 兜底
    // —— MSMF 盲探不存在的索引会 crash，但对"已知存在"的索引调用是安全的。
    QVector<QString> dsNames = enumerateDShowNames();
    int n = qMin<int>(dsNames.size(), maxProbe);
    for (int i = 0; i < n; ++i) {
        // 先试 DSHOW
        CameraInfo dsInfo;
        dsInfo.index   = i;
        dsInfo.backend = cv::CAP_DSHOW;
        dsInfo.name    = dsNames[i];
        bool dsOk = probeCamera(i, cv::CAP_DSHOW,
                                &dsInfo.width, &dsInfo.height, &dsInfo.fps);
        if (dsOk) {
            result.push_back(dsInfo);
        }

        // 再试 MSMF —— 对已知存在的索引调用是安全的
        CameraInfo msInfo;
        msInfo.index   = i;
        msInfo.backend = cv::CAP_MSMF;
        msInfo.name    = dsNames[i];
        bool msOk = probeCamera(i, cv::CAP_MSMF,
                                &msInfo.width, &msInfo.height, &msInfo.fps);
        if (msOk) {
            result.push_back(msInfo);
        }

        // DSHOW 和 MSMF 都失败：仍然列一条让用户知情
        if (!dsOk && !msOk) {
            CameraInfo failInfo;
            failInfo.index   = i;
            failInfo.backend = cv::CAP_DSHOW;
            failInfo.name    = dsNames[i] + QStringLiteral("  (被占用/无法打开)");
            result.push_back(failInfo);
        }
    }
#else
    for (int i = 0; i < maxProbe; ++i) {
        CameraInfo info;
        info.index   = i;
        info.backend = cv::CAP_ANY;
        if (probeCamera(i, cv::CAP_ANY, &info.width, &info.height, &info.fps)) {
            info.name = QStringLiteral("Camera #%1").arg(i);
            result.push_back(info);
        }
    }
#endif
    return result;
}

void CaptureWorker::start()
{
    m_running = true;

    if (m_type == SourceType::Image) {
        cv::Mat img = imreadUnicode(m_path);
        if (img.empty()) {
            emit errorOccurred(QStringLiteral("无法读取图像: %1").arg(m_path));
            emit finished();
            return;
        }
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        QImage qi(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888);
        emit sourceInfo(img.cols, img.rows, 0);
        emit frameReady(qi.copy(), QDateTime::currentMSecsSinceEpoch());
        emit finished();
        return;
    }

    cv::VideoCapture cap;
    QString tempVideoPath;
    if (m_type == SourceType::Camera) {
        int idx = m_path.toInt();
        if (m_backend > 0) {
            cap.open(idx, m_backend);
        } else {
            // 自动按优先级尝试
            const int backends[] = { cv::CAP_DSHOW, cv::CAP_MSMF, cv::CAP_ANY };
            for (int be : backends) {
                cap.open(idx, be);
                if (cap.isOpened()) break;
            }
        }
        if (cap.isOpened()) {
            // 热身读几帧，避免首帧黑屏
            cv::Mat warm;
            for (int k = 0; k < 3; ++k) { if (cap.read(warm) && !warm.empty()) break; }
        }
    } else {
        openVideoUnicode(cap, m_path, &tempVideoPath);
    }

    if (!cap.isOpened()) {
        QString msg = (m_type == SourceType::Camera)
            ? QStringLiteral("无法打开相机 #%1").arg(m_path)
            : QStringLiteral("无法打开视频: %1").arg(m_path);
        emit errorOccurred(msg);
        emit finished();
        return;
    }

    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    emit sourceInfo(w, h, fps);

    int delayMs = static_cast<int>(1000.0 / fps);

    cv::Mat frame, rgb;
    while (m_running) {
        if (!cap.read(frame) || frame.empty()) {
            if (m_type == SourceType::Video) break;
            QThread::msleep(10);
            continue;
        }
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        QImage qi(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888);
        emit frameReady(qi.copy(), QDateTime::currentMSecsSinceEpoch());

        if (m_type == SourceType::Video)
            QThread::msleep(delayMs > 1 ? delayMs - 1 : 1);
    }

    cap.release();
    if (!tempVideoPath.isEmpty()) QFile::remove(tempVideoPath);
    emit finished();
}

void CaptureWorker::stop()
{
    m_running = false;
}

} // namespace trtforge
