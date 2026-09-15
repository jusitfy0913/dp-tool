#include "MainWindow.h"
#include "BuildOptions.h"
#include "InferWorker.h"

#include <QApplication>
#include <QCoreApplication>
#include <QFile>
#include <QIcon>
#include <QImage>
#include <QDir>
#include <QStringList>

namespace {

void prependRuntimeDllPaths()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QString paddleRoot = QDir(appDir).filePath("paddle_inference");
    const QStringList runtimeDirs = {
        appDir,
        QDir(paddleRoot).filePath("paddle/lib"),
        QDir(paddleRoot).filePath("third_party/install/mklml/lib"),
        QDir(paddleRoot).filePath("third_party/install/onednn/lib"),
        QDir(paddleRoot).filePath("third_party/install/glog/lib"),
        QDir(paddleRoot).filePath("third_party/install/gflags/lib"),
        QDir(paddleRoot).filePath("third_party/install/protobuf/lib"),
        QDir(paddleRoot).filePath("third_party/install/yaml-cpp/lib")
    };

    QByteArray path = qgetenv("PATH");
    for (auto it = runtimeDirs.crbegin(); it != runtimeDirs.crend(); ++it) {
        const QString native = QDir::toNativeSeparators(*it);
        if (QDir(native).exists()) {
            path.prepend(';');
            path.prepend(native.toLocal8Bit());
        }
    }
    qputenv("PATH", path);
}

} // namespace

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    prependRuntimeDllPaths();
    QApplication::setApplicationName("DefectVision AI Studio");
    QApplication::setOrganizationName("DeepLearning");
    QApplication::setApplicationDisplayName("DefectVision AI Studio - Industrial AI Defect Inspection System");

    qRegisterMetaType<trtforge::BuildOptions>("trtforge::BuildOptions");
    qRegisterMetaType<trtforge::TestReport>("trtforge::TestReport");
    qRegisterMetaType<QVector<trtforge::Detection>>("QVector<trtforge::Detection>");
    qRegisterMetaType<QImage>("QImage");

    QFile qss(":/industrial_dark.qss");
    if (qss.open(QIODevice::ReadOnly | QIODevice::Text)) {
        app.setStyleSheet(QString::fromUtf8(qss.readAll()));
        qss.close();
    }

    trtforge::MainWindow w;
    w.show();
    return app.exec();
}
