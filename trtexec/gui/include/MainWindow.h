#pragma once

#include <QMainWindow>

class QLabel;
class QWidget;

namespace trtforge {

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override = default;

private:
    QWidget* buildHeader();
    void refreshGpuStatus();

    QLabel* m_lblGpuStatus { nullptr };
};

} // namespace trtforge
