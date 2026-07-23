#pragma once

// ─── Qt Core ─────────────────────────────────────────────────────────────────
#include <QAction>
#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QList>
#include <QMap>
#include <QMenu>
#include <QObject>
#include <QPointer>
#include <QPushButton>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QWidget>

// ─── Qt Widgets (frequently used across plugins) ─────────────────────────────
#include <QMainWindow>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QStackedWidget>
#include <QDockWidget>
#include <QFrame>
#include <QSplitter>
#include <QToolButton>
#include <QComboBox>
#include <QLineEdit>
#include <QTextEdit>
#include <QScrollArea>
#include <QGroupBox>

// ─── Qt Network / JSON (used by AI, Mail, etc.) ─────────────────────────────
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonValue>

// ─── Qt GUI ──────────────────────────────────────────────────────────────────
#include <QPixmap>
#include <QIcon>
#include <QFont>
#include <QColor>
#include <QPainter>
#include <QMouseEvent>

// ─── STL ─────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ─── OpenCV ──────────────────────────────────────────────────────────────────
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ─── PCL ─────────────────────────────────────────────────────────────────────
// Note: Including core PCL headers here prevents the compiler from re-parsing 
// massive Eigen/Boost templates for every single .cpp file.
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

// ─── VTK ─────────────────────────────────────────────────────────────────────
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
