#ifndef SIGNALBUS_H
#define SIGNALBUS_H

#include <QObject>
#include <QVariant>
#include "Global.h"

class APP_EXPORT SignalBus : public QObject {
    Q_OBJECT
public:
    explicit SignalBus(QObject *parent = nullptr);

signals:
    // --- Existing signals (backward-compatible) ---
    // Data/State signals
    void stateChanged();
    void imageIndexChanged(int index, int total);
    void imageListUpdated(const QStringList& images, int currentIndex);
    void autoNavigationChanged(bool active, bool isNext);
    void languageChanged(const QString &lang);
    void userChanged(const QString &username);

    // --- Reconstruction lifecycle ---
    void reconstructionStarted();
    void reconstructionFinished(bool success, int pointCount);
    void reconstructionProgress(int percent, const QString &stage);
    void reconstructionStopped();

    // --- AI events ---
    void aiModelLoaded(const QString &modelType);
    void aiInferenceStarted(const QString &type);
    void aiInferenceFinished(const QString &type);

    // --- AI Assistant application-control bridge ---
    // The local AI server selects an action, while the Qt plugin that owns the
    // feature executes it on the GUI thread.  Keeping the signal in the core
    // avoids coupling the assistant plugin to every optional feature plugin.
    void agentUiActionRequested(const QString &action, const QVariantMap &parameters);
    void agentUiActionCompleted(const QString &requestId, bool success, const QVariantMap &result);

    // --- Scene events ---
    void sceneContentChanged();
    void sceneRendered();

    // --- App events ---
    void themeChanged(const QString &themeName);

    /**
     * Signal chung cho plugin giao tiếp với nhau.
     * @param pluginName  Tên plugin phát signal
     * @param event       Tên sự kiện (e.g. "modelLoaded", "exportDone")
     * @param data        Dữ liệu kèm theo (có thể là QVariantMap)
     */
    void pluginEvent(const QString &pluginName,
                     const QString &event,
                     const QVariant &data);
};

#endif // SIGNALBUS_H
