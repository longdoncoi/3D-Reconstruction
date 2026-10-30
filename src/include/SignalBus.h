#ifndef SIGNALBUS_H
#define SIGNALBUS_H

#include <QObject>
#include <QVariant>
#include "Global.h"

class APP_EXPORT SignalBus : public QObject {
    Q_OBJECT
public:
    explicit SignalBus(QObject *parent = nullptr);

    // Ownership convention (trusted first-party plugins — NOT a security
    // boundary). Signals that mirror AppCore state (image list/index,
    // auto-navigation, language, scene) are emitted by the service that owns
    // that state; feature signals (user, reconstruction, AI, agent actions)
    // are emitted by the feature plugin that owns the feature, and every other
    // plugin only subscribes. `stateChanged` is the generic "settings/state
    // changed" pulse after a settings write. The bus stays concrete and
    // mutable on purpose: splitting it into emitter/observer interfaces would
    // add an ABI surface for a convention the plugin layer already follows,
    // and CheckAgentActions enforces the same one-owner rule for actions.

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
};

#endif // SIGNALBUS_H
