#ifndef AGENTACTIONMANIFEST_H
#define AGENTACTIONMANIFEST_H

#include <QString>
#include <QVariantMap>
#include "Global.h"

// Loads Config/agent_action_manifest.json, the same source of truth consumed
// by the Python AI server.  This prevents Qt from dispatching stale aliases.
class APP_EXPORT AgentActionManifest final {
public:
    static bool canonicalize(QString &action, QVariantMap &parameters, QString *error = nullptr);
};

#endif // AGENTACTIONMANIFEST_H
