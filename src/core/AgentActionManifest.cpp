#include "AgentActionManifest.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include "AppConfig.h"

bool AgentActionManifest::canonicalize(QString &action, QVariantMap &parameters, QString *error) {
    const QString path = AppConfig::instance().projectRootDir() + "/Config/agent_action_manifest.json";
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = "Unable to load agent action manifest";
        return false;
    }
    const QJsonArray actions = QJsonDocument::fromJson(file.readAll()).object().value("actions").toArray();
    QJsonObject definition;
    for (const QJsonValue &value : actions) {
        const QJsonObject candidate = value.toObject();
        bool isAlias = false;
        for (const QJsonValue &alias : candidate.value("aliases").toArray()) {
            isAlias |= alias.toString() == action;
        }
        if (candidate.value("id").toString() == action || isAlias) {
            definition = candidate;
            action = candidate.value("id").toString();
            break;
        }
    }
    if (definition.isEmpty()) {
        if (error) *error = "Unsupported agent action: " + action;
        return false;
    }
    const QJsonObject fields = definition.value("parameters").toObject();
    for (auto it = fields.begin(); it != fields.end(); ++it) {
        const QJsonObject field = it.value().toObject();
        const QVariant value = parameters.value(it.key());
        if (field.value("required").toBool() && (!value.isValid() || value.toString().isEmpty())) {
            if (error) *error = action + " requires " + it.key();
            return false;
        }
        const QJsonArray allowed = field.value("enum").toArray();
        if (value.isValid() && !allowed.isEmpty()) {
            bool matches = false;
            for (const QJsonValue &allowedValue : allowed) matches |= allowedValue.toString() == value.toString();
            if (!matches) {
                if (error) *error = action + "." + it.key() + " has an invalid value";
                return false;
            }
        }
    }
    parameters["action"] = action;
    return true;
}
