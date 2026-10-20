#ifndef LICENSE_KEY_LEDGER_H
#define LICENSE_KEY_LEDGER_H

#include <QMap>
#include <QString>
#include <QStringList>

#include "Global.h"

class QSettings;

/**
 * @brief Persistent inventory of issued license keys and their owners.
 *
 * The ledger only knows keys and ownership; user/licence policy stays in
 * UserManager. It is deliberately not thread-safe: callers already hold the
 * UserManager mutex that serialises user-database and key-database writes.
 */
class APP_EXPORT LicenseKeyLedger {
public:
    void attachSettings(QSettings *settings) { m_settings = settings; }

    void clear();
    void load();
    void save();

    /// Generate a new unused key and add it to the in-memory inventory.
    QString generate();

    QStringList available() const;
    bool contains(const QString &key) const;
    bool isUsed(const QString &key) const;
    QString owner(const QString &key) const;

    /// Mark an unused key as consumed by @p username.
    bool claim(const QString &key, const QString &username);

    /// Remove a key and return its previous owner (empty when unassigned).
    bool release(const QString &key, QString &owner);

private:
    QSettings *m_settings = nullptr;
    QMap<QString, bool> m_keys;     // key -> isUsed
    QMap<QString, QString> m_owners; // key -> username
};

#endif // LICENSE_KEY_LEDGER_H
