#include <QTest>
#include <QSignalSpy>
#include <QVariantMap>
#include "SignalBus.h"

class TestAgentActions : public QObject {
    Q_OBJECT

private slots:
    void testAuthActions() {
        SignalBus bus;
        QSignalSpy spy(&bus, &SignalBus::agentUiActionRequested);

        QVariantMap params;
        params["username"] = "Admin";
        params["password"] = "1";
        emit bus.agentUiActionRequested("admin.login", params);

        QCOMPARE(spy.count(), 1);
        QList<QVariant> args = spy.takeFirst();
        QCOMPARE(args.at(0).toString(), QString("admin.login"));
        QCOMPARE(args.at(1).toMap()["username"].toString(), QString("Admin"));
        QCOMPARE(args.at(1).toMap()["password"].toString(), QString("1"));

        emit bus.agentUiActionRequested("admin.logout", QVariantMap());
        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy.takeFirst().at(0).toString(), QString("admin.logout"));
    }

    void testAssistantActions() {
        SignalBus bus;
        QSignalSpy spy(&bus, &SignalBus::agentUiActionRequested);
        
        QStringList actions = {
            "assistant.open", "assistant.close",
            "assistant.reload_model", "assistant.reload_rag",
            "assistant.reload_agent", "assistant.reload_server"
        };
        
        for (const QString& action : actions) {
            emit bus.agentUiActionRequested(action, QVariantMap());
            QCOMPARE(spy.count(), 1);
            QCOMPARE(spy.takeFirst().at(0).toString(), action);
        }
    }

    void testMailActions() {
        SignalBus bus;
        QSignalSpy spy(&bus, &SignalBus::agentUiActionRequested);
        
        QStringList actions = { "mail.open", "mail.close", "mail.settings" };
        for (const QString& action : actions) {
            emit bus.agentUiActionRequested(action, QVariantMap());
            QCOMPARE(spy.count(), 1);
            QCOMPARE(spy.takeFirst().at(0).toString(), action);
        }
    }
    
    void testAIActions() {
        SignalBus bus;
        QSignalSpy spy(&bus, &SignalBus::agentUiActionRequested);
        
        QStringList actions = {
            "ai.run_detection", "ai.run_segmentation", "ai.video_tracking",
            "ai.hide_results", "ai.training_model", "ai.view_training_charts"
        };
        
        for (const QString& action : actions) {
            emit bus.agentUiActionRequested(action, QVariantMap());
            QCOMPARE(spy.count(), 1);
            QCOMPARE(spy.takeFirst().at(0).toString(), action);
        }
    }

    void testReconstructionActions() {
        SignalBus bus;
        QSignalSpy spy(&bus, &SignalBus::agentUiActionRequested);
        
        QStringList actions = {
            "reconstruction.load_images", "reconstruction.start_reconstruction",
            "reconstruction.view_3d_model", "reconstruction.close_3d_model"
        };
        
        for (const QString& action : actions) {
            emit bus.agentUiActionRequested(action, QVariantMap());
            QCOMPARE(spy.count(), 1);
            QCOMPARE(spy.takeFirst().at(0).toString(), action);
        }
    }

    void testViewerActions() {
        SignalBus bus;
        QSignalSpy spy(&bus, &SignalBus::agentUiActionRequested);
        
        QStringList actions = {
            "viewer.load_2d", "viewer.load_3d", "viewer.load_dicom"
        };
        
        for (const QString& action : actions) {
            emit bus.agentUiActionRequested(action, QVariantMap());
            QCOMPARE(spy.count(), 1);
            QCOMPARE(spy.takeFirst().at(0).toString(), action);
        }
    }
    
    void testGeneralActions() {
        SignalBus bus;
        QSignalSpy spy(&bus, &SignalBus::agentUiActionRequested);
        
        QVariantMap langParams;
        langParams["language"] = "vi";
        emit bus.agentUiActionRequested("language.change", langParams);
        QCOMPARE(spy.count(), 1);
        QList<QVariant> args = spy.takeFirst();
        QCOMPARE(args.at(0).toString(), QString("language.change"));
        QCOMPARE(args.at(1).toMap()["language"].toString(), QString("vi"));
        
        emit bus.agentUiActionRequested("help.about", QVariantMap());
        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy.takeFirst().at(0).toString(), QString("help.about"));
    }

    void testActionAcknowledgement() {
        SignalBus bus;
        QSignalSpy spy(&bus, &SignalBus::agentUiActionCompleted);

        QVariantMap result;
        result["action"] = "mail.open";
        emit bus.agentUiActionCompleted("request-42", true, result);

        QCOMPARE(spy.count(), 1);
        const QList<QVariant> args = spy.takeFirst();
        QCOMPARE(args.at(0).toString(), QString("request-42"));
        QCOMPARE(args.at(1).toBool(), true);
        QCOMPARE(args.at(2).toMap().value("action").toString(), QString("mail.open"));
    }
};

QTEST_MAIN(TestAgentActions)
#include "TestAgentActions.moc"
