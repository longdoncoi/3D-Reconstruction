#include "MailMimeParser.h"

#include <QtTest>

class TestMailMimeParser : public QObject {
    Q_OBJECT

private slots:
    void splitsDelimitedAddresses()
    {
        const QStringList addresses =
            MailMimeParser::splitAddresses("Alice <alice@example.com>; bob@example.com");

        QCOMPARE(addresses, QStringList({"alice@example.com", "bob@example.com"}));
    }

    void parsesPlainTextFetchedMessage()
    {
        const QString raw =
            "* 1 FETCH (FLAGS (\\Seen) BODY[] {191}\r\n"
            "From: =?UTF-8?B?QWxpY2U=?= <alice@example.com>\r\n"
            "To: bob@example.com\r\n"
            "Subject: =?UTF-8?Q?Hello=20World?=\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "Content-Transfer-Encoding: quoted-printable\r\n"
            "\r\n"
            "Line=201\r\n"
            "Line=202\r\n"
            ")\r\n";

        const MailMessage message = MailMimeParser::parseFetchedMessage("42", raw);

        QCOMPARE(message.uid, QString("42"));
        QVERIFY(message.isRead);
        QCOMPARE(message.from, QString("Alice <alice@example.com>"));
        QCOMPARE(message.subject, QString("Hello World"));
        QVERIFY(message.htmlBody.contains("Line 1"));
        QVERIFY(message.htmlBody.contains("Line 2"));
        QVERIFY(message.htmlBody.startsWith("<pre"));
    }
};

QTEST_GUILESS_MAIN(TestMailMimeParser)
#include "TestMailMimeParser.moc"
