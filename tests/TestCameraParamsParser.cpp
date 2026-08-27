#include "CameraParamsParser.h"

#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QtTest>

class TestCameraParamsParser : public QObject {
    Q_OBJECT

private slots:
    void parsesFormatAFile()
    {
        const QString path = QDir::temp().filePath("camera_params_format_a.txt");
        QFile file(path);
        QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
        QTextStream out(&file);
        out << "1\n";
        out << "image001.jpg "
            << "1000 0 320 0 1000 240 0 0 1 "
            << "1 0 0 0 1 0 0 0 1 "
            << "1 2 3\n";
        file.close();

        std::vector<CameraParams> params;
        QVERIFY(CameraParamsParser::loadFromFile(path, params));
        QCOMPARE(params.size(), size_t(1));
        QCOMPARE(params[0].imageName, QString("image001.jpg"));
        QCOMPARE(params[0].K.at<double>(0, 0), 1000.0);
        QCOMPARE(params[0].t.at<double>(2, 0), 3.0);
        QCOMPARE(params[0].P.rows, 3);
        QCOMPARE(params[0].P.cols, 4);

        QFile::remove(path);
    }
};

QTEST_GUILESS_MAIN(TestCameraParamsParser)
#include "TestCameraParamsParser.moc"
