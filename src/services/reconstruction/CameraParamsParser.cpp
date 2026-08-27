#include "CameraParamsParser.h"

#include <QDebug>
#include <QFile>
#include <QIODevice>
#include <QRegularExpression>
#include <QTextStream>

namespace {

bool hasCountHeader(const QStringList &lines, int &startLine)
{
    bool ok = false;
    const int count = lines[0].toInt(&ok);
    if (ok && count > 0 && count < lines.size()) {
        startLine = 1;
        return true;
    }

    startLine = 0;
    return false;
}

int findImageNameToken(const QStringList &tokens)
{
    for (int tokenIndex = 0; tokenIndex < tokens.size(); ++tokenIndex) {
        const QString &token = tokens[tokenIndex];
        if (token.contains('.') && !token.startsWith('-') && !token.startsWith('+') &&
            tokens.size() - tokenIndex - 1 >= 21) {
            return tokenIndex;
        }
    }

    return -1;
}

CameraParams parseCameraParams(const QStringList &tokens, int nameIndex)
{
    CameraParams params;
    params.imageName = tokens[nameIndex];
    params.K = cv::Mat(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            params.K.at<double>(row, col) = tokens[nameIndex + 1 + row * 3 + col].toDouble();
        }
    }

    params.R = cv::Mat(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            params.R.at<double>(row, col) = tokens[nameIndex + 10 + row * 3 + col].toDouble();
        }
    }

    params.t = cv::Mat(3, 1, CV_64F);
    for (int row = 0; row < 3; ++row) {
        params.t.at<double>(row, 0) = tokens[nameIndex + 19 + row].toDouble();
    }

    cv::Mat rotationTranslation;
    cv::hconcat(params.R, params.t, rotationTranslation);
    params.P = params.K * rotationTranslation;
    return params;
}

}

namespace CameraParamsParser {

bool loadFromFile(const QString &paramsFilePath, std::vector<CameraParams> &cameraParams)
{
    QFile file(paramsFilePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Cannot open params file:" << paramsFilePath;
        return false;
    }

    cameraParams.clear();
    QTextStream input(&file);
    QStringList lines;
    while (!input.atEnd()) {
        const QString line = input.readLine().trimmed();
        if (!line.isEmpty()) lines.append(line);
    }
    file.close();

    if (lines.isEmpty()) {
        qWarning() << "Params file is empty.";
        return false;
    }

    int startLine = 0;
    const bool isFormatA = hasCountHeader(lines, startLine);
    qDebug() << "loadCameraParams: format ="
             << (isFormatA ? "A (count header)" : "B (Middlebury/no header)")
             << " startLine=" << startLine;

    for (int lineIndex = startLine; lineIndex < lines.size(); ++lineIndex) {
        const QStringList tokens = lines[lineIndex].split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        const int nameIndex = findImageNameToken(tokens);
        if (nameIndex < 0) {
            qWarning() << "Skipping line" << lineIndex << "- imageName not found:" << lines[lineIndex].left(60);
            continue;
        }

        cameraParams.push_back(parseCameraParams(tokens, nameIndex));
    }

    qDebug() << "Loaded" << cameraParams.size() << "camera params";
    return !cameraParams.empty();
}

}
