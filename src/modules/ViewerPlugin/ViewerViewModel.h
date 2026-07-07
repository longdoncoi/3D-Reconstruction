#ifndef VIEWERVIEWMODEL_H
#define VIEWERVIEWMODEL_H

#include <QObject>
#include <QString>
#include "IAppContext.h"

class ViewerViewModel : public QObject {
    Q_OBJECT
public:
    explicit ViewerViewModel(IAppContext* ctx, QObject* parent = nullptr);

    void load2DImage(const QString& filePath);
    void load3DModel(const QString& filePath);
    void loadDicom(const QString& directoryPath);

signals:
    void loadingStarted(const QString& message);
    void progressUpdated(int percent);
    void loadingFinished();
    void showNavigationUI(bool show);
    void errorOccurred(const QString& errorMsg);

private:
    IAppContext* m_ctx;
};

#endif // VIEWERVIEWMODEL_H
