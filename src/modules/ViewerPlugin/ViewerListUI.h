#ifndef VIEWER_LIST_UI_H
#define VIEWER_LIST_UI_H

#include <QObject>
#include <QStringList>

class IAppContext;
class QListWidget;

class ViewerListUI : public QObject {
    Q_OBJECT
public:
    explicit ViewerListUI(IAppContext* ctx, QObject* parent = nullptr);
    ~ViewerListUI() override = default;

    void setupUI();
    QListWidget* widget() const { return m_imageList; }

private slots:
    void onListRowChanged(int row);
    void onImageListUpdated(const QStringList& images, int currentIndex);
    void onImageIndexChanged(int index, int total);

private:
    IAppContext* m_ctx = nullptr;
    QListWidget* m_imageList = nullptr;
};

#endif // VIEWER_LIST_UI_H
