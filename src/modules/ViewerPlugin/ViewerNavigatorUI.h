#ifndef VIEWERNAVIGATORUI_H
#define VIEWERNAVIGATORUI_H

#include <QObject>

class QWidget;
class QPushButton;
class IAppContext;

class ViewerNavigatorUI : public QObject {
    Q_OBJECT
public:
    explicit ViewerNavigatorUI(IAppContext* ctx, QObject* parent = nullptr);
    
    QPushButton* btnPrev() const { return m_btnPrev; }
    QPushButton* btnAutoPrev() const { return m_btnAutoPrev; }
    QPushButton* btnAutoNext() const { return m_btnAutoNext; }
    QPushButton* btnNext() const { return m_btnNext; }
    QWidget* widget() const { return m_widget; }

private:
    IAppContext* m_ctx = nullptr;
    QWidget* m_widget = nullptr;
    QPushButton* m_btnPrev;
    QPushButton* m_btnNext;
    QPushButton* m_btnAutoPrev;
    QPushButton* m_btnAutoNext;
};

#endif // VIEWERNAVIGATORUI_H
