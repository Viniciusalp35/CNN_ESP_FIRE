#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QProgressBar>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QTimer>
#include <QJsonDocument>
#include <QJsonObject>
#include <QGraphicsDropShadowEffect>
#include <QThread>
#include <QPixmap>
#include <QDateTime>
#include <QFontDatabase>

// Configuração da URL do ESP32-CAM
const QString ESP_URL = "http://192.168.2.30"; 

// Worker de Monitoramento em Segundo Plano
class MonitorWorker : public QObject {
    Q_OBJECT
public:
    MonitorWorker() { manager = new QNetworkAccessManager(this); }
public slots:
    void startLoop() {
        QTimer* timer = new QTimer(this);
        connect(timer, &QTimer::timeout, this, &MonitorWorker::fetchData);
        timer->start(100); 
    }
    void fetchData() {
        QNetworkRequest reqStatus{QUrl(ESP_URL + "/status")};
        QNetworkReply* replyStatus = manager->get(reqStatus);
        connect(replyStatus, &QNetworkReply::finished, [this, replyStatus]() {
            if (replyStatus->error() == QNetworkReply::NoError) {
                QByteArray data = replyStatus->readAll();
                QJsonDocument doc = QJsonDocument::fromJson(data);
                emit statusReceived(doc.object());
            }
            replyStatus->deleteLater();
        });

        QString imgUrl = ESP_URL + "/capture?t=" + QString::number(QDateTime::currentMSecsSinceEpoch());
        QNetworkRequest reqImg{QUrl(imgUrl)};
        QNetworkReply* replyImg = manager->get(reqImg);
        connect(replyImg, &QNetworkReply::finished, [this, replyImg]() {
            if (replyImg->error() == QNetworkReply::NoError) {
                QByteArray data = replyImg->readAll();
                QPixmap pixmap;
                pixmap.loadFromData(data);
                if (!pixmap.isNull()) emit imageReceived(pixmap);
            }
            replyImg->deleteLater();
        });
    }
signals:
    void statusReceived(QJsonObject status);
    void imageReceived(QPixmap image);
private:
    QNetworkAccessManager* manager;
};

// Janela Principal da Aplicação
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow() {
        setupUI();
        startWorker();
        
        // Timer para relógio
        QTimer *clockTimer = new QTimer(this);
        connect(clockTimer, &QTimer::timeout, this, &MainWindow::updateClock);
        clockTimer->start(1000);
    }

private slots:
    void updateClock() {
        lblTime->setText(QDateTime::currentDateTime().toString("HH:mm:ss  |  dd/MM/yyyy"));
    }

    void updateStatus(QJsonObject status) {
        bool fire = status["fire"].toBool();
        double score = status["score"].toDouble(); // 0 a 100

        // Atualiza Barra de Ameaça
        threatBar->setValue((int)score);

        if (fire) {
            // FOGO DETECTADO (Visual Vermelho/Alerta)
            lblStatusMain->setText("PERIGO: FOGO");
            lblStatusMain->setStyleSheet("color: #ff5252; font-size: 24px; font-weight: bold;");
            
            frameVideo->setStyleSheet("QFrame { border: 4px solid #ff1744; border-radius: 8px; background: #000; }");
            
            // Estilo da barra (Vermelho)
            threatBar->setStyleSheet(
                "QProgressBar { border: 1px solid #444; border-radius: 5px; background: #222; text-align: center; color: white; }"
                "QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d32f2f, stop:1 #ff5252); border-radius: 4px; }"
            );
            
            lblDetails->setText(QString("CONFIDENCE: %1%").arg(score, 0, 'f', 1));
            
        } else {
            // SEGURO (Visual Azul Tecnológico/Cyan)
            lblStatusMain->setText("SISTEMA SEGURO");
            lblStatusMain->setStyleSheet("color: #00e5ff; font-size: 24px; font-weight: bold;");
            
            frameVideo->setStyleSheet("QFrame { border: 2px solid #00e5ff; border-radius: 8px; background: #000; }");
            
            // Estilo da barra (Azul/Verde)
            threatBar->setStyleSheet(
                "QProgressBar { border: 1px solid #444; border-radius: 5px; background: #222; text-align: center; color: white; }"
                "QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00695c, stop:1 #00e5ff); border-radius: 4px; }"
            );

            lblDetails->setText(QString("NÍVEL DE AMEAÇA: %1%").arg(score, 0, 'f', 1));
        }
    }

    void updateImage(QPixmap pixmap) {
        lblVideo->setPixmap(pixmap);
    }

private:
    void setupUI() {
        this->setWindowTitle("FireMonitor Pro");
        this->resize(1024, 600);
        
        // Estilo Global (Fundo Dark Grey + Fonte Monospace)
        this->setStyleSheet("QMainWindow { background-color: #1a1a1a; font-family: 'Monospace', 'Consolas', sans-serif; } QLabel { color: #eee; }");

        QWidget *central = new QWidget;
        setCentralWidget(central);
        
        // Layout Principal: Horizontal (Esquerda: Sidebar, Direita: Vídeo)
        QHBoxLayout *mainLayout = new QHBoxLayout(central);
        mainLayout->setContentsMargins(20, 20, 20, 20);
        mainLayout->setSpacing(20);

        // --- COLUNA DA ESQUERDA (SIDEBAR) ---
        QVBoxLayout *sideLayout = new QVBoxLayout();
        sideLayout->setAlignment(Qt::AlignTop);

        // 1. Cabeçalho
        QLabel *lblTitle = new QLabel("FIRE GUARD\nSYSTEM");
        lblTitle->setStyleSheet("font-size: 28px; font-weight: bold; color: #aaa; letter-spacing: 2px; border-bottom: 2px solid #444; padding-bottom: 10px;");
        sideLayout->addWidget(lblTitle);
        
        sideLayout->addSpacing(30);

        // 2. Status Principal
        QLabel *lblStatusTitle = new QLabel("STATUS ATUAL:");
        lblStatusTitle->setStyleSheet("font-size: 12px; color: #888;");
        sideLayout->addWidget(lblStatusTitle);

        lblStatusMain = new QLabel("INICIALIZANDO...");
        sideLayout->addWidget(lblStatusMain);

        sideLayout->addSpacing(30);

        // 3. Threat Meter (Barra de Progresso)
        QLabel *lblMetricTitle = new QLabel("ANÁLISE DE RISCO (IA):");
        lblMetricTitle->setStyleSheet("font-size: 12px; color: #888;");
        sideLayout->addWidget(lblMetricTitle);

        threatBar = new QProgressBar();
        threatBar->setRange(0, 100);
        threatBar->setValue(0);
        threatBar->setFixedHeight(30);
        threatBar->setTextVisible(true);
        threatBar->setFormat("%p%"); 
        sideLayout->addWidget(threatBar);

        lblDetails = new QLabel("Aguardando dados...");
        lblDetails->setStyleSheet("font-size: 14px; color: #ccc; margin-top: 5px;");
        sideLayout->addWidget(lblDetails);

        // Spacer para empurrar o resto para baixo
        sideLayout->addStretch();

        // 4. Rodapé da Sidebar (Relógio)
        lblTime = new QLabel("--:--:--");
        lblTime->setStyleSheet("font-size: 14px; color: #666; border-top: 1px solid #333; padding-top: 10px;");
        sideLayout->addWidget(lblTime);

        // Adiciona Sidebar ao Layout Principal (Ratio 1 de 4)
        mainLayout->addLayout(sideLayout, 1);


        // --- COLUNA DA DIREITA (VÍDEO) ---
        frameVideo = new QFrame();
        // Sombra/Glow
        QGraphicsDropShadowEffect *glow = new QGraphicsDropShadowEffect;
        glow->setBlurRadius(20);
        glow->setColor(QColor(0, 0, 0, 150));
        frameVideo->setGraphicsEffect(glow);

        QVBoxLayout *videoLayout = new QVBoxLayout(frameVideo);
        videoLayout->setContentsMargins(0,0,0,0);

        lblVideo = new QLabel();
        lblVideo->setAlignment(Qt::AlignCenter);
        lblVideo->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        lblVideo->setScaledContents(true);
        lblVideo->setStyleSheet("background-color: #000; border-radius: 6px;");

        videoLayout->addWidget(lblVideo);

        // Adiciona Vídeo ao Layout Principal (Ratio 3 de 4)
        mainLayout->addWidget(frameVideo, 3);
    }

    void startWorker() {
        QThread* thread = new QThread;
        MonitorWorker* worker = new MonitorWorker();
        worker->moveToThread(thread);
        connect(thread, &QThread::started, worker, &MonitorWorker::startLoop);
        connect(worker, &MonitorWorker::statusReceived, this, &MainWindow::updateStatus);
        connect(worker, &MonitorWorker::imageReceived, this, &MainWindow::updateImage);
        connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        connect(thread, &QThread::finished, thread, &QObject::deleteLater);
        thread->start();
    }

    // Widgets da UI
    QLabel *lblStatusMain;
    QLabel *lblDetails;
    QLabel *lblVideo;
    QLabel *lblTime;
    QProgressBar *threatBar;
    QFrame *frameVideo;
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show(); 
    return app.exec();
}
