#ifndef APP_CONSTANTS_H
#define APP_CONSTANTS_H

#include <QString>
#include <QStringList>

namespace AppConstants {

    // ── Reconstruction Pipeline ──────────────────────────────────────────────
    namespace Reconstruction {
        constexpr int    SIFT_MAX_FEATURES              = 20000;
        constexpr int    SIFT_OCTAVE_LAYERS             = 3;
        constexpr double SIFT_CONTRAST_THRESHOLD        = 0.01;
        constexpr double SIFT_EDGE_THRESHOLD            = 10.0;
        constexpr double SIFT_SIGMA                     = 1.6;

        constexpr float  MATCH_RATIO_THRESHOLD          = 0.8f;
        constexpr int    MIN_MATCHES_FOR_POSE           = 30;
        constexpr int    MIN_INLIERS_FOR_POSE           = 12;
        constexpr int    MIN_INLIERS_FOR_ESTIMATED_POSE = 30;

        constexpr int    SEARCH_WINDOW_SIZE             = 15;
        constexpr double DEPTH_LIMIT_MULTIPLIER         = 30.0;

        constexpr float  DEPTH_MIN_MEDIAN_RATIO         = 0.05f;
        constexpr float  DEPTH_MAX_MEDIAN_RATIO         = 20.0f;
        constexpr double REPROJECTION_ERROR_THRESHOLD   = 2.0;
    }

    // ── AI Assistant UI ──────────────────────────────────────────────────────
    namespace AIAssistant {
        constexpr int DEFAULT_DOCK_WIDTH        = 520;
        constexpr int SESSION_PANEL_MIN_WIDTH   = 120;
        constexpr int SESSION_PANEL_MAX_WIDTH   = 280;
        constexpr int MAX_ATTACHMENT_SIZE_MB     = 10;
        constexpr int VISION_MODEL_INDEX         = 2;

        // AI model selector options
        inline QStringList modelNames() {
            return {
                QStringLiteral("Qwen2.5-7B (Q4_K_M) — Text"),
                QStringLiteral("Qwen2.5-coder-7B (Q4_K_M) — Coder"),
                QStringLiteral("Qwen2.5-VL-7B (Q4_K_M) — Vision 👁️")
            };
        }
    }

    // ── AI Server & Network ──────────────────────────────────────────────────
    namespace AIServer {
        constexpr int    SERVER_PORT                = 8080;
        constexpr double DEFAULT_TEMPERATURE        = 0.7;
        constexpr int    DEFAULT_MAX_TOKENS         = 512;
        constexpr int    STOP_SERVER_TIMEOUT_MS     = 3000;
        constexpr int    TEXT_INFERENCE_TIMEOUT_MS  = 300000;  // 5 min
        constexpr int    VISION_INFERENCE_TIMEOUT_MS = 900000; // 15 min
        constexpr int    INFERENCE_TIMEOUT_MS       = TEXT_INFERENCE_TIMEOUT_MS;

        inline QString apiEndpoint() {
            return QStringLiteral("http://127.0.0.1:%1/v1/chat/completions").arg(SERVER_PORT);
        }

        inline QString chatbotScript() {
            return QStringLiteral("StartChatbotServer.py");
        }
    }

    // ── AI Processor / ONNX Models ───────────────────────────────────────────
    namespace AIProcessor {
        inline QString detectionModelFile()     { return QStringLiteral("yolo11n.onnx"); }
        inline QString segmentationModelFile()  { return QStringLiteral("yolo11n-seg.onnx"); }
        inline QString trainScript()            { return QStringLiteral("TrainModel.py"); }

        constexpr int    TENSORBOARD_PORT           = 6006;
        constexpr int    TENSORBOARD_WAIT_SECONDS   = 12;

        inline QString tensorboardUrl() {
            return QStringLiteral("http://localhost:%1/").arg(TENSORBOARD_PORT);
        }
    }

    // ── UI Dimensions ────────────────────────────────────────────────────────
    namespace UI {
        // Common button heights
        constexpr int BTN_HEIGHT_SMALL   = 34;
        constexpr int BTN_HEIGHT_MEDIUM  = 38;
        constexpr int BTN_HEIGHT_LARGE   = 42;
        constexpr int BTN_HEIGHT_XLARGE  = 46;

        // Input field heights
        constexpr int INPUT_HEIGHT       = 40;

        // Ribbon tool button minimum width
        constexpr int TOOL_BTN_MIN_WIDTH = 80;

        // Navigator
        constexpr int NAV_WIDGET_HEIGHT     = 80;
        constexpr int NAV_BTN_MIN_WIDTH     = 80;
        constexpr int NAV_AUTO_BTN_MIN_WIDTH = 100;

        // Settings dialog
        constexpr int SETTINGS_WIDTH         = 660;
        constexpr int SETTINGS_HEIGHT        = 480;
        constexpr int SETTINGS_NAV_WIDTH     = 200;

        // Dialogs
        constexpr int ABOUT_DIALOG_MIN_WIDTH    = 440;
        constexpr int LICENSE_DIALOG_MIN_WIDTH  = 480;

        // Icon sizes
        constexpr int ICON_SIZE_SMALL  = 32;
        constexpr int ICON_SIZE_LARGE  = 64;
    }

    // ── File Filters ─────────────────────────────────────────────────────────
    namespace FileFilter {
        inline QString images()     { return QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp)"); }
        inline QString objFiles()   { return QStringLiteral("OBJ Files (*.obj)"); }
        inline QString allFiles()   { return QStringLiteral("All Files (*.*)"); }
    }

    // ── Session Title Limits ─────────────────────────────────────────────────
    namespace Chat {
        constexpr int SESSION_TITLE_MAX_LENGTH = 30;
    }

    // ── Date / Time Formats ──────────────────────────────────────────────────
    namespace Format {
        inline QString dateForFolder()      { return QStringLiteral("yyyy-MM-dd"); }
        inline QString dateTimeForFile()    { return QStringLiteral("yyyyMMdd_HHmmss"); }
        inline QString sessionDateTime()    { return QStringLiteral("dd/MM HH:mm"); }
        inline QString chatTimestamp()      { return QStringLiteral("HH:mm"); }
    }
}

#endif // APP_CONSTANTS_H
