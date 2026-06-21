#ifndef CHAT_TEMPLATES_H
#define CHAT_TEMPLATES_H

#include <QString>

namespace ChatTemplates {
    const QString CSS = R"(
        <style>
            .user-text { 
                background-color: #2f3037; 
                color: #ffffff; 
                border-radius: 12px; 
                padding: 10px; 
                font-size: 14px; 
                line-height: 1.4; 
            }
            .ai-text { 
                background-color: transparent; 
                color: #ececec; 
                font-size: 14px; 
                line-height: 1.6; 
            }
            .source-citation { 
                background-color: #2a2a35; 
                color: #10a37f; 
                border: 1px solid #3a3a4a; 
                border-radius: 4px; 
                padding: 2px 6px; 
                text-decoration: none; 
                font-size: 12px; 
                margin-right: 4px; 
            }
            .doc-citation { 
                color: #3a86ff; 
            }
            h3 { 
                color: #10a37f; 
                margin-top: 12px; 
                margin-bottom: 6px; 
                font-size: 16px; 
            }
            ul { 
                margin-top: 6px; 
                padding-left: 20px; 
            }
            li { 
                margin-bottom: 5px; 
            }
            .typing { 
                color: #10a37f; 
                font-style: italic; 
                padding: 10px; 
                font-size: 13px; 
            }
            .chat-start {
                color: #666; 
                font-size: 11px; 
                margin-bottom: 20px; 
                text-align: center;
            }
        </style>
    )";

    const QString USER_MESSAGE_CONTAINER = R"(
        <div align='right' style='margin-bottom:15px;'>
            %1 %2
        </div>
    )";

    const QString USER_TEXT_TABLE = R"(
        <table border='0' cellspacing='0' cellpadding='0'>
            <tr><td class='user-text'>%1</td></tr>
        </table>
    )";

    const QString AI_MESSAGE_CONTAINER = R"(
        <div align='left' style='margin-bottom:15px;'>
            <div class='ai-text'>%1</div>
        </div>
    )";

    const QString ATTACHMENT_CONTAINER = R"(
        <div style='margin-bottom:8px;'>%1</div>
    )";

    const QString IMAGE_ATTACHMENT = R"(
        <a href='img:%1'><img src='%2' width='150' style='border-radius:10px; margin-right:10px;'></a>
    )";

    const QString FILE_ATTACHMENT = R"(
        <div style='margin-bottom:4px;'><a href='%1' style='color:#10a37f; text-decoration:none;'>📎 %2</a></div>
    )";
}

#endif // CHAT_TEMPLATES_H
