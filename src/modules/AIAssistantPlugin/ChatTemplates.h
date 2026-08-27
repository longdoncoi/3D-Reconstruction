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
            %1 %2 %3
        </div>
    )";

    const QString USER_ACTION_FOOTER = R"(
        <div style='font-size:11px; color:#888; text-align:right; margin-top:4px;'>
            %1
            <a href='action:retry:%2' style='text-decoration:none; color:#aaa; margin-left:8px; font-size:14px;' title='Retry'>&#x21BB;</a>
            <a href='action:edit:%2' style='text-decoration:none; color:#aaa; margin-left:8px; font-size:14px;' title='Edit'>&#x270E;</a>
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

    const QString AGENT_THINKING = R"(
        <div style='background-color:#202026; border-left: 3px solid #8b5cf6; border-radius: 4px; padding: 8px; margin-bottom: 8px;'>
            <div style='color:#8b5cf6; font-weight:bold; font-size:12px; margin-bottom:4px;'>%1</div>
            <div style='color:#bbb; font-style:italic; font-size:12px; white-space:pre-wrap;'>%2</div>
        </div>
    )";

    const QString AGENT_TOOL_CALL = R"(
        <div style='background-color:#2f3037; border-left: 3px solid #3b82f6; border-radius: 4px; padding: 8px; margin-bottom: 8px;'>
            <div style='color:#3b82f6; font-weight:bold; font-size:12px; margin-bottom:4px;'>%1 %2</div>
            <div style='color:#aaa; font-family:monospace; font-size:11px; white-space:pre-wrap;'>%3</div>
        </div>
    )";

    const QString AGENT_TOOL_RESULT = R"(
        <div style='background-color:#2a2a35; border-left: 3px solid #10a37f; border-radius: 4px; padding: 8px; margin-bottom: 12px;'>
            <div style='color:#10a37f; font-weight:bold; font-size:12px; margin-bottom:4px;'>✓ Result</div>
            <div style='color:#ccc; font-family:monospace; font-size:11px; white-space:pre-wrap; max-height:200px; overflow:hidden;'>%1</div>
        </div>
    )";

    const QString AGENT_TOOL_RESULT_LOCALIZED = R"(
        <div style='background-color:#2a2a35; border-left: 3px solid #10a37f; border-radius: 4px; padding: 8px; margin-bottom: 12px;'>
            <div style='color:#10a37f; font-weight:bold; font-size:12px; margin-bottom:4px;'>&#10003; %1</div>
            <div style='color:#ccc; font-family:monospace; font-size:11px; white-space:pre-wrap; max-height:200px; overflow:hidden;'>%2</div>
        </div>
    )";

    const QString AGENT_ACTION_PROCESSED = R"(
        <div style='background-color:#173a32; border-left: 3px solid #10a37f; border-radius: 4px; padding: 12px; margin-bottom: 15px;'>
            <div style='color:#a7f3d0; font-weight:bold; font-size:13px; margin-bottom:8px;'>&#10003; %1</div>
            <div style='color:#eee; font-size:12px;'>%2</div>
        </div>
    )";

    const QString AGENT_APPROVAL_BLOCK = R"(
        <div style='background-color:#4a1515; border-left: 3px solid #ef4444; border-radius: 4px; padding: 12px; margin-bottom: 15px;'>
            <div style='color:#ef4444; font-weight:bold; font-size:13px; margin-bottom:8px;'>⚠️ %1</div>
            <div style='color:#eee; font-size:12px; margin-bottom:12px;'>%2</div>
            <div>
                <a href='agent:approve:%3' style='display:inline-block; background-color:#10a37f; color:white; text-decoration:none; padding:4px 12px; border-radius:4px; font-weight:bold; margin-right:10px;'>%4</a>
                <a href='agent:reject:%3' style='display:inline-block; background-color:#3f3f46; color:white; text-decoration:none; padding:4px 12px; border-radius:4px;'>%5</a>
            </div>
        </div>
    )";
}

#endif // CHAT_TEMPLATES_H
