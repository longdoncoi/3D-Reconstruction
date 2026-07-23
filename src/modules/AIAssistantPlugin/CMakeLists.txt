

add_library(AIAssistantPlugin SHARED
    AIAssistantPlugin.h
    AIAssistantPlugin.cpp
    AIAssistantRibbonUI.h
    AIAssistantRibbonUI.cpp
    ChatBotDockWidget.h
    ChatBotDockWidget.cpp
    ChatMessageRenderer.h
    ChatMessageRenderer.cpp
    ChatTemplates.h
)

target_include_directories(AIAssistantPlugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/app
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/services/scene
    ${PROJECT_SOURCE_DIR}/src/services/viewer
    ${PROJECT_SOURCE_DIR}/src/services/settings
    ${PROJECT_SOURCE_DIR}/src/services/reconstruction
    ${PROJECT_SOURCE_DIR}/src/services/ai
    ${PROJECT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../UserAuthPlugin
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_precompile_headers(AIAssistantPlugin REUSE_FROM SharedPCH)
target_link_libraries(AIAssistantPlugin PRIVATE SharedPCH)

target_link_libraries(AIAssistantPlugin
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Network
        ${VTK_LIBRARIES}
        3D-Reconstruction
        UserAuthPlugin
)

set_target_properties(AIAssistantPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
)
