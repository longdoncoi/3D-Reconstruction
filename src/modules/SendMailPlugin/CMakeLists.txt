

add_library(SendMailPlugin SHARED
    SendMailPlugin.h
    SendMailPlugin.cpp
    SendMailRibbonUI.h
    SendMailRibbonUI.cpp
    MailDockWidget.h
    MailDockWidget.cpp
)

target_include_directories(SendMailPlugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/app
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/services/mail
    ${VTK_INCLUDE_DIRS}
)

target_precompile_headers(SendMailPlugin REUSE_FROM SharedPCH)
target_link_libraries(SendMailPlugin PRIVATE SharedPCH)

target_link_libraries(SendMailPlugin
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Network
        Qt6::Concurrent
        ${VTK_LIBRARIES}
        3D-Reconstruction
)

set_target_properties(3D-Reconstruction PROPERTIES ENABLE_EXPORTS ON)

set_target_properties(SendMailPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
)
