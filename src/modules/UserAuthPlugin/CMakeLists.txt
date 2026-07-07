set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_library(UserAuthPlugin SHARED
    UserAuthPlugin.h
    UserAuthPlugin.cpp
    LoginDialog.h
    LoginDialog.cpp
    ForgotPasswordDialog.h
    ForgotPasswordDialog.cpp
    ChangePasswordDialog.h
    ChangePasswordDialog.cpp
    LicenseActivationDialog.h
    LicenseActivationDialog.cpp
    AdminUserManagerDialog.h
    AdminUserManagerDialog.cpp
    AboutDialog.h
    AboutDialog.cpp
    AvatarCropperDialog.h
    AvatarCropperDialog.cpp
    ThemeSelectionDialog.h
    ThemeSelectionDialog.cpp
    SettingsDialog.h
    SettingsDialog.cpp
)

target_include_directories(UserAuthPlugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/app
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/services/scene
    ${PROJECT_SOURCE_DIR}/src/services/viewer
    ${PROJECT_SOURCE_DIR}/src/services/settings
    ${PROJECT_SOURCE_DIR}/src/services/reconstruction
    ${PROJECT_SOURCE_DIR}/src/services/ai
    ${VTK_INCLUDE_DIRS}
)

target_precompile_headers(UserAuthPlugin PRIVATE ${PROJECT_SOURCE_DIR}/src/pch.h)

target_link_libraries(UserAuthPlugin
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Network
        ${VTK_LIBRARIES}
        3D-Reconstruction
)

set_target_properties(UserAuthPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
)
