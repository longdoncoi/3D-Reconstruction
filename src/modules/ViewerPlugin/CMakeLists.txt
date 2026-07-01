set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_library(ViewerPlugin SHARED
    ViewerPlugin.h
    ViewerPlugin.cpp
    ViewerRibbonUI.h
    ViewerRibbonUI.cpp
    ViewerNavigatorUI.h
    ViewerNavigatorUI.cpp
    ViewerListUI.h
    ViewerListUI.cpp
)

target_include_directories(ViewerPlugin PRIVATE
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
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ViewerPlugin
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        ${VTK_LIBRARIES}
        3D-Reconstruction # Link against the main executable if symbols are exported
)

# Export symbols from the main executable on Windows
set_target_properties(3D-Reconstruction PROPERTIES ENABLE_EXPORTS ON)

# Install plugin to a 'plugins' directory in the build folder
set_target_properties(ViewerPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
)
