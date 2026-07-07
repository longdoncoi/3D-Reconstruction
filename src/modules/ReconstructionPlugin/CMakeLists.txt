set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_library(ReconstructionPlugin SHARED
    ReconstructionPlugin.h
    ReconstructionPlugin.cpp
    ReconstructionRibbonUI.h
    ReconstructionRibbonUI.cpp
)

target_include_directories(ReconstructionPlugin PRIVATE
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
    ${PCL_INCLUDE_DIRS}
)

target_precompile_headers(ReconstructionPlugin PRIVATE ${PROJECT_SOURCE_DIR}/src/pch.h)

target_link_libraries(ReconstructionPlugin
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        3D-Reconstruction
)

set_target_properties(ReconstructionPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
)
