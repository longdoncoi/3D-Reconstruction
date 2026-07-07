set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_library(AIProcessorPlugin SHARED
    AIProcessorPlugin.h
    AIProcessorPlugin.cpp
    AITrainDockWidget.h
    AITrainDockWidget.cpp
    AIProcessorRibbonUI.h
    AIProcessorRibbonUI.cpp
    VideoTrackerThread.h
    VideoTrackerThread.cpp
)

target_include_directories(AIProcessorPlugin PRIVATE
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
    ${onnxruntime_SOURCE_DIR}/include
)

target_precompile_headers(AIProcessorPlugin PRIVATE ${PROJECT_SOURCE_DIR}/src/pch.h)

target_link_libraries(AIProcessorPlugin
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Concurrent
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
        3D-Reconstruction
)

set_target_properties(AIProcessorPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
)
