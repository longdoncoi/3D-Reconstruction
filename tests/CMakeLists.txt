find_package(Qt6 REQUIRED COMPONENTS Test Core)
set(CMAKE_AUTOMOC ON)

# Test ServiceRegistry
qt_add_executable(TestServiceRegistry TestServiceRegistry.cpp)
target_include_directories(TestServiceRegistry PRIVATE ../src/core)
target_link_libraries(TestServiceRegistry PRIVATE Qt6::Test Qt6::Core)
add_test(NAME TestServiceRegistry COMMAND TestServiceRegistry)

# Test SignalBus
qt_add_executable(TestSignalBus TestSignalBus.cpp ../src/core/SignalBus.cpp)
target_include_directories(TestSignalBus PRIVATE ../src/core ../src/utils)
target_link_libraries(TestSignalBus PRIVATE Qt6::Test Qt6::Core)
target_compile_definitions(TestSignalBus PRIVATE APP_LIBRARY)
add_test(NAME TestSignalBus COMMAND TestSignalBus)

# Test AppConfig
qt_add_executable(TestAppConfig 
    TestAppConfig.cpp 
    ../src/core/AppConfig.cpp 
    ../src/core/LanguageManager.cpp
    ../src/utils/ModernDialog.cpp
    ../src/utils/IconFactory.cpp
)
target_include_directories(TestAppConfig PRIVATE ../src/core ../src/utils)
target_link_libraries(TestAppConfig PRIVATE Qt6::Test Qt6::Core Qt6::Widgets)
target_compile_definitions(TestAppConfig PRIVATE APP_LIBRARY)
add_test(NAME TestAppConfig COMMAND TestAppConfig)

# Test LanguageManager
qt_add_executable(TestLanguageManager 
    TestLanguageManager.cpp 
    ../src/core/LanguageManager.cpp 
    ../src/core/AppConfig.cpp
    ../src/utils/ModernDialog.cpp
    ../src/utils/IconFactory.cpp
)
target_include_directories(TestLanguageManager PRIVATE ../src/core ../src/utils)
target_link_libraries(TestLanguageManager PRIVATE Qt6::Test Qt6::Core Qt6::Widgets)
target_compile_definitions(TestLanguageManager PRIVATE APP_LIBRARY)
add_test(NAME TestLanguageManager COMMAND TestLanguageManager)

# Test Mail MIME parser
qt_add_executable(TestMailMimeParser
    TestMailMimeParser.cpp
    ../src/services/mail/MailMimeParser.cpp
)
target_include_directories(TestMailMimeParser PRIVATE ../src/core ../src/services/mail)
target_link_libraries(TestMailMimeParser PRIVATE Qt6::Test Qt6::Core)
add_test(NAME TestMailMimeParser COMMAND TestMailMimeParser)

# Test Camera Params Parser
find_package(OpenCV REQUIRED COMPONENTS core)
qt_add_executable(TestCameraParamsParser
    TestCameraParamsParser.cpp
    ../src/services/reconstruction/CameraParamsParser.cpp
)
target_include_directories(TestCameraParamsParser PRIVATE ../src/services/reconstruction)
target_link_libraries(TestCameraParamsParser PRIVATE Qt6::Test Qt6::Core ${OpenCV_LIBS})
add_test(NAME TestCameraParamsParser COMMAND TestCameraParamsParser)

# Test PointCloudFilter
find_package(PCL REQUIRED COMPONENTS common filters io)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d)

qt_add_executable(TestPointCloudFilter TestPointCloudFilter.cpp ../src/services/reconstruction/PointCloudFilter.cpp)
target_include_directories(TestPointCloudFilter PRIVATE 
    ../src/services/reconstruction
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestPointCloudFilter PRIVATE 
    Qt6::Test Qt6::Core Qt6::Widgets
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)
add_test(NAME TestPointCloudFilter COMMAND TestPointCloudFilter)

# Test ReconstructionPipeline
qt_add_executable(TestReconstructionPipeline TestReconstructionPipeline.cpp 
    ../src/services/reconstruction/ReconstructionPipeline.cpp
    ../src/services/reconstruction/CameraParamsParser.cpp
    ../src/services/reconstruction/FeatureExtractor.cpp
    ../src/services/reconstruction/FeatureMatcher.cpp
    ../src/services/reconstruction/PoseEstimator.cpp
    ../src/services/reconstruction/Triangulator.cpp
    ../src/services/reconstruction/PointCloudFilter.cpp
    ../src/services/reconstruction/FeatureTrack.h
    ../src/services/reconstruction/MultiViewTriangulator.cpp
)
target_include_directories(TestReconstructionPipeline PRIVATE 
    ../src/services/reconstruction
    ../src/core
    ../src/app
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestReconstructionPipeline PRIVATE 
    Qt6::Test Qt6::Core Qt6::Widgets Qt6::Concurrent
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)
target_compile_definitions(TestReconstructionPipeline PRIVATE APP_LIBRARY)
add_test(NAME TestReconstructionPipeline COMMAND TestReconstructionPipeline)

# Test Agent Actions
qt_add_executable(TestAgentActions TestAgentActions.cpp ../src/core/SignalBus.cpp)
target_include_directories(TestAgentActions PRIVATE ../src/core ../src/utils)
target_link_libraries(TestAgentActions PRIVATE Qt6::Test Qt6::Core)
target_compile_definitions(TestAgentActions PRIVATE APP_LIBRARY)
add_test(NAME TestAgentActions COMMAND TestAgentActions)
