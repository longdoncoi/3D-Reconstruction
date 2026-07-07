target_sources(3D-Reconstruction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/SceneService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/SceneService.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/CrosshairManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/CrosshairManager.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/ViewerService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/ViewerService.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/Image2DLoader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/Image2DLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/Model3DLoader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/Model3DLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/DicomLoader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/DicomLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/ChatImageViewer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/ChatImageViewer.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/settings/SettingsService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/settings/SettingsService.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionPipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructThread.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructThread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionService.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/FeatureExtractor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/FeatureExtractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/FeatureMatcher.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/FeatureMatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/PoseEstimator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/PoseEstimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/Triangulator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/Triangulator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/PointCloudFilter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/PointCloudFilter.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIProcessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIAssistant.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIAssistant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIService.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/MailService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/MailService.cpp
)
