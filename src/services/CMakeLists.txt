target_sources(AppCore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/SceneService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/SceneService.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/CrosshairGeometry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/CrosshairGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/CrosshairManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/CrosshairManager.cpp
    # VTK interaction style — lives here (not in src/utils) so the plugin-facing
    # utils surface stays free of VTK headers (enforced by the architecture guard).
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/PanStyle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/PanStyle.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/ViewerService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/ViewerService.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/Image2DLoader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/Image2DLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/Model3DLoader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/Model3DLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/DicomLoader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/DicomLoader.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/settings/SettingsService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/settings/SettingsService.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionPipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionPostProcessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionPostProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/SfmStage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/SfmStage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/TrackSfM.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/TrackSfM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/IncrementalSfM.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/IncrementalSfM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/CameraParamsParser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/CameraParamsParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionWorker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/ReconstructionWorker.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/FeatureTrack.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/MultiViewTriangulator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/MultiViewTriangulator.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIProcessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIYoloPostProcessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIYoloPostProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIAssistant.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIAssistant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIServerProcessController.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIServerProcessController.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIRequestQueue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIRequestQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/ChatSessionStore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/ChatSessionStore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/ChatSessionManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/ChatSessionManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIService.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIModelLoadService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ai/AIModelLoadService.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/MailService.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/MailService.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/MailMimeParser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/MailMimeParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/MailThreadGuard.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/SmtpMailer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/SmtpMailer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/SmtpOutbox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mail/SmtpOutbox.cpp

    
)
