#ifndef CAMERA_PARAMS_PARSER_H
#define CAMERA_PARAMS_PARSER_H

#include "ReconstructionConfig.h"

#include <QString>
#include <vector>

namespace CameraParamsParser {

bool loadFromFile(const QString &paramsFilePath, std::vector<CameraParams> &cameraParams);

}

#endif // CAMERA_PARAMS_PARSER_H
