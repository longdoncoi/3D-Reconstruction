#include "CrosshairGeometry.h"

#include "CrosshairManager.h"

#include <algorithm>

namespace {

const double kAxes[3][16] = {
    {0, 0, 1, 0,
     1, 0, 0, 0,
     0, 1, 0, 0,
     0, 0, 0, 1},
    {1, 0, 0, 0,
     0, 0, 1, 0,
     0, 1, 0, 0,
     0, 0, 0, 1},
    {1,  0, 0, 0,
     0, -1, 0, 0,
     0,  0, 1, 0,
     0,  0, 0, 1}
};

const char *kLabels[3][4] = {
    {"P", "A", "S", "I"},
    {"R", "L", "S", "I"},
    {"R", "L", "A", "P"}
};

const double kLineColor[3][3] = {
    {1.0, 1.0, 0.0},
    {0.0, 1.0, 1.0},
    {1.0, 0.5, 0.0}
};

}

namespace CrosshairGeometry {

const double *resliceAxes(int orientation)
{
    return kAxes[orientation];
}

const double *lineColor(int orientation)
{
    return kLineColor[orientation];
}

void labels(int orientation, const char *out[4])
{
    for (int index = 0; index < 4; ++index) {
        out[index] = kLabels[orientation][index];
    }
}

void worldToNormalized(int orientation, const double world[3], const double bounds[6], double &nx, double &ny)
{
    const double minX = bounds[0];
    const double maxX = bounds[1];
    const double minY = bounds[2];
    const double maxY = bounds[3];
    const double minZ = bounds[4];
    const double maxZ = bounds[5];

    switch (orientation) {
    case ORI_SAGITTAL:
        nx = (world[1] - minY) / (maxY - minY + 1e-9);
        ny = (world[2] - minZ) / (maxZ - minZ + 1e-9);
        break;
    case ORI_CORONAL:
        nx = (world[0] - minX) / (maxX - minX + 1e-9);
        ny = (world[2] - minZ) / (maxZ - minZ + 1e-9);
        break;
    case ORI_AXIAL:
        nx = (world[0] - minX) / (maxX - minX + 1e-9);
        ny = ((-world[1]) - (-maxY)) / ((-minY) - (-maxY) + 1e-9);
        break;
    }

    nx = std::max(0.0, std::min(1.0, nx));
    ny = std::max(0.0, std::min(1.0, ny));
}

void normalizedToWorld(int orientation, double nx, double ny, const double bounds[6],
                       const double oldWorld[3], double newWorld[3])
{
    newWorld[0] = oldWorld[0];
    newWorld[1] = oldWorld[1];
    newWorld[2] = oldWorld[2];

    const double minX = bounds[0];
    const double maxX = bounds[1];
    const double minY = bounds[2];
    const double maxY = bounds[3];
    const double minZ = bounds[4];
    const double maxZ = bounds[5];

    switch (orientation) {
    case ORI_SAGITTAL:
        newWorld[1] = minY + nx * (maxY - minY);
        newWorld[2] = minZ + ny * (maxZ - minZ);
        break;
    case ORI_CORONAL:
        newWorld[0] = minX + nx * (maxX - minX);
        newWorld[2] = minZ + ny * (maxZ - minZ);
        break;
    case ORI_AXIAL:
        newWorld[0] = minX + nx * (maxX - minX);
        newWorld[1] = maxY - ny * (maxY - minY);
        break;
    }
}

}
