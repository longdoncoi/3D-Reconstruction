#ifndef CROSSHAIR_GEOMETRY_H
#define CROSSHAIR_GEOMETRY_H

namespace CrosshairGeometry {

const double *resliceAxes(int orientation);
const double *lineColor(int orientation);
void labels(int orientation, const char *out[4]);
void worldToNormalized(int orientation, const double world[3], const double bounds[6], double &nx, double &ny);
void normalizedToWorld(int orientation, double nx, double ny, const double bounds[6],
                       const double oldWorld[3], double newWorld[3]);

}

#endif // CROSSHAIR_GEOMETRY_H
