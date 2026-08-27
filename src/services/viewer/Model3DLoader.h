#ifndef MODEL3DLOADER_H
#define MODEL3DLOADER_H

#include <QString>
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkProperty.h>
#include <vtkOBJImporter.h>
#include <vtkOBJReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vector>

#include "Global.h"
class APP_EXPORT Model3DLoader {
public:
    // Supports OBJ (+ optional MTL), STL, PLY, legacy VTK and XML VTP meshes.
    static std::vector<vtkSmartPointer<vtkActor>> load(const QString& modelPath, const QString& materialPath = {});
};

#endif // MODEL3DLOADER_H
