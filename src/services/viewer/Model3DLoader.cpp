#include "Model3DLoader.h"
#include <QFileInfo>
#include <QDebug>
#include <vtkPLYReader.h>
#include <vtkAlgorithm.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkSTLReader.h>
#include <vtkXMLPolyDataReader.h>

std::vector<vtkSmartPointer<vtkActor>> Model3DLoader::load(const QString& modelPath, const QString& materialPath) {
    std::vector<vtkSmartPointer<vtkActor>> actorsList;
    QFileInfo modelFile(modelPath);
    QFileInfo materialFile(materialPath);

    if (!modelFile.exists()) {
        qWarning() << "3D model file not found:" << modelPath;
        return actorsList;
    }

    const QString suffix = modelFile.suffix().toLower();
    bool imported = false;
    if (suffix == "obj" && materialFile.exists()) {
        vtkNew<vtkOBJImporter> importer;
        importer->SetFileName(modelPath.toStdString().c_str());
        importer->SetFileNameMTL(materialPath.toStdString().c_str());
        importer->Update();

        vtkRenderer *importerRenderer = importer->GetRenderer();
        if (importerRenderer) {
            vtkActorCollection *actors = importerRenderer->GetActors();
            actors->InitTraversal();
            vtkActor *actor;
            while ((actor = actors->GetNextActor())) {
                actor->GetProperty()->SetLighting(true);
                actor->GetProperty()->SetInterpolationToPhong();
                actor->GetProperty()->SetAmbient(0.3);
                actor->GetProperty()->SetDiffuse(0.8);
                actorsList.push_back(actor);
            }
            imported = true;
        }
    }

    if (!imported) {
        vtkSmartPointer<vtkAlgorithm> reader;
        if (suffix == "obj") {
            auto concreteReader = vtkSmartPointer<vtkOBJReader>::New();
            concreteReader->SetFileName(modelPath.toStdString().c_str());
            reader = concreteReader;
        } else if (suffix == "stl") {
            auto concreteReader = vtkSmartPointer<vtkSTLReader>::New();
            concreteReader->SetFileName(modelPath.toStdString().c_str());
            reader = concreteReader;
        } else if (suffix == "ply") {
            auto concreteReader = vtkSmartPointer<vtkPLYReader>::New();
            concreteReader->SetFileName(modelPath.toStdString().c_str());
            reader = concreteReader;
        } else if (suffix == "vtk") {
            auto concreteReader = vtkSmartPointer<vtkPolyDataReader>::New();
            concreteReader->SetFileName(modelPath.toStdString().c_str());
            reader = concreteReader;
        } else if (suffix == "vtp") {
            auto concreteReader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
            concreteReader->SetFileName(modelPath.toStdString().c_str());
            reader = concreteReader;
        }
        else {
            qWarning() << "Unsupported 3D model format:" << suffix;
            return actorsList;
        }
        reader->Update();
        auto* polyData = vtkPolyData::SafeDownCast(reader->GetOutputDataObject(0));
        if (!polyData || polyData->GetNumberOfPoints() == 0) {
            qWarning() << "3D model contains no mesh data:" << modelPath;
            return actorsList;
        }
        
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(reader->GetOutputPort());
        
        vtkSmartPointer<vtkActor> fallbackActor = vtkSmartPointer<vtkActor>::New();
        fallbackActor->SetMapper(mapper);
        fallbackActor->GetProperty()->SetColor(0.7, 0.7, 0.7);
        fallbackActor->GetProperty()->SetInterpolationToPhong();
        actorsList.push_back(fallbackActor);
    }

    return actorsList;
}
