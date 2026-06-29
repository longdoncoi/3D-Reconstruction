# AI Coding Instructions for 3D-Reconstruction

## Mission
Build maintainable, scalable, testable and production-quality code.

Priority order:
1. Correctness
2. Architecture
3. Safety
4. Maintainability
5. Performance
6. Convenience

Never sacrifice architecture for short-term implementation speed.

---

# Project Overview

This project is a modular medical imaging and 3D reconstruction platform.

Main technologies:

- C++20
- Qt 6
- OpenCV
- VTK
- PCL
- ONNX Runtime
- DICOM
- Python AI Services

Architecture style:

- Service-Oriented Architecture
- Modular Plugin Architecture
- Event Driven Communication
- Dependency Inversion

---

# Setup & Build

## Prerequisites

- CMake 3.20+
- Qt 6
- C++20 compiler (MSVC 2022+ on Windows, GCC 11+ on Linux)
- OpenCV, VTK, PCL, ONNX Runtime

## Build on Windows

```powershell
# Configure
cmake -B build

# Build
cmake --build build --config Release

# Run tests
ctest --test-dir build --output-on-failure
```

## Build on Linux

```bash
# Configure
cmake -B build

# Build
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure
```

## Run Application

```powershell
# Windows
.\build\Release\3D-Reconstruction.exe

# Linux
./build/3D-Reconstruction
```

## Agent Pipeline

Automated build & test pipeline (Local Pre-commit Check):

```powershell
# Windows
.\scripts\agent_pipeline.ps1

# Linux/Mac
bash ./scripts/agent_pipeline.sh
```

---

# AI Agent Automation

## Task Completion Workflow

**IMPORTANT: AI Agent MUST follow this workflow after completing each task:**

1. **After coding is complete:**
   - All files created/modified
   - Code tested locally
   - Architecture verified

2. **Automatically execute pipeline:**
   
   **Windows:**
   ```powershell
   .\scripts\agent_pipeline.ps1
   ```
   
   **Linux/Mac:**
   ```bash
   bash ./scripts/agent_pipeline.sh
   ```

3. **Pipeline steps (automatic local checks before push):**
   - `git pull` - Fetch latest changes
   - `cmake --build build --config Release` - Build project (C++)
   - `ctest --output-on-failure` - Run local tests (C++)
   - `ruff check AITraining/` - Run Python linting
   - `git add .` - Stage changes
   - `git commit -m "Agent task completed"` - Commit
   - `git push` - Push to remote (Triggers `ci.yml` and `python-ci.yml`)

4. **Report results:**
   - ✅ If success: Inform user task completed with all tests passing locally.

## Exit Code Handling

- Exit code `0` = Success → Continue to next step
- Exit code `≠ 0` = Failure → Stop pipeline, report error to user

---

# Core Architectural Rules

## Plugin Isolation

Plugins are independent modules.

Allowed:

Plugin -> IAppContext
Plugin -> SignalBus
Plugin -> Core Services

Forbidden:

Plugin -> MainWindow
Plugin -> Another Plugin
Plugin -> Global State

Plugins must communicate only through:

- IAppContext
- SignalBus
- Service Interfaces

Never create direct dependencies between plugins.

---

## MainWindow Rules

MainWindow is UI composition only.

Forbidden:

- AI inference
- Reconstruction logic
- DICOM processing
- Business logic

MainWindow may:

- Host widgets
- Dispatch actions
- Manage docking layouts

---

## Service Layer

Business logic belongs inside services.

Examples:

- ReconstructionService
- DicomLoaderService
- InferenceService
- PointCloudService

Widgets must call services.

Services must never depend on widgets.

---

# SOLID Requirements

## Single Responsibility Principle

Each class must have exactly one reason to change.

Bad:

ViewerWidget
- Load DICOM
- Run AI
- Reconstruct 3D
- Display image

Good:

ViewerWidget
DicomLoader
InferenceService
ReconstructionEngine

---

## Open Closed Principle

Prefer extension over modification.

New functionality should be implemented using:

- New Plugin
- New Service
- New Strategy

Avoid editing stable modules.

---

## Liskov Substitution Principle

Derived classes must be usable anywhere their interfaces are expected.

Never weaken interface contracts.

---

## Interface Segregation Principle

Prefer multiple small interfaces.

Bad:

IMedicalSystem

Good:

IViewerService
ISceneService
IInferenceService
IReconstructionService

---

## Dependency Inversion Principle

Depend on abstractions.

Always inject interfaces.

Never inject concrete implementations when an abstraction exists.

---

# Design Patterns

Preferred Patterns:

- Factory
- Abstract Factory
- Builder
- Strategy
- Observer
- Mediator
- Command
- Adapter
- Facade
- Repository
- Dependency Injection
- Service Locator

---

## Strategy Pattern

Required for reconstruction algorithms.

Examples:

IReconstructionStrategy

- SfMStrategy
- StereoStrategy
- FringeProjectionStrategy

---

## Factory Pattern

Required for:

- Model loading
- Plugin creation
- Algorithm selection

---

## Observer Pattern

Implemented through SignalBus.

Use SignalBus instead of direct module references.

---

# Forbidden Architecture

Never introduce:

- God Objects
- Circular Dependencies
- Global Mutable State
- Tight Coupling
- Hidden Dependencies
- Singleton Abuse

---

# C++ Coding Standards

## Naming

Classes:
PascalCase

Interfaces:
Prefix I

Member Variables:
m_

Static Variables:
s_

Constants:
kPascalCase

Namespaces:
lowercase

---

# Smart Pointer Policy

Default:

std::unique_ptr

Shared ownership only:

std::shared_ptr

Break cycles:

std::weak_ptr

Forbidden:

raw new
raw delete
malloc
free

Exceptions:

- Qt parent ownership
- vtkSmartPointer
- Legacy third-party APIs

---

# Thread Safety Rules

All shared mutable data must be synchronized.

Use:

- std::mutex
- std::shared_mutex
- std::lock_guard
- std::scoped_lock

Never access shared containers without synchronization.

---

## UI Thread Rule

Qt Widgets must only be updated from UI thread.

Worker threads must communicate using:

- signals/slots
- queued connections
- QMetaObject::invokeMethod

Direct UI access from worker threads is forbidden.

---

# Race Condition Prevention

Avoid shared mutable state.

Prefer:

- immutable data
- message passing
- SignalBus events

Avoid:

- static mutable variables
- hidden caches without locking

---

# Memory Rules

Avoid copies of large objects.

Pass:

const cv::Mat&
const std::vector<T>&
const QString&

Use move semantics where ownership transfers.

Reserve vector capacity when size is known.

---

# OpenCV Rules

OpenCV handles:

- image processing
- calibration
- feature extraction
- triangulation

Do not place OpenCV algorithms inside widgets.

---

# PCL Rules

PCL handles:

- point clouds
- filtering
- registration

Point cloud operations belong inside services.

---

# VTK Rules

VTK handles visualization only.

Rendering code must stay separate from reconstruction algorithms.

---

# ONNX Runtime Rules

Model loading through factories.

Inference must run in background threads.

UI must never block waiting for inference.

---

# Error Handling

Never swallow exceptions.

Forbidden:

catch(...)
{
}

Required:

- log
- recover
- rethrow

---

# Logging

No std::cout
No printf

Use centralized logger.

Levels:

TRACE
DEBUG
INFO
WARNING
ERROR
FATAL

---

# Testing Requirements

Business logic must be testable.

Business logic must not depend on:

- QWidget
- MainWindow

Services should be independently testable.

---

# Refactoring Rules

Before creating new code:

1. Search existing implementation.
2. Reuse services.
3. Reuse interfaces.
4. Avoid duplication.

Follow Rule of Three before creating abstractions.

---

# Performance Rules

Avoid unnecessary heap allocations.

Avoid copying:

- cv::Mat
- pcl::PointCloud
- vtkPolyData

Prefer:

- reserve()
- emplace_back()
- move semantics

Measure before optimizing.

---

# AI Agent Workflow

Before coding:

1. Read related source files.
2. Understand architecture.
3. Search for existing implementations.
4. Identify ownership boundaries.
5. Identify threading implications.

While coding:

1. Preserve architecture.
2. Keep functions focused.
3. Keep classes cohesive.
4. Respect plugin boundaries.

After coding:

1. Build successfully.
2. Verify thread safety.
3. Verify ownership.
4. Verify no circular dependencies.
5. Verify no memory leaks.

---

# Pull Request Checklist

[ ] SOLID respected
[ ] Plugin architecture respected
[ ] No direct plugin dependencies
[ ] No MainWindow business logic
[ ] No memory leak
[ ] No race condition
[ ] No raw ownership
[ ] Thread safe
[ ] Build passes (Local & Remote `ci.yml`)
[ ] Tests pass (Local CTest)
[ ] Python Lint passes (Ruff, `python-ci.yml`)
[ ] Default branch targeted is `main`
[ ] Release triggers MSI Installer generation (`release.yml`)
[ ] No duplicated code
[ ] No God Object
