# Qt Creator includes this file before its generated package-manager setup.
# Its default vcpkg integration searches <source>/vcpkg first and creates a
# separate build/vcpkg_installed tree.  This project instead uses the shared,
# prebuilt vcpkg tree configured at the top of CMakeLists.txt.
set(QT_CREATOR_SKIP_VCPKG_SETUP ON CACHE BOOL
    "Use the project's shared vcpkg toolchain instead of Qt Creator auto-setup" FORCE)

# The Maintenance Tool provider wraps every find_package() call.  Together
# with vcpkg's package wrappers this can recurse indefinitely (for example
# when VTK requests GLEW).  Dependencies are already resolved by CMake and
# the selected Qt kit, so this provider is neither needed nor appropriate.
set(QT_CREATOR_SKIP_MAINTENANCE_TOOL_PROVIDER ON CACHE BOOL
    "Avoid Qt Creator Maintenance Tool find_package provider" FORCE)
