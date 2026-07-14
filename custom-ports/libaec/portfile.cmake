vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Deutsches-Klimarechenzentrum/libaec
    REF v1.1.6
    SHA512 76df7501d1b7d91a43b525ba828f092f18d83f8ab09a9331e5758f93942a9758ad580baca8f9316b92a98639bde2e23cacbc2f33f52d0dd98ce7efe412cf43cd
    HEAD_REF master
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" BUILD_STATIC)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_STATIC_LIBS=${BUILD_STATIC}
        -Dlibaec_INSTALL_CMAKEDIR=share/${PORT}
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup()

if(EXISTS "${CURRENT_PACKAGES_DIR}/share/libaec/libaec-config.cmake")
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/libaec/libaec-config.cmake"
        "if(libaec_USE_STATIC_LIBS)"
        "if(\"${BUILD_STATIC}\") # forced by vcpkg"
    )
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

file(INSTALL "${SOURCE_PATH}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
