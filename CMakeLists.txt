# eden.cpp v1.1 — root CMakeLists.txt reconstruction (proposed fix)
# Found broken 2026-08-05 (Ranger): shipped root CMakeLists.txt is a
# byte-identical copy of src/CMakeLists.txt. This file restores the
# missing parent build entry (llama.cpp-era structure, EDEN_* convention).
# Applies at repo root as CMakeLists.txt. NOT pushed anywhere yet —
# decision pending Levi/Haven (Haven owns R&D on eden.cpp).
#
# Apply:
#   cp ~/.eden/docs/eden-cpp-root-CMakeLists.fixed.txt <repo>/CMakeLists.txt
# Then build per README:
#   mkdir build && cd build && cmake .. -DGGML_VULKAN=ON -DGGML_CUDA=OFF \
#     -DEDEN_BUILD_SERVER=ON && make -j$(nproc) eden-server

cmake_minimum_required(VERSION 3.14...3.28)

project("eden.cpp" C CXX)

# --- build info (commit/compiler/target baked into version.h) ---
include(cmake/build-info.cmake)

# --- eden helper macros (defines eden_add_compile_flags) ---
include(cmake/common.cmake)
include(cmake/license.cmake)

# --- version ---
set(EDEN_INSTALL_VERSION "1.1.0")

# --- options ---
option(EDEN_BUILD_SERVER "build eden-server (OpenAI-compatible server)" ON)
option(BUILD_SHARED_LIBS   "build shared libraries" OFF)

# --- subprojects ---
add_subdirectory(ggml)    # frozen ggml core (CPU + Vulkan/CUDA backends)
add_subdirectory(common)  # eden-common-base (build info, common helpers)
add_subdirectory(src)     # eden library (the frozen llama.cpp fork core)
add_subdirectory(vendor/cpp-httplib)  # vendored HTTP lib for eden-server
add_subdirectory(tools)   # eden-server, cli, quantize, perplexity, ...

# --- license embedding (bakes LICENSE into the eden-common library) ---
license_add_file("eden.cpp" "LICENSE")
license_generate(eden-common)

# --- install rules ---
include(GNUInstallDirs)

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS eden
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

# --- package config (eden-config.cmake.in) ---
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/eden-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/eden-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/eden
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/eden-config-version.cmake
    VERSION ${EDEN_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/eden-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/eden-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/eden
)
