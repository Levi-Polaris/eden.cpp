set(TARGET eden-quantize)
add_executable(${TARGET} quantize.cpp)
target_link_libraries(${TARGET} PRIVATE eden-common eden ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ../../common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(EDEN_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
