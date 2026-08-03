set(TARGET eden-perplexity)
add_executable(${TARGET} perplexity.cpp)
target_link_libraries(${TARGET} PRIVATE eden-common eden ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
