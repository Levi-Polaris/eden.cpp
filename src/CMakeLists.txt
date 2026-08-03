eden_add_compile_flags()

#
# libraries
#

# eden

file(GLOB EDEN_MODELS_SOURCES "models/*.cpp")

add_library(eden
            ../include/eden.h
            eden-impl.cpp
            llama.cpp
            eden-adapter.cpp
            eden-arch.cpp
            eden-batch.cpp
            eden-chat.cpp
            eden-context.cpp
            eden-cparams.cpp
            eden-grammar.cpp
            eden-graph.cpp
            eden-hparams.cpp
            eden-impl.cpp
            llama.cpp
            eden-io.cpp
            eden-kv-cache.cpp
            eden-kv-cache-iswa.cpp
            eden-memory.cpp
            eden-memory-hybrid.cpp
            eden-memory-hybrid-iswa.cpp
            eden-memory-recurrent.cpp
            eden-mmap.cpp
            eden-model-loader.cpp
            eden-model-saver.cpp
            eden-model.cpp
            eden-quant.cpp
            eden-sampler.cpp
            eden-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${EDEN_MODELS_SOURCES}
            )

set_target_properties(eden PROPERTIES
    VERSION ${EDEN_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(eden PRIVATE .)
target_include_directories(eden PUBLIC ../include)
target_compile_features   (eden PRIVATE cxx_std_17) # don't bump

target_link_libraries(eden PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(eden PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(eden PRIVATE EDEN_BUILD)
    target_compile_definitions(eden PUBLIC  EDEN_SHARED)
endif()
