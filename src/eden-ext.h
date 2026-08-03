#pragma once

// this is a staging header for new eden.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP

#include "eden.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to eden_graph_reserve.
EDEN_API struct ggml_cgraph * eden_graph_reserve(
        struct eden_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
EDEN_API ggml_type eden_ftype_get_default_type(eden_ftype ftype);

struct quantize_state_impl;

EDEN_API quantize_state_impl * eden_quant_init(
        const eden_model * model,
        const eden_model_quantize_params * params);

EDEN_API void eden_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct eden_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with eden_model_free().
EDEN_API eden_model * eden_quant_model_from_metadata(const eden_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
EDEN_API bool eden_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use eden_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
EDEN_API void eden_quant_compute_types(
        quantize_state_impl * qs,
        eden_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct eden_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct eden_device_memory_data {
    int64_t total;
    int64_t free;
    eden_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using eden_memory_breakdown = std::map<ggml_backend_buffer_type_t, eden_memory_breakdown_data>;

EDEN_API int32_t eden_model_n_expert (const struct eden_model * model);
EDEN_API int32_t eden_model_n_devices(const struct eden_model * model);

EDEN_API ggml_backend_dev_t eden_model_get_device(const struct eden_model * model, int i);

EDEN_API eden_memory_breakdown eden_get_memory_breakdown(const struct eden_context * ctx);
