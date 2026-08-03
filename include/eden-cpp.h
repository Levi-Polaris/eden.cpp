#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "eden.h"

struct eden_model_deleter {
    void operator()(eden_model * model) { eden_model_free(model); }
};

struct eden_context_deleter {
    void operator()(eden_context * context) { eden_free(context); }
};

struct eden_sampler_deleter {
    void operator()(eden_sampler * sampler) { eden_sampler_free(sampler); }
};

struct eden_adapter_lora_deleter {
    void operator()(eden_adapter_lora * adapter) { eden_adapter_lora_free(adapter); }
};

typedef std::unique_ptr<eden_model, eden_model_deleter> eden_model_ptr;
typedef std::unique_ptr<eden_context, eden_context_deleter> eden_context_ptr;
typedef std::unique_ptr<eden_sampler, eden_sampler_deleter> eden_sampler_ptr;
typedef std::unique_ptr<eden_adapter_lora, eden_adapter_lora_deleter> eden_adapter_lora_ptr;
