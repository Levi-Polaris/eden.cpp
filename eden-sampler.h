#pragma once

#include "eden.h"

#include <vector>

struct eden_vocab;
struct eden_grammar;

// sampler chain

struct eden_sampler_chain {
    eden_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        eden_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for eden_sampler_sample to avoid repeated allocations
    std::vector<eden_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct eden_sampler * eden_sampler_init_dry_testing(
        int32_t context_size,
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<eden_token>> & seq_breakers);
