#pragma once
#include "eden.h"  // eden.cpp public API types (eden_token, EDEN_API, etc.)
#ifdef EDEN_EDEN_INTEGRATION
#include <string>
#include <cstdint>
bool eden_init();
void eden_publish_inference_start(const std::string& model_id, uint64_t request_id, int32_t prompt_tokens);
void eden_publish_inference_complete(const std::string& model_id, uint64_t request_id, int32_t prompt_tokens, int32_t completion_tokens, int32_t elapsed_ms, float tok_per_sec);
#endif
