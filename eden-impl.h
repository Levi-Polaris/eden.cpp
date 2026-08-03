#pragma once

#include "ggml.h" // for ggml_log_level

#include <string>
#include <vector>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define EDEN_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define EDEN_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define EDEN_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

EDEN_ATTRIBUTE_FORMAT(2, 3)
void eden_log_internal        (ggml_log_level level, const char * format, ...);
void eden_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define EDEN_LOG(...)       eden_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define EDEN_LOG_INFO(...)  eden_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define EDEN_LOG_WARN(...)  eden_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define EDEN_LOG_ERROR(...) eden_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define EDEN_LOG_DEBUG(...) eden_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define EDEN_LOG_CONT(...)  eden_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() = default;
};

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to eden_format ?
EDEN_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string eden_format_tensor_shape(const std::vector<int64_t> & ne);
std::string eden_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define EDEN_TENSOR_NAME_FATTN   "__fattn__"
#define EDEN_TENSOR_NAME_FGDN_AR "__fgdn_ar__"
#define EDEN_TENSOR_NAME_FGDN_CH "__fgdn_ch__"
