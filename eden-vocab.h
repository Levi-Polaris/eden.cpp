#pragma once

#include "eden.h"

#include <string>
#include <vector>
#include <memory>

// pre-tokenization types
enum eden_vocab_pre_type {
    EDEN_VOCAB_PRE_TYPE_DEFAULT         = 0,
    EDEN_VOCAB_PRE_TYPE_LLAMA3          = 1,
    EDEN_VOCAB_PRE_TYPE_DEEPSEEK_LLM    = 2,
    EDEN_VOCAB_PRE_TYPE_DEEPSEEK_CODER  = 3,
    EDEN_VOCAB_PRE_TYPE_FALCON          = 4,
    EDEN_VOCAB_PRE_TYPE_MPT             = 5,
    EDEN_VOCAB_PRE_TYPE_STARCODER       = 6,
    EDEN_VOCAB_PRE_TYPE_GPT2            = 7,
    EDEN_VOCAB_PRE_TYPE_REFACT          = 8,
    EDEN_VOCAB_PRE_TYPE_COMMAND_R       = 9,
    EDEN_VOCAB_PRE_TYPE_STABLELM2       = 10,
    EDEN_VOCAB_PRE_TYPE_QWEN2           = 11,
    EDEN_VOCAB_PRE_TYPE_OLMO            = 12,
    EDEN_VOCAB_PRE_TYPE_DBRX            = 13,
    EDEN_VOCAB_PRE_TYPE_SMAUG           = 14,
    EDEN_VOCAB_PRE_TYPE_PORO            = 15,
    EDEN_VOCAB_PRE_TYPE_CHATGLM3        = 16,
    EDEN_VOCAB_PRE_TYPE_CHATGLM4        = 17,
    EDEN_VOCAB_PRE_TYPE_VIKING          = 18,
    EDEN_VOCAB_PRE_TYPE_JAIS            = 19,
    EDEN_VOCAB_PRE_TYPE_TEKKEN          = 20,
    EDEN_VOCAB_PRE_TYPE_SMOLLM          = 21,
    EDEN_VOCAB_PRE_TYPE_CODESHELL       = 22,
    EDEN_VOCAB_PRE_TYPE_BLOOM           = 23,
    EDEN_VOCAB_PRE_TYPE_GPT3_FINNISH    = 24,
    EDEN_VOCAB_PRE_TYPE_EXAONE          = 25,
    EDEN_VOCAB_PRE_TYPE_CHAMELEON       = 26,
    EDEN_VOCAB_PRE_TYPE_MINERVA         = 27,
    EDEN_VOCAB_PRE_TYPE_DEEPSEEK3_LLM   = 28,
    EDEN_VOCAB_PRE_TYPE_GPT4O           = 29,
    EDEN_VOCAB_PRE_TYPE_SUPERBPE        = 30,
    EDEN_VOCAB_PRE_TYPE_TRILLION        = 31,
    EDEN_VOCAB_PRE_TYPE_BAILINGMOE      = 32,
    EDEN_VOCAB_PRE_TYPE_LLAMA4          = 33,
    EDEN_VOCAB_PRE_TYPE_PIXTRAL         = 34,
    EDEN_VOCAB_PRE_TYPE_SEED_CODER      = 35,
    EDEN_VOCAB_PRE_TYPE_HUNYUAN         = 36,
    EDEN_VOCAB_PRE_TYPE_KIMI_K2         = 37,
    EDEN_VOCAB_PRE_TYPE_HUNYUAN_DENSE   = 38,
    EDEN_VOCAB_PRE_TYPE_GROK_2          = 39,
    EDEN_VOCAB_PRE_TYPE_GRANITE_DOCLING = 40,
    EDEN_VOCAB_PRE_TYPE_MINIMAX_M2      = 41,
    EDEN_VOCAB_PRE_TYPE_AFMOE           = 42,
    EDEN_VOCAB_PRE_TYPE_SOLAR_OPEN      = 43,
    EDEN_VOCAB_PRE_TYPE_YOUTU           = 44,
    EDEN_VOCAB_PRE_TYPE_EXAONE_MOE      = 45,
    EDEN_VOCAB_PRE_TYPE_QWEN35          = 46,
    EDEN_VOCAB_PRE_TYPE_TINY_AYA        = 47,
    EDEN_VOCAB_PRE_TYPE_JOYAI_LLM       = 48,
    EDEN_VOCAB_PRE_TYPE_JAIS2           = 49,
    EDEN_VOCAB_PRE_TYPE_GEMMA4          = 50,
    EDEN_VOCAB_PRE_TYPE_SARVAM_MOE      = 51,
};

struct LLM_KV;
struct eden_model_loader;

struct eden_vocab {
    struct token_data {
        std::string      text;
        float            score;
        eden_token_attr attr;
    };

    eden_vocab();
    ~eden_vocab();

    void load(eden_model_loader & ml, const LLM_KV & kv);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum eden_vocab_type     get_type()     const;
    enum eden_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (eden_token id) const;
    bool is_unknown     (eden_token id) const;
    bool is_control     (eden_token id) const;
    bool is_byte        (eden_token id) const;
    bool is_user_defined(eden_token id) const;
    bool is_unused      (eden_token id) const;
    bool is_eog         (eden_token id) const;

    uint8_t     token_to_byte(eden_token id) const;
    eden_token byte_to_token(uint8_t ch)     const;

    eden_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(eden_token id) const;

    const char *     token_get_text (eden_token id) const;
    float            token_get_score(eden_token id) const;
    eden_token_attr token_get_attr (eden_token id) const;

    eden_token token_bos() const;
    eden_token token_eos() const;
    eden_token token_eot() const;
    eden_token token_eom() const;
    eden_token token_unk() const;
    eden_token token_sep() const;
    eden_token token_nl () const;
    eden_token token_pad() const;
    eden_token token_mask() const;

    eden_token token_prefix() const;
    eden_token token_middle() const;
    eden_token token_suffix() const;

    eden_token token_fim_pre() const;
    eden_token token_fim_suf() const;
    eden_token token_fim_mid() const;
    eden_token token_fim_pad() const;
    eden_token token_fim_rep() const;
    eden_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  eden_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<eden_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  eden_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(eden_token token) const;

    int32_t detokenize(
            const eden_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<eden_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
