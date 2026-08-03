#pragma once

#include "eden-model.h"
#include "eden-graph.h"
#include "eden-model-loader.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>

//
// base classes
//

struct llm_build_mamba_base : public llm_graph_context {
    llm_build_mamba_base(const llm_graph_params & params);

    virtual ~llm_build_mamba_base() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const eden_model & model, const eden_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const eden_model & model, const eden_ubatch & ubatch, int il) const;

};

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // fused op with keep_intermediates=true: returns the raw [attn | T snapshots]
    // output tensor. Caller slices snapshot views and routes them to recurrent slots.
    ggml_tensor * build_delta_net_fused_keep_intermediates(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // choose one of two implementations above based on the number of tokens
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);
};

struct llm_build_rwkv6_base : public llm_graph_context {
    const eden_model & model;

    llm_build_rwkv6_base(const eden_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv6_base() = default;

    ggml_tensor * build_rwkv6_channel_mix(const eden_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;

    ggml_tensor * build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       const eden_ubatch & ubatch,
                                       int                  il) const;
};

// Base class for RWKV7-related models
struct llm_build_rwkv7_base : public llm_graph_context {
    const eden_model & model;

    llm_build_rwkv7_base(const eden_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv7_base() = default;

    // RWKV7-specific graph building methods
    ggml_tensor * build_rwkv7_channel_mix(const eden_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;
    ggml_tensor * build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       ggml_tensor *&       first_layer_value,
                                       const eden_ubatch & ubatch,
                                       int                  il) const;
};

//
// models
//

struct eden_model_eden : public eden_model_base {
    eden_model_eden(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool embed>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_eden4 : public eden_model_base {
    eden_model_eden4(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_eden_embed : public eden_model_eden {
    eden_model_eden_embed(const struct eden_model_params & params) : eden_model_eden(params) {}
    // reuse load_arch_hparams and load_arch_tensors from eden_model_eden

    template <bool embed>
    using graph = eden_model_eden::graph<embed>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_maincoder : public eden_model_base {
    eden_model_maincoder(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_deci : public eden_model_base {
    eden_model_deci(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_baichuan : public eden_model_base {
    eden_model_baichuan(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_falcon : public eden_model_base {
    eden_model_falcon(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_grok : public eden_model_base {
    eden_model_grok(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_starcoder : public eden_model_base {
    eden_model_starcoder(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_refact : public eden_model_base {
    eden_model_refact(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_bert : public eden_model_base {
    eden_model_bert(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_jina_bert_v2 : public eden_model_base {
    eden_model_jina_bert_v2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_jina_bert_v3 : public eden_model_base {
    eden_model_jina_bert_v3(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_nomic_bert : public eden_model_base {
    eden_model_nomic_bert(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_nomic_bert_moe : public eden_model_base {
    eden_model_nomic_bert_moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_modern_bert : public eden_model_base {
    eden_model_modern_bert(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_neo_bert : public eden_model_base {
    eden_model_neo_bert(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_eurobert : public eden_model_base {
    eden_model_eurobert(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_bloom : public eden_model_base {
    eden_model_bloom(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_mpt : public eden_model_base {
    eden_model_mpt(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_stablelm : public eden_model_base {
    eden_model_stablelm(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen : public eden_model_base {
    eden_model_qwen(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen2 : public eden_model_base {
    eden_model_qwen2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_dream : public eden_model_base {
    eden_model_dream(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_llada : public eden_model_base {
    eden_model_llada(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_llada_moe : public eden_model_base {
    eden_model_llada_moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_rnd1 : public eden_model_base {
    eden_model_rnd1(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen2vl : public eden_model_base {
    eden_model_qwen2vl(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen2moe : public eden_model_base {
    eden_model_qwen2moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen3 : public eden_model_base {
    eden_model_qwen3(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen3moe : public eden_model_base {
    eden_model_qwen3moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen3vl : public eden_model_base {
    eden_model_qwen3vl(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen3vlmoe : public eden_model_base {
    eden_model_qwen3vlmoe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_phi2 : public eden_model_base {
    eden_model_phi2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_phi3 : public eden_model_base {
    eden_model_phi3(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_phimoe : public eden_model_base {
    eden_model_phimoe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    using graph = eden_model_phi3::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_plamo : public eden_model_base {
    eden_model_plamo(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_plamo2 : public eden_model_base {
    eden_model_plamo2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const eden_model & model, const llm_graph_params & params);
        private:
            ggml_tensor * build_plamo2_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const eden_model & model, const eden_ubatch & ubatch, int il);
            ggml_tensor * build_plamo2_attn_layer(llm_graph_input_attn_kv * inp, ggml_tensor * inp_pos, ggml_tensor * cur,
                                                    const eden_model & model, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_plamo3 : public eden_model_base {
    eden_model_plamo3(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_gpt2 : public eden_model_base {
    eden_model_gpt2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_codeshell : public eden_model_base {
    eden_model_codeshell(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_orion : public eden_model_base {
    eden_model_orion(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_internlm2 : public eden_model_base {
    eden_model_internlm2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_minicpm3 : public eden_model_base {
    eden_model_minicpm3(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_gemma : public eden_model_base {
    eden_model_gemma(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_gemma2 : public eden_model_base {
    eden_model_gemma2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_gemma3 : public eden_model_base {
    eden_model_gemma3(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_gemma3n : public eden_model_base {
    eden_model_gemma3n(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const eden_model & model;

        const int64_t n_embd_head;
        const int64_t n_embd_altup;
        const int64_t n_altup;
        const int     i_altup_act;
        const int     n_layer_sparsity = 10; // number of layers using activation sparsity
        const float   f_sparsity_std_mul = 1.6448533535003662f; // std_multiplier = normal_dist.icdf(0.95)

        graph(const eden_model & model, const llm_graph_params & params);
        ggml_tensor * calc_magnitude(ggml_tensor * x);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);

        ggml_tensor * gaussian_topk(ggml_tensor * x);
        ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il);
        ggml_tensor * altup_predict(ggml_tensor * cur, int il);
        ggml_tensor * laurel(ggml_tensor * cur, int il);
        ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_gemma4 : public eden_model_base {
    eden_model_gemma4(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const eden_model & model;

        const int64_t n_embd_per_layer;

        graph(const eden_model & model, const llm_graph_params & params);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_gemma_embedding : public eden_model_base {
    eden_model_gemma_embedding(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_starcoder2 : public eden_model_base {
    eden_model_starcoder2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_mamba : public eden_model_base {
    eden_model_mamba(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_mamba2 : public eden_model_base {
    eden_model_mamba2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_mamba::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_jamba : public eden_model_base {
    eden_model_jamba(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_xverse : public eden_model_base {
    eden_model_xverse(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_command_r : public eden_model_base {
    eden_model_command_r(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_cohere2 : public eden_model_base {
    eden_model_cohere2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_dbrx : public eden_model_base {
    eden_model_dbrx(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_olmo : public eden_model_base {
    eden_model_olmo(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_olmo2 : public eden_model_base {
    eden_model_olmo2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_olmoe : public eden_model_base {
    eden_model_olmoe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_openelm : public eden_model_base {
    eden_model_openelm(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_gptneox : public eden_model_base {
    eden_model_gptneox(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_arctic : public eden_model_base {
    eden_model_arctic(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_deepseek : public eden_model_base {
    eden_model_deepseek(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_deepseek2 : public eden_model_base {
    eden_model_deepseek2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_deepseek2ocr : public eden_model_base {
    eden_model_deepseek2ocr(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_glm_dsa : public eden_model_base {
    eden_model_glm_dsa(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_mistral4 : public eden_model_deepseek2 {
    eden_model_mistral4(const struct eden_model_params & params) : eden_model_deepseek2(params) {}
    // reuse load_arch_hparams and load_arch_tensors from eden_model_deepseek2

    using graph = eden_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_chatglm : public eden_model_base {
    eden_model_chatglm(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_glm4 : public eden_model_base {
    eden_model_glm4(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_glm4_moe : public eden_model_base {
    eden_model_glm4_moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_bitnet : public eden_model_base {
    eden_model_bitnet(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_t5 : public eden_model_base {
    eden_model_t5(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_t5encoder : public eden_model_base {
    eden_model_t5encoder(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_t5::graph<true>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_jais : public eden_model_base {
    eden_model_jais(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_jais2 : public eden_model_base {
    eden_model_jais2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_nemotron : public eden_model_base {
    eden_model_nemotron(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_nemotron_h : public eden_model_base {
    eden_model_nemotron_h(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const eden_model & model, const llm_graph_params & params);
        ggml_tensor * build_ffn_layer(ggml_tensor * cur, const eden_model & model, int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn,
            const eden_model & model, int64_t n_embd_head, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_nemotron_h_moe : public eden_model_nemotron_h {
    eden_model_nemotron_h_moe(const struct eden_model_params & params) : eden_model_nemotron_h(params) {}
    // reuse load_arch_hparams and load_arch_tensors from eden_model_nemotron_h

    using graph = eden_model_nemotron_h::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_exaone : public eden_model_base {
    eden_model_exaone(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_exaone4 : public eden_model_base {
    eden_model_exaone4(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_exaone_moe : public eden_model_base {
    eden_model_exaone_moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_rwkv6 : public eden_model_base {
    eden_model_rwkv6(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_rwkv6_base {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_rwkv6qwen2 : public eden_model_base {
    eden_model_rwkv6qwen2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_rwkv6_base {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_rwkv7 : public eden_model_base {
    eden_model_rwkv7(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_rwkv7_base {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_arwkv7 : public eden_model_base {
    eden_model_arwkv7(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_rwkv7_base {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_granite : public eden_model_base {
    eden_model_granite(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);

    private:
        ggml_tensor * build_attention_layer(
                  ggml_tensor             * cur,
                  ggml_tensor             * inp_pos,
                  llm_graph_input_attn_kv * inp_attn,
            const eden_model             & model,
            const int64_t                 n_embd_head,
            const int                     il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor       * cur,
                  ggml_tensor       * inpSA,
            const eden_model       & model,
            const int                 il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_granite_moe : public eden_model_base {
    eden_model_granite_moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_granite::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_minicpm : public eden_model_base {
    eden_model_minicpm(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    using graph = eden_model_granite::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_granite_hybrid : public eden_model_base {
    eden_model_granite_hybrid(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const eden_model & model, const llm_graph_params & params);
        ggml_tensor * build_layer_ffn(ggml_tensor * cur, ggml_tensor * inpSA, const eden_model & model, const int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn,
            const eden_model & model,const int64_t n_embd_head, const int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_chameleon : public eden_model_base {
    eden_model_chameleon(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_wavtokenizer_dec : public eden_model_base {
    eden_model_wavtokenizer_dec(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_plm : public eden_model_base {
    eden_model_plm(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_bailingmoe : public eden_model_base {
    eden_model_bailingmoe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_bailingmoe2 : public eden_model_base {
    eden_model_bailingmoe2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_seed_oss : public eden_model_base {
    eden_model_seed_oss(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_dots1 : public eden_model_base {
    eden_model_dots1(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_arcee : public eden_model_base {
    eden_model_arcee(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_afmoe : public eden_model_base {
    eden_model_afmoe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_ernie4_5 : public eden_model_base {
    eden_model_ernie4_5(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_ernie4_5_moe : public eden_model_ernie4_5 {
    eden_model_ernie4_5_moe(const struct eden_model_params & params) : eden_model_ernie4_5(params) {}
    // reuse load_arch_hparams and load_arch_tensors from eden_model_ernie4_5

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_paddleocr : public eden_model_ernie4_5 {
    eden_model_paddleocr(const struct eden_model_params & params) : eden_model_ernie4_5(params) {}
    // reuse load_arch_hparams and load_arch_tensors from eden_model_ernie4_5

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_hunyuan_moe : public eden_model_base {
    eden_model_hunyuan_moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_hunyuan_vl : public eden_model_base {
    eden_model_hunyuan_vl(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_hunyuan_dense : public eden_model_hunyuan_vl {
    eden_model_hunyuan_dense(const struct eden_model_params & params) : eden_model_hunyuan_vl(params) {}
    // reuse load_arch_hparams and load_arch_tensors from eden_model_hunyuan_vl

    using graph = eden_model_hunyuan_vl::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_smollm3 : public eden_model_base {
    eden_model_smollm3(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_openai_moe : public eden_model_base {
    eden_model_openai_moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_falcon_h1 : public eden_model_base {
    eden_model_falcon_h1(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_lfm2 : public eden_model_base {
    eden_model_lfm2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_lfm2moe : public eden_model_base {
    eden_model_lfm2moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    using graph = eden_model_lfm2::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_smallthinker : public eden_model_base {
    eden_model_smallthinker(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_grovemoe : public eden_model_base {
    eden_model_grovemoe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_apertus : public eden_model_base {
    eden_model_apertus(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_minimax_m2 : public eden_model_base {
    eden_model_minimax_m2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_cogvlm : public eden_model_base {
    eden_model_cogvlm(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_pangu_embed : public eden_model_base {
    eden_model_pangu_embed(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen3next : public eden_model_base {
    eden_model_qwen3next(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const eden_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const eden_model & model;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen35 : public eden_model_base {
    eden_model_qwen35(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const eden_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const eden_model & model;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen35moe : public eden_model_base {
    eden_model_qwen35moe(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const eden_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const eden_model & model;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen35_mtp : public eden_model_base {
    eden_model_qwen35_mtp(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;
    void link_shared_tensors(const eden_model * main_model) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_qwen35moe_mtp : public eden_model_base {
    eden_model_qwen35moe_mtp(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;
    void link_shared_tensors(const eden_model * main_model) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_mistral3 : public eden_model_base {
    eden_model_mistral3(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_mimo2 : public eden_model_base {
    eden_model_mimo2(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_kimi_linear : public eden_model_base {
    eden_model_kimi_linear(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const eden_model & model, const llm_graph_params & params);

        std::pair<ggml_tensor *, ggml_tensor *> build_kda_autoregressive(
                    ggml_tensor * q,
                    ggml_tensor * k,
                    ggml_tensor * v,
                    ggml_tensor * gk,
                    ggml_tensor * beta,
                    ggml_tensor * state,
                            int   il);

        std::pair<ggml_tensor *, ggml_tensor *> build_kda_chunking(
                    ggml_tensor * q,
                    ggml_tensor * k,
                    ggml_tensor * v,
                    ggml_tensor * gk,
                    ggml_tensor * beta,
                    ggml_tensor * state,
                    ggml_tensor * causal_mask,
                    ggml_tensor * identity,
                    ggml_tensor * diag_mask,
                            int   il);

        const eden_model & model;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct eden_model_step35 : public eden_model_base {
    eden_model_step35(const struct eden_model_params & params) : eden_model_base(params) {}
    void load_arch_hparams(eden_model_loader & ml) override;
    void load_arch_tensors(eden_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const eden_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
