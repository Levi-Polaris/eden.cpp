#pragma once

#include "eden-batch.h"
#include "eden-graph.h"
#include "eden-kv-cache-iswa.h"
#include "eden-memory.h"
#include "eden-memory-recurrent.h"

#include <memory>
#include <vector>

//
// eden_memory_hybrid_iswa
//

// utilizes instances of eden_memory_recurrent and eden_kv_cache_iswa to
//   support models where each layer may be either attention-based (with SWA support) or recurrent

class eden_memory_hybrid_iswa : public eden_memory_i {
public:
    eden_memory_hybrid_iswa(
        const eden_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    ~eden_memory_hybrid_iswa() = default;

    //
    // eden_memory_i
    //

    eden_memory_context_ptr init_batch(
            eden_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    eden_memory_context_ptr init_full() override;

    eden_memory_context_ptr init_update(eden_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (eden_seq_id seq_id,                              eden_pos p0, eden_pos p1) override;
    void seq_cp  (eden_seq_id seq_id_src, eden_seq_id seq_id_dst, eden_pos p0, eden_pos p1) override;
    void seq_keep(eden_seq_id seq_id)                                                          override;
    void seq_add (eden_seq_id seq_id,                              eden_pos p0, eden_pos p1, eden_pos shift) override;
    void seq_div (eden_seq_id seq_id,                              eden_pos p0, eden_pos p1, int d) override;

    eden_pos seq_pos_min(eden_seq_id seq_id) const override;
    eden_pos seq_pos_max(eden_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(eden_io_write_i & io, eden_seq_id seq_id = -1, eden_state_seq_flags flags = 0) const override;
    void state_read (eden_io_read_i  & io, eden_seq_id seq_id = -1, eden_state_seq_flags flags = 0)       override;

    //
    // eden_memory_hybrid_iswa specific API
    //

    eden_kv_cache_iswa * get_mem_attn() const;
    eden_memory_recurrent * get_mem_recr() const;

private:
    const eden_hparams & hparams;

    const std::unique_ptr<eden_kv_cache_iswa> mem_attn;
    const std::unique_ptr<eden_memory_recurrent> mem_recr;
};

class eden_memory_hybrid_iswa_context : public eden_memory_context_i {
public:
    using slot_info_vec_t = eden_kv_cache::slot_info_vec_t;

    // init failure
    explicit eden_memory_hybrid_iswa_context(eden_memory_status status);

    // init full
    explicit eden_memory_hybrid_iswa_context(eden_memory_hybrid_iswa * mem);

    // init update
    explicit eden_memory_hybrid_iswa_context(
        eden_memory_hybrid_iswa * mem,
                   eden_context * lctx,
                            bool   optimize);

    // init success
    eden_memory_hybrid_iswa_context(
           eden_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<eden_ubatch>   ubatches);

    ~eden_memory_hybrid_iswa_context() = default;

    bool next()  override;
    bool apply() override;

    eden_memory_status  get_status() const override;
    const eden_ubatch & get_ubatch() const override;

    //
    // eden_memory_hybrid_iswa_context
    //

    const eden_kv_cache_iswa_context * get_attn() const;
    const eden_memory_recurrent_context * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<eden_ubatch> ubatches;

    const eden_memory_context_ptr ctx_attn;
    const eden_memory_context_ptr ctx_recr;

    const eden_memory_status status;
};
