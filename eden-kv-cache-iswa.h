#pragma once

#include "eden-kv-cache.h"

#include <vector>

//
// eden_kv_cache_iswa
//

// utilizes two instances of eden_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class eden_kv_cache_iswa : public eden_memory_i {
public:
    eden_kv_cache_iswa(
            const eden_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~eden_kv_cache_iswa() = default;

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
    void state_read (eden_io_read_i  & io, eden_seq_id seq_id = -1, eden_state_seq_flags flags = 0) override;

    //
    // eden_kv_cache_iswa specific API
    //

    eden_kv_cache * get_base() const;
    eden_kv_cache * get_swa () const;

private:
    const eden_hparams & hparams;

    const bool unified;

    std::unique_ptr<eden_kv_cache> kv_base;
    std::unique_ptr<eden_kv_cache> kv_swa;
};

class eden_kv_cache_iswa_context : public eden_memory_context_i {
public:
    using slot_info_vec_t = eden_kv_cache::slot_info_vec_t;

    // used for errors
    eden_kv_cache_iswa_context(eden_memory_status status);

    // used to create a full-cache context
    eden_kv_cache_iswa_context(
            eden_kv_cache_iswa * kv);

    // used to create an update context
    eden_kv_cache_iswa_context(
            eden_kv_cache_iswa * kv,
            eden_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    eden_kv_cache_iswa_context(
            eden_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<eden_ubatch> ubatches);

    virtual ~eden_kv_cache_iswa_context();

    //
    // eden_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    eden_memory_status  get_status() const override;
    const eden_ubatch & get_ubatch() const override;

    //
    // eden_kv_cache_iswa_context specific API
    //

    const eden_kv_cache_context * get_base() const;
    const eden_kv_cache_context * get_swa()  const;

private:
    //eden_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<eden_ubatch> ubatches;

    const eden_memory_context_ptr ctx_base;
    const eden_memory_context_ptr ctx_swa;

    const eden_memory_status status;
};
