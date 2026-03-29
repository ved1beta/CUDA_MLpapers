/*
 * PyTorch C++ extension bindings for Engram CUDA kernels.
 * Exposes three operations:
 *   - multi_head_ngram_lookup: fused hash + embedding gather
 *   - fused_engram_gate: fused K-projection + gating + V-scaling
 *   - sparse_embed_backward: gradient scatter-add to embedding table
 */

#include "include/engram.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Engram CUDA kernels for conditional memory via N-gram lookup";

    m.def(
        "multi_head_ngram_lookup",
        &multi_head_ngram_lookup,
        "Multi-head N-gram hash computation fused with embedding gather",
        py::arg("compressed_ids"),
        py::arg("embed_table"),
        py::arg("multipliers"),
        py::arg("head_primes"),
        py::arg("head_offsets"),
        py::arg("max_ngram_size"),
        py::arg("n_heads_per_ngram"),
        py::arg("pad_id")
    );

    m.def(
        "fused_engram_gate",
        &fused_engram_gate,
        "Fused gating: K-projection + RMSNorm + dot + sqrt-abs-sigmoid + V-scaling",
        py::arg("e_t"),
        py::arg("h_t"),
        py::arg("W_K"),
        py::arg("W_V"),
        py::arg("norm_q_weight"),
        py::arg("norm_k_weight"),
        py::arg("inv_sqrt_d"),
        py::arg("norm_eps") = 1e-5f
    );

    m.def(
        "sparse_embed_backward",
        &sparse_embed_backward,
        "Sparse gradient scatter-add to embedding table",
        py::arg("grad_output"),
        py::arg("indices"),
        py::arg("total_vocab_size")
    );
}
