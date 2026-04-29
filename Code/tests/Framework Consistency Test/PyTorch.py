import os, time, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MLIR_CRASH_REPRODUCER_DIRECTORY'] = 'disabled'

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

TEST_CONFIGS = [
    {"name": "LLaMA-7B (LoRA r=8)", "dense_size": 4_200_000,  "cr": 0.10},
    {"name": "ResNet-18 (Full)",     "dense_size": 11_180_000, "cr": 0.10},
    {"name": "ResNet-50 (Full)",     "dense_size": 25_600_000, "cr": 0.10},
    {"name": "ViT-Base (Full)",      "dense_size": 86_000_000, "cr": 0.10},
]
NUM_TRIALS, WARMUP = 100, 15

print("=" * 115)
print(" Rigorous PyTorch Architecture End-to-End Latency Breakdown (Aligned with Papers) [Corrected]")
print("=" * 115)

if not torch.cuda.is_available():
    raise RuntimeError("Please run this benchmark in a GPU environment.")
device = torch.device("cuda:0")

# ── C++ extension: proper Rice coding (bit-level) + serial decoder ──────────
print("Compiling C++ Rice coding operator (STC)...")
cpp_source = """
#include <torch/extension.h>
#include <cmath>

// Rice encode: gaps (int64) → packed bitstream (uint8)
// k: Rice parameter (M = 2^k)
torch::Tensor encode_rice(torch::Tensor gaps_tensor, int k) {
    auto gaps = gaps_tensor.contiguous();
    int64_t n  = gaps.size(0);
    auto gp    = gaps.data_ptr<int64_t>();
    int64_t mask = (1LL << k) - 1;

    // Count total bits needed
    int64_t total_bits = 0;
    for (int64_t i = 0; i < n; i++) total_bits += (gp[i] >> k) + 1 + k;
    int64_t total_bytes = (total_bits + 7) / 8;

    auto out = torch::zeros({total_bytes}, torch::dtype(torch::kUInt8).device(torch::kCPU));
    auto op  = out.data_ptr<uint8_t>();
    int64_t bp = 0;

    for (int64_t i = 0; i < n; i++) {
        int64_t g = gp[i];
        int64_t q = g >> k;
        int64_t r = g & mask;
        // Unary: q ones
        for (int64_t j = 0; j < q; j++) {
            op[bp >> 3] |= (uint8_t)(1 << (7 - (bp & 7)));
            ++bp;
        }
        ++bp;  // terminating 0
        // Remainder: k bits MSB-first
        for (int kk = k - 1; kk >= 0; --kk) {
            if ((r >> kk) & 1) op[bp >> 3] |= (uint8_t)(1 << (7 - (bp & 7)));
            ++bp;
        }
    }
    return out;
}

// Rice decode: packed bitstream → original indices (serial)
torch::Tensor decode_rice(torch::Tensor packed_tensor, int n_values, int k) {
    auto pb  = packed_tensor.contiguous();
    auto pp  = pb.data_ptr<uint8_t>();
    auto out = torch::empty({n_values}, torch::dtype(torch::kInt64).device(torch::kCPU));
    auto op  = out.data_ptr<int64_t>();
    int64_t bp = 0, current = -1;

    for (int i = 0; i < n_values; i++) {
        // Read unary quotient
        int64_t q = 0;
        while ((pp[bp >> 3] >> (7 - (bp & 7))) & 1) { ++q; ++bp; }
        ++bp;  // skip terminating 0
        // Read k-bit remainder
        int64_t r = 0;
        for (int kk = 0; kk < k; ++kk)
            r = (r << 1) | ((pp[bp >> 3] >> (7 - (bp & 7))) & 1), ++bp;
        int64_t gap = (q << k) | r;
        current += gap + 1;
        op[i] = current;
    }
    return out;
}
"""
stc_cpp_module = load_inline(
    name='stc_rice_extension',
    cpp_sources=cpp_source,
    functions=['encode_rice', 'decode_rice'],
    verbose=False
)
print("C++ operator compilation completed!\n")

POWERS_PT = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device)
benchmark_results = []

for config in TEST_CONFIGS:
    DENSE_SIZE = config["dense_size"]
    CR         = config["cr"]
    K_SIZE     = max(1, int(DENSE_SIZE * CR))
    SKETCH_SIZE = int(np.log2(DENSE_SIZE) * 100)
    pad_rem    = DENSE_SIZE % 8
    PAD_LEN    = (8 - pad_rem) if pad_rem > 0 else 0

    # Optimal Rice k for this sparsity (geometric distribution, Golomb M ≈ ceil(log(2-p)/(-log(1-p))))
    golomb_M = math.ceil(math.log(2 - CR) / (-math.log(1 - CR)))
    k_rice   = max(1, round(math.log2(golomb_M)))  # nearest power-of-2 exponent

    print(f"\nScenario: {config['name']} | Params: {DENSE_SIZE/1e6:.2f}M | "
          f"Non-zeros: {K_SIZE} (Sparsity {CR*100:.0f}%) | Rice k={k_rice}")

    pt_dense  = torch.rand(DENSE_SIZE, dtype=torch.float32, device=device)
    pt_idxs   = torch.sort(torch.randperm(DENSE_SIZE, device=device)[:K_SIZE])[0].to(torch.int64)
    pt_vals   = pt_dense[pt_idxs]

    NUM_ROWS = 5
    pt_hash_idxs_multi  = torch.randint(0, SKETCH_SIZE, (NUM_ROWS, DENSE_SIZE), dtype=torch.int64, device=device)
    pt_hash_signs_multi = torch.randint(0, 2, (NUM_ROWS, DENSE_SIZE), dtype=torch.float32, device=device) * 2 - 1

    pt_alpha    = torch.tensor(0.01, dtype=torch.float32, device=device)
    pt_frozen_s = torch.randint(0, 15, (DENSE_SIZE,), dtype=torch.float32, device=device)
    bit_index   = 2
    QSGD_LEVELS = 255.0

    # ── Algorithm implementations ────────────────────────────────────────────

    # 1. Ours (GPU Bitmap) — BiSARC bit-level packing
    def pack_ours(vals, idxs):
        mask = torch.zeros(DENSE_SIZE, dtype=torch.bool, device=device)
        mask[idxs] = True
        if PAD_LEN > 0:
            mask = torch.nn.functional.pad(mask, (0, PAD_LEN))
        packed_mask = (mask.view(-1, 8).to(torch.uint8) * POWERS_PT).sum(dim=1).to(torch.uint8)
        return packed_mask, vals

    def unpack_ours(packed):
        p_mask, v = packed
        unmask = (torch.bitwise_and(p_mask.unsqueeze(1), POWERS_PT) > 0).view(-1)[:DENSE_SIZE]
        dense = torch.zeros(DENSE_SIZE, dtype=torch.float32, device=device)
        dense[unmask] = v
        return dense

    # 2. Std Top-K (baseline, no compression)
    def pack_std(vals, idxs):  return vals, idxs

    def unpack_std(packed):
        v, i = packed
        dense = torch.zeros(DENSE_SIZE, dtype=torch.float32, device=device)
        dense.scatter_(0, i, v)
        return dense

    # 3. FedBiF (1-bit sign bitmap + frozen-state reconstruction)
    def pack_fedbif_strict(dense):
        activated_bits = (dense > 0)
        if PAD_LEN > 0:
            activated_bits = torch.nn.functional.pad(activated_bits, (0, PAD_LEN))
        packed_bits = (activated_bits.view(-1, 8).to(torch.uint8) * POWERS_PT).sum(dim=1).to(torch.uint8)
        return packed_bits

    def unpack_fedbif_strict(packed_bits):
        unbits = (torch.bitwise_and(packed_bits.unsqueeze(1), POWERS_PT) > 0).view(-1)[:DENSE_SIZE]
        recovered = pt_alpha * ((2 ** bit_index) * unbits.to(torch.float32) + pt_frozen_s)
        return recovered

    # 4. FedPAQ — L2-norm stochastic quantization (Reisizadeh et al. AISTATS 2020)
    #    Uses the same unbiased quantizer as QSGD; dense int16 storage (all d elements).
    #    CORRECTED: original code used min-max normalization which does not match the paper.
    def pack_fedpaq_strict(dense):
        l2_norm  = torch.norm(dense, p=2) + 1e-7
        abs_scaled = dense.abs() * 255.0 / l2_norm
        floor_val  = abs_scaled.floor()
        prob       = abs_scaled - floor_val
        q_val      = (floor_val + torch.bernoulli(prob)) * torch.sign(dense)
        return q_val.to(torch.int16), l2_norm

    def unpack_fedpaq_strict(packed):
        q, l2_norm = packed
        return q.to(torch.float32) * l2_norm / 255.0

    # 5. FetchSGD (Count Sketch, multi-row median reconstruction)
    def pack_fetchsgd_strict(dense):
        sketches = torch.zeros((NUM_ROWS, SKETCH_SIZE), dtype=torch.float32, device=device)
        for r in range(NUM_ROWS):
            sketches[r].scatter_add_(0, pt_hash_idxs_multi[r], dense * pt_hash_signs_multi[r])
        return sketches

    def unpack_fetchsgd_strict(sketches):
        estimates = torch.zeros((NUM_ROWS, DENSE_SIZE), dtype=torch.float32, device=device)
        for r in range(NUM_ROWS):
            estimates[r] = sketches[r][pt_hash_idxs_multi[r]] * pt_hash_signs_multi[r]
        dense_median, _ = torch.median(estimates, dim=0)
        return dense_median

    # 6. QSGD — L2-norm stochastic quantization, sparse non-zero storage
    #    (Alistarh et al. NeurIPS 2017; Elias coding approximated by int16 + int64 index)
    def pack_qsgd_strict(dense):
        l2_norm    = torch.norm(dense, p=2) + 1e-7
        abs_scaled = dense.abs() * QSGD_LEVELS / l2_norm
        floor_val  = abs_scaled.floor()
        prob       = abs_scaled - floor_val
        q_val      = (floor_val + torch.bernoulli(prob)) * torch.sign(dense)
        non_zero_mask = q_val != 0
        idxs = non_zero_mask.nonzero(as_tuple=True)[0]
        vals = q_val[idxs].to(torch.int16)
        return vals, idxs, l2_norm

    def unpack_qsgd_strict(packed):
        vals, idxs, l2_norm = packed
        dense = torch.zeros(DENSE_SIZE, dtype=torch.float32, device=device)
        decoded_vals = vals.to(torch.float32) * l2_norm / QSGD_LEVELS
        dense.scatter_(0, idxs, decoded_vals)
        return dense

    # 7. STC (Sattler et al. TNNLS 2020) — ternary values + proper Rice coding
    #    CORRECTED: original code used nibble-split (M=16 fixed), not actual Golomb/Rice coding.
    #    Proper Rice encoding writes variable-length unary + fixed-k-bit remainder to a bitstream,
    #    which is the serialisation bottleneck described in the paper.
    def pack_stc(vals, idxs):
        mu    = vals.abs().mean()
        signs = (torch.sign(vals) > 0)
        if (len(signs) % 8) > 0:
            signs = torch.nn.functional.pad(signs, (0, 8 - len(signs) % 8))
        p_signs   = (signs.view(-1, 8).to(torch.uint8) * POWERS_PT).sum(dim=1).to(torch.uint8)
        idxs_cpu  = idxs.cpu().numpy()
        gaps      = np.diff(np.insert(idxs_cpu, 0, -1)) - 1
        packed_bits = stc_cpp_module.encode_rice(
            torch.from_numpy(gaps.astype(np.int64)), k_rice)
        return p_signs, packed_bits, mu

    def unpack_stc(packed):
        p_signs, packed_bits, mu = packed
        r_idxs  = stc_cpp_module.decode_rice(packed_bits, K_SIZE, k_rice).to(device)
        un_signs = (torch.bitwise_and(p_signs.unsqueeze(1), POWERS_PT) > 0).view(-1)[:K_SIZE]
        dense   = torch.zeros(DENSE_SIZE, dtype=torch.float32, device=device)
        dense.scatter_(0, r_idxs, (un_signs.to(torch.float32) * 2 - 1) * mu)
        return dense

    methods = {
        "Ours (GPU Bitmap)":   (pack_ours,           unpack_ours,           (pt_vals, pt_idxs)),
        "Std Top-K (No Comp)": (pack_std,            unpack_std,            (pt_vals, pt_idxs)),
        "FedBiF (Corrected)":  (pack_fedbif_strict,  unpack_fedbif_strict,  (pt_dense,)),
        "FedPAQ (Corrected)":  (pack_fedpaq_strict,  unpack_fedpaq_strict,  (pt_dense,)),
        "FetchSGD (Corrected)":(pack_fetchsgd_strict, unpack_fetchsgd_strict,(pt_dense,)),
        "QSGD (Added)":        (pack_qsgd_strict,    unpack_qsgd_strict,    (pt_dense,)),
        "STC (Rice)":          (pack_stc,            unpack_stc,            (pt_vals, pt_idxs)),
    }

    torch.cuda.empty_cache()
    print(f"{'Algorithm':<28} | {'Pack Time (ms)':<15} | {'Unpack Time (ms)':<17} | {'Total Time (ms)':<15}")
    print("-" * 80)

    for name, (p_fn, u_fn, inputs) in methods.items():
        p_data = p_fn(*inputs)
        for _ in range(WARMUP):
            p_fn(*inputs); u_fn(p_data)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(NUM_TRIALS): p_fn(*inputs)
        torch.cuda.synchronize()
        t_pack = ((time.time() - t0) / NUM_TRIALS) * 1000

        t1 = time.time()
        for _ in range(NUM_TRIALS): u_fn(p_data)
        torch.cuda.synchronize()
        t_unpack = ((time.time() - t1) / NUM_TRIALS) * 1000

        print(f"{name:<28} | {t_pack:14.3f}  | {t_unpack:16.3f}  | {t_pack+t_unpack:14.3f}")
        benchmark_results.append({
            'Config': config['name'].strip(), 'Algorithm': name.strip(),
            'PackTime_ms': t_pack, 'UnpackTime_ms': t_unpack,
            'TotalTime_ms': t_pack + t_unpack
        })

# ── Save CSV and stacked-bar chart ──────────────────────────────────────────
df = pd.DataFrame(benchmark_results)
df.to_csv("pytorch_benchmark_results_corrected.csv", index=False)
print("\n Data saved to: pytorch_benchmark_results_corrected.csv")

fig, axes = plt.subplots(1, len(TEST_CONFIGS), figsize=(24, 7), sharey=False)
color_pack, color_unpack = "#3498db", "#e74c3c"

for ax, config in zip(axes, TEST_CONFIGS):
    config_name = config['name']
    sub_df = df[df['Config'] == config_name.strip()].copy().sort_values('TotalTime_ms')
    algos       = sub_df['Algorithm'].tolist()
    pack_times  = sub_df['PackTime_ms'].tolist()
    unpack_times= sub_df['UnpackTime_ms'].tolist()
    x = np.arange(len(algos))

    ax.bar(x, pack_times,   label='Pack (Client Encode)',  color=color_pack,   edgecolor='black', zorder=3)
    ax.bar(x, unpack_times, bottom=pack_times, label='Unpack (Server Decode)', color=color_unpack, edgecolor='black', zorder=3)
    ax.set_title(f"{config_name}\n({config['dense_size']/1e6:.1f}M Params | 10% Density)", fontsize=13, pad=15)
    ax.set_ylabel("End-to-End Latency (ms)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=40, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    for i, (p, u) in enumerate(zip(pack_times, unpack_times)):
        total = p + u
        ax.text(i, total + total*0.02, f"{total:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    if ax == axes[0]:
        ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig("latency_breakdown_corrected.png", dpi=300, bbox_inches='tight')
print(" Stacked bar chart saved to: latency_breakdown_corrected.png")
