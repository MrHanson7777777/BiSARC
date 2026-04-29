import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.eager import context

tf.random.set_seed(42)
np.random.seed(42)

physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    raise RuntimeError("No GPU detected, please run this benchmark in a GPU environment.")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

TEST_CONFIGS = [
    {"name": "LLaMA-7B (LoRA r=8)", "dense_size": 4_200_000,  "cr": 0.10},
    {"name": "ResNet-18 (Full)",     "dense_size": 11_180_000, "cr": 0.10},
    {"name": "ResNet-50 (Full)",     "dense_size": 25_600_000, "cr": 0.10},
    {"name": "ViT-Base (Full)",      "dense_size": 86_000_000, "cr": 0.10},
]
NUM_TRIALS, WARMUP = 100, 15

print("=" * 115)
print(" Rigorous TensorFlow End-to-End Latency Breakdown [Corrected]")
print("=" * 115)

POWERS_TF = tf.constant([128, 64, 32, 16, 8, 4, 2, 1], dtype=tf.uint8)
benchmark_results = []

for config in TEST_CONFIGS:
    DENSE_SIZE  = config["dense_size"]
    CR          = config["cr"]
    K_SIZE      = max(1, int(DENSE_SIZE * CR))
    SKETCH_SIZE = int(np.log2(DENSE_SIZE) * 100)
    pad_rem     = DENSE_SIZE % 8
    PAD_LEN     = (8 - pad_rem) if pad_rem > 0 else 0

    print(f"\nScenario: {config['name']} | Params: {DENSE_SIZE/1e6:.2f}M | Sparsity: {CR*100:.0f}%")

    tf_dense  = tf.random.uniform(shape=[DENSE_SIZE], dtype=tf.float32)
    tf_idxs   = tf.sort(tf.random.shuffle(tf.range(DENSE_SIZE, dtype=tf.int32))[:K_SIZE])
    tf_vals   = tf.gather(tf_dense, tf_idxs)

    NUM_ROWS = 5
    tf_hash_idxs_multi  = tf.random.uniform([NUM_ROWS, DENSE_SIZE], minval=0, maxval=SKETCH_SIZE, dtype=tf.int32)
    tf_hash_signs_multi = tf.cast(tf.random.uniform([NUM_ROWS, DENSE_SIZE], minval=0, maxval=2, dtype=tf.int32) * 2 - 1, tf.float32)

    tf_alpha    = tf.constant(0.01, dtype=tf.float32)
    tf_frozen_s = tf.cast(tf.random.uniform([DENSE_SIZE], minval=0, maxval=15, dtype=tf.int32), tf.float32)
    tf_bit_index = tf.constant(2, dtype=tf.float32)
    QSGD_LEVELS  = 255.0

    # ── Algorithm implementations ────────────────────────────────────────────

    # 1. Ours (GPU Bitmap)
    @tf.function(jit_compile=True)
    def pack_ours(vals, idxs):
        mask = tf.scatter_nd(tf.expand_dims(idxs, 1), tf.ones_like(idxs, dtype=tf.bool), [DENSE_SIZE])
        if PAD_LEN > 0:
            mask = tf.pad(mask, [[0, PAD_LEN]])
        reshaped_mask = tf.cast(tf.reshape(mask, [-1, 8]), tf.uint8)
        packed_mask   = tf.reduce_sum(reshaped_mask * POWERS_TF, axis=1)
        return tf.cast(packed_mask, tf.uint8), vals

    # Remove jit_compile: real bitmap decode via tf.where (parallel addressing)
    @tf.function
    def unpack_ours(p_mask, vals):
        unmask = tf.reshape(tf.bitwise.bitwise_and(tf.expand_dims(p_mask, 1), POWERS_TF) > 0, [-1])[:DENSE_SIZE]
        idxs   = tf.cast(tf.where(unmask), tf.int32)
        dense  = tf.scatter_nd(idxs, vals, [DENSE_SIZE])
        return dense

    # 2. Std Top-K
    @tf.function(jit_compile=True)
    def pack_std(vals, idxs):  return vals, idxs

    @tf.function(jit_compile=True)
    def unpack_std(vals, idxs):
        return tf.scatter_nd(tf.expand_dims(idxs, 1), vals, [DENSE_SIZE])

    # 3. FedBiF
    @tf.function(jit_compile=True)
    def pack_fedbif_strict(dense):
        activated_bits = dense > 0
        if PAD_LEN > 0:
            activated_bits = tf.pad(activated_bits, [[0, PAD_LEN]])
        reshaped_bits = tf.cast(tf.reshape(activated_bits, [-1, 8]), tf.uint8)
        packed_bits   = tf.cast(tf.reduce_sum(reshaped_bits * POWERS_TF, axis=1), tf.uint8)
        return packed_bits

    @tf.function(jit_compile=True)
    def unpack_fedbif_strict(packed_bits):
        unbits = tf.reshape(tf.bitwise.bitwise_and(tf.expand_dims(packed_bits, 1), POWERS_TF) > 0, [-1])[:DENSE_SIZE]
        recovered = tf_alpha * ((2.0 ** tf_bit_index) * tf.cast(unbits, tf.float32) + tf_frozen_s)
        return recovered

    # 4. FedPAQ — L2-norm stochastic quantization (Reisizadeh et al. AISTATS 2020)
    #    CORRECTED: original code used min-max normalization which does not match the paper.
    #    FedPAQ uses the same unbiased L2-norm quantizer as QSGD (dense int16 storage).
    @tf.function(jit_compile=True)
    def pack_fedpaq_strict(dense):
        l2_norm    = tf.norm(dense, ord=2) + 1e-7
        abs_scaled = tf.abs(dense) * 255.0 / l2_norm
        floor_val  = tf.floor(abs_scaled)
        prob       = abs_scaled - floor_val
        rand_val   = tf.random.uniform(tf.shape(prob), dtype=tf.float32)
        q_val      = (floor_val + tf.cast(rand_val < prob, tf.float32)) * tf.sign(dense)
        return tf.cast(q_val, tf.int16), l2_norm

    @tf.function(jit_compile=True)
    def unpack_fedpaq_strict(q, l2_norm):
        return tf.cast(q, tf.float32) * l2_norm / 255.0

    # 5. FetchSGD
    @tf.function(jit_compile=True)
    def pack_fetchsgd_strict(dense):
        sketches_list = tf.TensorArray(tf.float32, size=NUM_ROWS)
        for r in tf.range(NUM_ROWS):
            row_sketch = tf.zeros([SKETCH_SIZE], dtype=tf.float32)
            row_sketch = tf.tensor_scatter_nd_add(row_sketch,
                         tf.expand_dims(tf_hash_idxs_multi[r], 1), dense * tf_hash_signs_multi[r])
            sketches_list = sketches_list.write(r, row_sketch)
        return sketches_list.stack()

    @tf.function(jit_compile=True)
    def unpack_fetchsgd_strict(sketches):
        estimates_list = tf.TensorArray(tf.float32, size=NUM_ROWS)
        for r in tf.range(NUM_ROWS):
            est = tf.gather(sketches[r], tf_hash_idxs_multi[r]) * tf_hash_signs_multi[r]
            estimates_list = estimates_list.write(r, est)
        all_estimates = estimates_list.stack()
        sorted_ests   = tf.sort(all_estimates, axis=0)
        return sorted_ests[NUM_ROWS // 2]

    # 6. QSGD — L2-norm stochastic quantization, sparse non-zero storage
    # Remove jit_compile: dynamic tf.where prevents XLA compilation
    @tf.function
    def pack_qsgd_strict(dense):
        l2_norm    = tf.norm(dense, ord=2) + 1e-7
        abs_scaled = tf.abs(dense) * QSGD_LEVELS / l2_norm
        floor_val  = tf.floor(abs_scaled)
        prob       = abs_scaled - floor_val
        rand_val   = tf.random.uniform(tf.shape(prob), dtype=tf.float32)
        q_val      = (floor_val + tf.cast(rand_val < prob, tf.float32)) * tf.sign(dense)
        non_zero_mask = q_val != 0
        idxs = tf.where(non_zero_mask)
        vals = tf.gather_nd(q_val, idxs)
        return tf.cast(vals, tf.int16), tf.cast(idxs, tf.int32), l2_norm

    @tf.function(jit_compile=True)
    def unpack_qsgd_strict(vals, idxs, l2_norm):
        decoded_vals = tf.cast(vals, tf.float32) * l2_norm / QSGD_LEVELS
        dense = tf.scatter_nd(idxs, decoded_vals, [DENSE_SIZE])
        return dense

    methods = {
        "Ours (GPU Bitmap)":    (pack_ours,            unpack_ours,            (tf_vals, tf_idxs)),
        "Std Top-K (No Comp)":  (pack_std,             unpack_std,             (tf_vals, tf_idxs)),
        "FedBiF (Corrected)":   (pack_fedbif_strict,   unpack_fedbif_strict,   (tf_dense,)),
        "FedPAQ (Corrected)":   (pack_fedpaq_strict,   unpack_fedpaq_strict,   (tf_dense,)),
        "FetchSGD (Corrected)": (pack_fetchsgd_strict,  unpack_fetchsgd_strict, (tf_dense,)),
        "QSGD (Added)":         (pack_qsgd_strict,     unpack_qsgd_strict,     (tf_dense,)),
    }

    print(f"{'Algorithm':<28} | {'Pack Time (ms)':<15} | {'Unpack Time (ms)':<17} | {'Total Time (ms)':<15}")
    print("-" * 80)

    for name, (p_fn, u_fn, inputs) in methods.items():
        p_data = p_fn(*inputs)
        if name in ["Ours (GPU Bitmap)", "Std Top-K (No Comp)"]:
            u_data_input = (p_data[0], p_data[1])
        elif name in ["FedPAQ (Corrected)", "QSGD (Added)"]:
            u_data_input = p_data
        else:
            u_data_input = (p_data,)

        for _ in range(WARMUP):
            _ = p_fn(*inputs)
            _ = u_fn(*u_data_input)
        context.async_wait()

        t0 = time.time()
        for _ in range(NUM_TRIALS): _ = p_fn(*inputs)
        context.async_wait()
        t_pack = ((time.time() - t0) / NUM_TRIALS) * 1000

        t1 = time.time()
        for _ in range(NUM_TRIALS): _ = u_fn(*u_data_input)
        context.async_wait()
        t_unpack = ((time.time() - t1) / NUM_TRIALS) * 1000

        print(f"{name:<28} | {t_pack:14.3f}  | {t_unpack:16.3f}  | {t_pack+t_unpack:14.3f}")
        benchmark_results.append({
            'Config': config['name'].strip(), 'Algorithm': name.strip(),
            'PackTime_ms': t_pack, 'UnpackTime_ms': t_unpack,
            'TotalTime_ms': t_pack + t_unpack
        })

# ── Save CSV and stacked-bar chart ──────────────────────────────────────────
df = pd.DataFrame(benchmark_results)
df.to_csv("tf_benchmark_results_corrected.csv", index=False)
print("\n Data saved to: tf_benchmark_results_corrected.csv")

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
plt.savefig("tf_latency_breakdown_corrected.png", dpi=300, bbox_inches='tight')
print(" Stacked bar chart saved to: tf_latency_breakdown_corrected.png")
