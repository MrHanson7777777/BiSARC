import os
import re

# ==========================================
# 填入你实际生成的 packing_ratio_stats.txt 的路径
# ==========================================
stats_file = 'log/cifar100+100/cifar100+labeldir0.1/resnet18+fedbif+bit_8+lr_0.01/packing_ratio_stats.txt'

def calc_and_append_global_average(file_path):
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    total_time_sum = 0.0
    pack_time_sum = 0.0
    unpack_time_sum = 0.0
    train_time_sum = 0.0
    client_count = 0

    # 逐行读取各个客户端写入的数据
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if "总耗时:" in line:
            m = re.search(r"总耗时:\s*([\d\.]+)ms", line)
            if m: total_time_sum += float(m.group(1))
            client_count += 1
        elif "打包开销:" in line:
            m = re.search(r"打包开销:\s*([\d\.]+)ms", line)
            if m: pack_time_sum += float(m.group(1))
        elif "解包开销:" in line:
            m = re.search(r"解包开销:\s*([\d\.]+)ms", line)
            if m: unpack_time_sum += float(m.group(1))
        elif "本地训练:" in line:
            m = re.search(r"本地训练:\s*([\d\.]+)ms", line)
            if m: train_time_sum += float(m.group(1))

    if client_count == 0:
        print("文件中没有找到有效的客户端数据，请检查日志！")
        return

    # 1. 计算所有客户端的平均耗时 (ms)
    avg_total = total_time_sum / client_count
    avg_pack = pack_time_sum / client_count
    avg_unpack = unpack_time_sum / client_count
    avg_train = train_time_sum / client_count

    # 2. 计算全局平均占比 (%)
    ratio_pack = (avg_pack / avg_total) * 100 if avg_total > 0 else 0
    ratio_unpack = (avg_unpack / avg_total) * 100 if avg_total > 0 else 0
    ratio_train = (avg_train / avg_total) * 100 if avg_total > 0 else 0

    # 3. 构造漂亮的统计总结信息
    summary = (
        f"\n{'='*50}\n"
        f"🌟 全局平均统计 (共聚合 {client_count} 个客户端数据)\n"
        f"{'='*50}\n"
        f"全局平均 总耗时: {avg_total:.2f}ms\n"
        f"  ├─ 平均 打包开销: {avg_pack:.2f}ms ({ratio_pack:.2f}%)\n"
        f"  ├─ 平均 解包开销: {avg_unpack:.2f}ms ({ratio_unpack:.2f}%)\n"
        f"  └─ 平均 本地训练: {avg_train:.2f}ms ({ratio_train:.2f}%)\n"
        f"{'='*50}\n"
    )

    # 4. 追加写入到原本的 txt 文件末尾
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(summary)

    print(summary)
    print(f"✅ 全局平均统计已成功追加到文件末尾: {file_path}")

if __name__ == "__main__":
    calc_and_append_global_average(stats_file)