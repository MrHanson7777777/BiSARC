import os
import re

# ==========================================
# 填入你生成的 packing_time_stats.txt 的实际路径
# ==========================================
stats_file = 'log/cifar100+100/cifar100+labeldir0.5/resnet18+fedbif+bit_8+lr_0.01/packing_time_stats.txt'

def calc_global_average(file_path):
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    total_pack = 0.0
    total_unpack = 0.0
    valid_client_count = 0

    # 用正则表达式提取文件中的浮点数
    # 匹配格式: "平均打包(量化): 12.34 ms | 平均解包(反量化): 5.67 ms"
    pattern = re.compile(r"平均打包\(量化\):\s*([\d\.]+)\s*ms\s*\|\s*平均解包\(反量化\):\s*([\d\.]+)\s*ms")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                total_pack += float(match.group(1))
                total_unpack += float(match.group(2))
                valid_client_count += 1

    if valid_client_count > 0:
        global_avg_pack = total_pack / valid_client_count
        global_avg_unpack = total_unpack / valid_client_count
        
        print("\n" + "="*40)
        print(f"📊 FedBiF 全局打包/解包开销统计")
        print("="*40)
        print(f"参与统计的客户端总数: {valid_client_count}")
        print(f"全局平均 打包(量化)时间: {global_avg_pack:.2f} ms")
        print(f"全局平均 解包(反量化)时间: {global_avg_unpack:.2f} ms")
        print("="*40 + "\n")
        print("💡 提示：你可以直接将这两个全局平均值填入你论文的对比表格中！")
    else:
        print("文件内容为空或格式不匹配，无法提取时间数据。")

if __name__ == "__main__":
    calc_global_average(stats_file)