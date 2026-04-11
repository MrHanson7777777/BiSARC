import os
import json
import matplotlib.pyplot as plt
import csv

# ==========================================
# 填入你刚才找到的 JSON 记录文件路径
# ==========================================
json_path = 'log/shakespeare+100/shakespeare+natural/lstm+fedbif+bit_8+lr_0.8/results/record_autodl-container-9mfyz1fuy6-8bd739bf_2026-03-13-17-49-01.json'

def extract_and_plot_json(file_path):
    if not os.path.exists(file_path):
        print(f"找不到 JSON 文件: {file_path}")
        return

    # 读取 JSON 数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # === 针对 FedBiF 官方格式的精确提取 ===
    try:
        accs = data['accuracy']
        losses = data['loss']
        # 官方格式没有直接保存 round 列表，我们根据列表长度自动生成 [1, 2, 3, ...]
        rounds = list(range(1, len(accs) + 1))
    except KeyError as e:
        print(f"提取失败，缺少关键字段: {e}")
        return

    if not rounds:
        print("JSON 文件中没有数据！可能训练刚刚开始。")
        return

    print(f"成功提取了 {len(rounds)} 轮的数据！")
    print(f"最终轮 (Round {rounds[-1]}) -> Loss: {losses[-1]:.4f}, Accuracy: {accs[-1]:.4f}")

    # ================= 绘制图表 =================
    plt.figure(figsize=(12, 5))

    # 画 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(rounds, losses, label='FedBiF (8-bit)', color='#1f77b4', linewidth=2)
    plt.title('Global Test Loss', fontsize=14)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 画 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(rounds, accs, label='FedBiF (8-bit)', color='#ff7f0e', linewidth=2)
    plt.title('Global Test Accuracy', fontsize=14)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    
    # 保存图片
    save_img_path = 'shakespeare.png'
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {save_img_path}")
    
    # 保存 CSV
    csv_path = 'shakespeare.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "accuracy"])
        for r, l, a in zip(rounds, losses, accs):
            writer.writerow([r, l, a])
    print(f"实验数据已导出至: {csv_path}")

if __name__ == '__main__':
    extract_and_plot_json(json_path)