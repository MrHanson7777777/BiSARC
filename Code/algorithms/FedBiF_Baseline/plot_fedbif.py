import os
import json
import matplotlib.pyplot as plt
import csv

json_path = 'log/shakespeare+100/shakespeare+natural/lstm+fedbif+bit_8+lr_0.8/results/record_autodl-container-9mfyz1fuy6-8bd739bf_2026-03-13-17-49-01.json'

def extract_and_plot_json(file_path):
    if not os.path.exists(file_path):
        print(f"JSON file not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        accs = data['accuracy']
        losses = data['loss']
        rounds = list(range(1, len(accs) + 1))
    except KeyError as e:
        print(f"Extraction failed, missing key field: {e}")
        return

    if not rounds:
        print("No data in JSON file! Training might have just started.")
        return

    print(f"Successfully extracted data for {len(rounds)} rounds!")
    print(f"Final round (Round {rounds[-1]}) -> Loss: {losses[-1]:.4f}, Accuracy: {accs[-1]:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, losses, label='FedBiF (8-bit)', color='#1f77b4', linewidth=2)
    plt.title('Global Test Loss', fontsize=14)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, accs, label='FedBiF (8-bit)', color='#ff7f0e', linewidth=2)
    plt.title('Global Test Accuracy', fontsize=14)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    
    save_img_path = 'shakespeare.png'
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    print(f"Convergence curve saved to: {save_img_path}")
    
    csv_path = 'shakespeare.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "accuracy"])
        for r, l, a in zip(rounds, losses, accs):
            writer.writerow([r, l, a])
    print(f"Experimental data exported to: {csv_path}")

if __name__ == '__main__':
    extract_and_plot_json(json_path)