import os
import re

stats_file = 'log/cifar100+100/cifar100+labeldir0.5/resnet18+fedbif+bit_8+lr_0.01/packing_time_stats.txt'

def calc_global_average(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    total_pack = 0.0
    total_unpack = 0.0
    valid_client_count = 0

    pattern = re.compile(r"(Average Pack \(Quantization\)):\s*([\d\.]+)\s*ms\s*\|\s*(?:平均解包\(反量化\)|Average Unpack \(Dequantization\)):\s*([\d\.]+)\s*ms")

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
        print(f"📊 FedBiF Global Pack/Unpack Overhead Statistics")
        print("="*40)
        print(f"Total number of statistical clients: {valid_client_count}")
        print(f"Global average Pack (Quantization) time: {global_avg_pack:.2f} ms")
        print(f"Global average Unpack (Dequantization) time: {global_avg_unpack:.2f} ms")
        print("="*40 + "\n")
        print("💡 Hint: You can fill these two global average values directly into the comparison table in your paper!")
    else:
        print("The file is empty or formatted incorrectly, unable to extract time data.")

if __name__ == "__main__":
    calc_global_average(stats_file)