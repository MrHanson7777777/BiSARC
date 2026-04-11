import os
import re


stats_file = 'log/cifar100+100/cifar100+labeldir0.1/resnet18+fedbif+bit_8+lr_0.01/packing_ratio_stats.txt'

def calc_and_append_global_average(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    total_time_sum = 0.0
    pack_time_sum = 0.0
    unpack_time_sum = 0.0
    train_time_sum = 0.0
    client_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if "总耗时:" in line or "Total time:" in line:
            m = re.search(r"(Total time):\s*([\d\.]+)ms", line)
            if m: total_time_sum += float(m.group(1))
            client_count += 1
        elif "打包开销:" in line or "Pack overhead:" in line:
            m = re.search(r"(Pack overhead):\s*([\d\.]+)ms", line)
            if m: pack_time_sum += float(m.group(1))
        elif "解包开销:" in line or "Unpack overhead:" in line:
            m = re.search(r"(Unpack overhead):\s*([\d\.]+)ms", line)
            if m: unpack_time_sum += float(m.group(1))
        elif "本地训练:" in line or "Local train:" in line:
            m = re.search(r"(Local train):\s*([\d\.]+)ms", line)
            if m: train_time_sum += float(m.group(1))

    if client_count == 0:
        print("No valid client data found in the file, please check the logs!")
        return


    avg_total = total_time_sum / client_count
    avg_pack = pack_time_sum / client_count
    avg_unpack = unpack_time_sum / client_count
    avg_train = train_time_sum / client_count

    ratio_pack = (avg_pack / avg_total) * 100 if avg_total > 0 else 0
    ratio_unpack = (avg_unpack / avg_total) * 100 if avg_total > 0 else 0
    ratio_train = (avg_train / avg_total) * 100 if avg_total > 0 else 0

    summary = (
        f"\n{'='*50}\n"
        f"🌟 Global Average Stats (aggregated data from {client_count} clients)\n"
        f"{'='*50}\n"
        f"Global average Total time: {avg_total:.2f}ms\n"
        f"  ├─ Average Pack overhead: {avg_pack:.2f}ms ({ratio_pack:.2f}%)\n"
        f"  ├─ Average Unpack overhead: {avg_unpack:.2f}ms ({ratio_unpack:.2f}%)\n"
        f"  └─ Average Local train: {avg_train:.2f}ms ({ratio_train:.2f}%)\n"
        f"{'='*50}\n"
    )

    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(summary)

    print(summary)
    print(f"✅ Global average stats successfully appended to the end of the file: {file_path}")

if __name__ == "__main__":
    calc_and_append_global_average(stats_file)