import json
import os

data_dir = r"/mnt/apple/k66/huynt119/ecg-MI-classification/ecg/runs"
folders = os.listdir(data_dir)
paths = [os.path.join(data_dir, folder, 'all_folds_test_results.json') for folder in folders]

best_avg_acc = 0

for file_path in paths:
    with open(file_path, "r") as f:
        data = json.load(f)

    acc_values = [fold_data["test_acc"] for fold_data in data.values()]
    avg_acc = sum(acc_values) / len(acc_values)

    print(f"{file_path}: average test_acc = {avg_acc:.4f}")

    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_file = file_path

print("\n📌 File có trung bình test_acc cao nhất:")
print(f"{best_file}: {best_avg_acc:.4f}")