import pandas as pd

def calc_test_acc(csv_path):
    df = pd.read_csv(csv_path)
    # Tổng thể
    correct = (df['gt_label'] == df['pred_label']).sum()
    total = len(df)
    acc = correct / total if total > 0 else 0
    print(f"Test accuracy (all folds): {acc:.4f} ({correct}/{total})")

    # Theo từng fold
    print("\nAccuracy by fold:")
    for fold, group in df.groupby('fold'):
        fold_correct = (group['gt_label'] == group['pred_label']).sum()
        fold_total = len(group)
        fold_acc = fold_correct / fold_total if fold_total > 0 else 0
        print(f"  Fold {fold}: {fold_acc:.4f} ({fold_correct}/{fold_total})")

    return acc

def count_label_per_patient(csv_path, out_csv_path):
    df = pd.read_csv(csv_path)
    results = []
    acc_dict = {}
    # Theo từng fold
    for fold, group in df.groupby('fold'):
        label_counts = group.groupby(['patient_number', 'pred_label']).size().unstack(fill_value=0)
        label_counts = label_counts.rename(columns={'MI': 'pred_MI', 'Healthy': 'pred_healthy'})
        for col in ['pred_MI', 'pred_healthy']:
            if col not in label_counts.columns:
                label_counts[col] = 0
        label_counts['final_label'] = label_counts[['pred_MI', 'pred_healthy']].idxmax(axis=1).map({'pred_MI': 'MI', 'pred_healthy': 'Healthy'})
        gt_label_map = group.groupby('patient_number')['gt_label'].first()
        label_counts['gt_label'] = gt_label_map
        label_counts['correct'] = (label_counts['final_label'] == label_counts['gt_label']).astype(int)
        label_counts['fold'] = fold
        label_counts = label_counts.reset_index()[['fold', 'patient_number', 'gt_label', 'pred_MI', 'pred_healthy', 'final_label', 'correct']]
        results.append(label_counts)
        # Tính accuracy cho fold này
        fold_acc = label_counts['correct'].mean()
        acc_dict[fold] = fold_acc
        print(f"Patient-level accuracy fold {fold}: {fold_acc:.4f} ({label_counts['correct'].sum()}/{len(label_counts)})")

    # Tổng hợp tất cả folds
    all_patients = pd.concat(results, ignore_index=True)
    total_acc = all_patients['correct'].mean()
    acc_dict['all_folds'] = total_acc
    print(f"\nPatient-level accuracy (all folds): {total_acc:.4f} ({all_patients['correct'].sum()}/{len(all_patients)})")

    if out_csv_path is not None:
        all_patients.to_csv(out_csv_path, index=False)
        print(f"\nĐã lưu chi tiết từng patient ra file: {out_csv_path}")

    return acc_dict

if __name__ == "__main__":
    folder = "20250814_221244"
    csv_path = f"runs/{folder}/checkpoints/all_checkpoints_predictions.csv"  # sửa lại đường dẫn nếu cần
    out_csv_path = f"runs/{folder}/checkpoints/patient_majority_vote_detail.csv"
    calc_test_acc(csv_path)
    acc_dict = count_label_per_patient(csv_path, out_csv_path)

    # Lưu accuracy từng fold và tổng hợp ra file TXT
    txt_path = f"runs/{folder}/checkpoints/patient_fold_accuracy.txt"
    with open(txt_path, "w") as f:
        for fold, acc in acc_dict.items():
            if fold == 'all_folds':
                f.write(f"\nPatient-level accuracy (all folds): {acc:.4f}\n")
            else:
                f.write(f"Patient-level accuracy fold {fold}: {acc:.4f}\n")
    print(f"\nĐã lưu accuracy từng fold và tổng hợp theo patient ra file: {txt_path}")