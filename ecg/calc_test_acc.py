import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def calc_test_acc(csv_path):
    df = pd.read_csv(csv_path)
    # Tổng thể
    correct = (df['gt_label'] == df['pred_label']).sum()
    total = len(df)
    acc = correct / total if total > 0 else 0

    # Tính sen, spe, f1 (MI là positive class)
    tn, fp, fn, tp = confusion_matrix(df['gt_label'], df['pred_label'], labels=['Healthy', 'MI']).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"Test accuracy (all folds): {acc:.4f} ({correct}/{total})")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Theo từng fold
    print("\nMetrics by fold:")
    for fold, group in df.groupby('fold'):
        fold_correct = (group['gt_label'] == group['pred_label']).sum()
        fold_total = len(group)
        fold_acc = fold_correct / fold_total if fold_total > 0 else 0

        # Tính sen, spe, f1 cho từng fold
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(group['gt_label'], group['pred_label'], labels=['Healthy', 'MI']).ravel()
        sen_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        spe_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0
        prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        f1_f = 2 * (prec_f * sen_f) / (prec_f + sen_f) if (prec_f + sen_f) > 0 else 0

        print(f"  Fold {fold}: Acc={fold_acc:.4f}, Sen={sen_f:.4f}, Spe={spe_f:.4f}, F1={f1_f:.4f}")

    return acc, sensitivity, specificity, f1

def count_label_per_patient(csv_path, out_csv_path):
    df = pd.read_csv(csv_path)
    results = []
    metrics_dict = {}
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
        # Tính metrics cho fold này
        fold_acc = label_counts['correct'].mean()
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(label_counts['gt_label'], label_counts['final_label'], labels=['Healthy', 'MI']).ravel()
        sen_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        spe_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0
        prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        f1_f = 2 * (prec_f * sen_f) / (prec_f + sen_f) if (prec_f + sen_f) > 0 else 0
        metrics_dict[fold] = {'acc': fold_acc, 'sen': sen_f, 'spe': spe_f, 'f1': f1_f}
        print(f"Patient-level fold {fold}: Acc={fold_acc:.4f}, Sen={sen_f:.4f}, Spe={spe_f:.4f}, F1={f1_f:.4f}")

    # Tổng hợp tất cả folds
    all_patients = pd.concat(results, ignore_index=True)
    total_acc = all_patients['correct'].mean()
    tn_all, fp_all, fn_all, tp_all = confusion_matrix(all_patients['gt_label'], all_patients['final_label'], labels=['Healthy', 'MI']).ravel()
    sen_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
    spe_all = tn_all / (tn_all + fp_all) if (tn_all + fp_all) > 0 else 0
    prec_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
    f1_all = 2 * (prec_all * sen_all) / (prec_all + sen_all) if (prec_all + sen_all) > 0 else 0
    metrics_dict['all_folds'] = {'acc': total_acc, 'sen': sen_all, 'spe': spe_all, 'f1': f1_all}
    print(f"\nPatient-level (all folds): Acc={total_acc:.4f}, Sen={sen_all:.4f}, Spe={spe_all:.4f}, F1={f1_all:.4f}")

    if out_csv_path is not None:
        all_patients.to_csv(out_csv_path, index=False)
        print(f"\nĐã lưu chi tiết từng patient ra file: {out_csv_path}")

    return metrics_dict

def count_label_per_pt_path(csv_path, out_csv_path):
    df = pd.read_csv(csv_path)
    results = []
    metrics_dict = {}
    # Theo từng fold
    for fold, group in df.groupby('fold'):
        label_counts = group.groupby(['pt_path', 'pred_label']).size().unstack(fill_value=0)
        label_counts = label_counts.rename(columns={'MI': 'pred_MI', 'Healthy': 'pred_healthy'})
        for col in ['pred_MI', 'pred_healthy']:
            if col not in label_counts.columns:
                label_counts[col] = 0
        label_counts['final_label'] = label_counts[['pred_MI', 'pred_healthy']].idxmax(axis=1).map({'pred_MI': 'MI', 'pred_healthy': 'Healthy'})
        gt_label_map = group.groupby('pt_path')['gt_label'].first()
        label_counts['gt_label'] = gt_label_map
        label_counts['correct'] = (label_counts['final_label'] == label_counts['gt_label']).astype(int)
        label_counts['fold'] = fold
        label_counts = label_counts.reset_index()[['fold', 'pt_path', 'gt_label', 'pred_MI', 'pred_healthy', 'final_label', 'correct']]
        results.append(label_counts)
        # Tính metrics cho fold này
        fold_acc = label_counts['correct'].mean()
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(label_counts['gt_label'], label_counts['final_label'], labels=['Healthy', 'MI']).ravel()
        sen_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        spe_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0
        prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        f1_f = 2 * (prec_f * sen_f) / (prec_f + sen_f) if (prec_f + sen_f) > 0 else 0
        metrics_dict[fold] = {'acc': fold_acc, 'sen': sen_f, 'spe': spe_f, 'f1': f1_f}
        print(f"pt_path-level fold {fold}: Acc={fold_acc:.4f}, Sen={sen_f:.4f}, Spe={spe_f:.4f}, F1={f1_f:.4f}")

    # Tổng hợp tất cả folds
    all_pt_paths = pd.concat(results, ignore_index=True)
    total_acc = all_pt_paths['correct'].mean()
    tn_all, fp_all, fn_all, tp_all = confusion_matrix(all_pt_paths['gt_label'], all_pt_paths['final_label'], labels=['Healthy', 'MI']).ravel()
    sen_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
    spe_all = tn_all / (tn_all + fp_all) if (tn_all + fp_all) > 0 else 0
    prec_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
    f1_all = 2 * (prec_all * sen_all) / (prec_all + sen_all) if (prec_all + sen_all) > 0 else 0
    metrics_dict['all_folds'] = {'acc': total_acc, 'sen': sen_all, 'spe': spe_all, 'f1': f1_all}
    print(f"\npt_path-level (all folds): Acc={total_acc:.4f}, Sen={sen_all:.4f}, Spe={spe_all:.4f}, F1={f1_all:.4f}")

    if out_csv_path is not None:
        all_pt_paths.to_csv(out_csv_path, index=False)
        print(f"\nĐã lưu chi tiết từng pt_path ra file: {out_csv_path}")

    return metrics_dict

if __name__ == "__main__":
    folder = "20251024_085235"
    csv_path = f"runs/{folder}/checkpoints/all_checkpoints_predictions.csv"  # sửa lại đường dẫn nếu cần
    out_csv_path = f"runs/{folder}/checkpoints/patient_majority_vote_detail.csv"
    calc_test_acc(csv_path)
    metrics_dict = count_label_per_patient(csv_path, out_csv_path)

    # Lưu metrics từng fold và tổng hợp ra file TXT
    txt_path = f"runs/{folder}/checkpoints/patient_fold_metrics.txt"
    with open(txt_path, "w") as f:
        for fold, metrics in metrics_dict.items():
            if fold == 'all_folds':
                f.write(f"\nPatient-level (all folds):\n")
                f.write(f"  Accuracy: {metrics['acc']:.4f}\n")
                f.write(f"  Sensitivity: {metrics['sen']:.4f}\n")
                f.write(f"  Specificity: {metrics['spe']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            else:
                f.write(f"Patient-level fold {fold}: Acc={metrics['acc']:.4f}, Sen={metrics['sen']:.4f}, Spe={metrics['spe']:.4f}, F1={metrics['f1']:.4f}\n")
    print(f"\nĐã lưu metrics từng fold và tổng hợp theo patient ra file: {txt_path}")

    # Thêm lưu kết quả theo pt_path
    pt_path_csv = f"runs/{folder}/checkpoints/pt_path_majority_vote_detail.csv"
    pt_path_metrics_dict = count_label_per_pt_path(csv_path, pt_path_csv)
    pt_path_txt = f"runs/{folder}/checkpoints/pt_path_fold_metrics.txt"
    with open(pt_path_txt, "w") as f:
        for fold, metrics in pt_path_metrics_dict.items():
            if fold == 'all_folds':
                f.write(f"\npt_path-level (all folds):\n")
                f.write(f"  Accuracy: {metrics['acc']:.4f}\n")
                f.write(f"  Sensitivity: {metrics['sen']:.4f}\n")
                f.write(f"  Specificity: {metrics['spe']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            else:
                f.write(f"pt_path-level fold {fold}: Acc={metrics['acc']:.4f}, Sen={metrics['sen']:.4f}, Spe={metrics['spe']:.4f}, F1={metrics['f1']:.4f}\n")
    print(f"\nĐã lưu metrics từng fold và tổng hợp theo pt_path ra file: {pt_path_txt}")