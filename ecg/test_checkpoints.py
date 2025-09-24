import os
import importlib
import torch
import json
import csv
from ECGDataLoader import ECGDataLoader
import lightning as L

def test_all_checkpoints(
    checkpoints_dir,
    label_file='ptb_fold.csv',
    data_dir='../',
    batch_size=256,
    num_workers=2,
    split_ratio=0.8,
    sample_before=198,
    sample_after=400,
    num_classes=2,
    learning_rate=1e-3,
    model_name='MCDANN'
):
    model_module = importlib.import_module(f'ECGModel.{model_name}')
    ModelClass = getattr(model_module, model_name)

    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    checkpoint_files.sort()

    all_results = {}
    all_rows = []

    label_dict = {0: "MI", 1: "Healthy"}

    for ckpt_file in checkpoint_files:
        fold_str = ckpt_file.split('-')[0].replace('[', '').replace(']', '')
        fold = int(fold_str)

        print(f"\n🔎 Testing checkpoint: {ckpt_file} (fold {fold})")

        dataloader = ECGDataLoader(
            csv_file=label_file,
            data_dir=data_dir,
            fold_train=[i for i in range(5) if i != fold],
            fold_test=[fold],
            batch_size=batch_size,
            num_workers=num_workers,
            split_ratio=split_ratio,
            sample_before=sample_before,
            sample_after=sample_after
        )
        dataloader.setup()

        ckpt_path = os.path.join(checkpoints_dir, ckpt_file)
        model = ModelClass.load_from_checkpoint(
            ckpt_path,
            num_classes=num_classes,
            learning_rate=learning_rate
        )
        model.eval()
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

        device = next(model.parameters()).device

        test_loader = dataloader.test_dataloader()
        with torch.no_grad():
            for batch in test_loader:
                inputs, gt_labels, patient_numbers, r_peak_indices, pt_paths = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
                gt_labels = gt_labels.cpu().numpy()
                patient_numbers = patient_numbers.cpu().numpy() if torch.is_tensor(patient_numbers) else patient_numbers
                r_peak_indices = r_peak_indices.cpu().numpy() if torch.is_tensor(r_peak_indices) else r_peak_indices
                pt_paths = pt_paths if isinstance(pt_paths, list) else pt_paths

                for i in range(len(gt_labels)):
                    all_rows.append({
                        'patient_number': patient_numbers[i],
                        'pt_path': pt_paths[i],
                        'r_peak_index': r_peak_indices[i],
                        'gt_label': label_dict[int(gt_labels[i])],
                        'pred_label': label_dict[int(pred_labels[i])],
                        'fold': fold
                    })

        trainer = L.Trainer(accelerator='auto', devices=1)
        test_results = trainer.test(model, dataloader.test_dataloader())
        print(f"✅ Test results for {ckpt_file}:")
        for k, v in test_results[0].items():
            print(f"{k}: {v:.4f}")
        all_results[ckpt_file] = test_results[0]

    # Save all results
    results_path = os.path.join(checkpoints_dir, "all_checkpoints_test_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n📄 All checkpoint test results saved to: {results_path}")

    # Save detailed predictions to CSV
    csv_path = os.path.join(checkpoints_dir, "all_checkpoints_predictions.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['patient_number', 'pt_path', 'r_peak_index', 'gt_label', 'pred_label', 'fold'])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"📝 Detailed predictions saved to: {csv_path}")

if __name__ == "__main__":
    checkpoints_dir = "runs/20250814_221244/checkpoints"
    test_all_checkpoints(checkpoints_dir)