from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from ECGDataset import ECGDataset
import lightning as L
from torch.utils.data import random_split
from fold import iterate_fold
from lightning.pytorch.callbacks import StochasticWeightAveraging
from ECGDataLoader import ECGDataLoader
import importlib
import torch
import random
import numpy as np
import json
import os
from datetime import datetime

def main():
    batch_size = 256
    num_workers = 2
    num_epochs = 15
    learning_rate = 1e-3
    val_metric = 'val_acc'
    mode = 'max'
    num_classes = 2
    label_file = 'ptb_fold.csv'
    data_dir = '../'
    split_ratio = 0.8
    sample_before = 198
    sample_after = 400
    # sample_before = 598
    # sample_after = 3400
    model_name = 'MCDANN'  #set model name here
    # model_name = 'SRT'  #set model name here
    model_module = importlib.import_module(f'ECGModel.{model_name}')
    ModelClass = getattr(model_module, model_name)

    random_seed = 30
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


            # === Create run folder ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n📂 Results and checkpoints will be saved in: {run_dir}\n")

    all_results = {}

    for current, remaining in iterate_fold([0, 1, 2, 3, 4]):

        dataloader = ECGDataLoader(
            csv_file=label_file,
            data_dir=data_dir,
            fold_train=remaining,
            fold_test=current,
            batch_size=batch_size,
            num_workers=num_workers,
            split_ratio=split_ratio,
            sample_before=sample_before,
            sample_after=sample_after
        )
        dataloader.setup()
        
        model = ModelClass(num_classes=num_classes, learning_rate=learning_rate)

        checkpoint_callback = ModelCheckpoint(
                    dirpath=checkpoints_dir,
                    filename=f'{current}-ecg-{{epoch:02d}}-{{{val_metric}:.4f}}',
                    save_top_k=1,
                    monitor=val_metric,
                    mode=mode
                )

        trainer = L.Trainer(
            max_epochs=num_epochs,
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-3)],
        )

        trainer.fit(model, dataloader.train_dataloader(), dataloader.val_dataloader())

        # Lưu model đã SWA
        swa_model_path = os.path.join(checkpoints_dir, f'{current}-ecg-swa.ckpt')
        trainer.save_checkpoint(swa_model_path)

        # Test với model đã SWA
        test_results = trainer.test(model, dataloader.test_dataloader())

        # Pretty print
        print(f"\n✅ Test results for fold {current}:")
        for k, v in test_results[0].items():
            print(f"{k}: {v:.4f}")

        # Save into the big dictionary
        all_results[f"fold_{current}"] = test_results[0]

    # Save all results in a single JSON file
    all_results_file = os.path.join(run_dir, "all_folds_test_results.json")
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    note_path = os.path.join(run_dir, "note.txt")    

    with open (note_path, "w") as note_file:
        note_file.write("Crop -> Augment -> Downsample -> Normalize\n")
        note_file.write('batch: 256, epoch=15\n')
        note_file.write('Attention + Positional Encoding\n')
        note_file.write("""
BaselineWander(prob=0.5, C=0.0001),
GaussianNoise(prob=0.5, scale=0.0001),
PowerlineNoise(prob=0.5, C=0.0001),
ChannelResize(magnitude_range=(0.5, 2.0)),
BaselineShift(prob=0.5, scale=0.01),
""")
        

    print(f"\n📄 All test results saved to: {all_results_file}\n")

if __name__ == '__main__':
    main()