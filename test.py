import torch
import matplotlib.pyplot as plt
from ecg.ECGDataset import ECGDataset
from ecg.ECGDataLoader import ECGDataLoader, ComposeTransforms, BaselineWander, GaussianNoise, PowerlineNoise, ChannelResize, BaselineShift

# Tạo transform
transform = ComposeTransforms([
BaselineWander(prob=0.5, C=0.0001),
GaussianNoise(prob=0.5, scale=0.0001),
PowerlineNoise(prob=0.5, C=0.0001),
ChannelResize(magnitude_range=(0.5, 2.0)),
BaselineShift(prob=0.5, scale=0.01),
])

# Lấy một mẫu từ dataset gốc
idx = 1
raw_dataset1 = ECGDataset(
    csv_file='ptb_fold.csv',
    data_dir='./',
    fold_list=[0],
    sample_before=198,
    sample_after=400,
    transform=None
)
wave_raw1 = raw_dataset1[idx][0]  

raw_dataset2 = ECGDataset(
    csv_file='ptb_fold.csv',
    data_dir='./',
    fold_list=[0],
    sample_before=198,
    sample_after=400,
    transform=transform
)
wave_raw2 = raw_dataset2[idx][0] 

# # Apply transform trực tiếp lên cùng một mẫu
# wave_transformed = transform(wave_raw.clone())  # clone để không thay đổi dữ liệu gốc

# Vẽ từng kênh (lead)
num_channels = wave_raw1.shape[0]
fig, axs = plt.subplots(num_channels, 1, figsize=(10, 2*num_channels))
for i in range(num_channels):
    axs[i].plot(wave_raw1[i].numpy(), label='Original')
    axs[i].plot(wave_raw2[i].numpy(), label='Transformed', alpha=0.7)
    axs[i].set_title(f'Lead {i+1}')
    axs[i].legend()
plt.tight_layout()
plt.savefig("ecg_transform_compare.png")