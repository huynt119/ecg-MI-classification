import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
import torchvision.transforms as transforms  
from Transform import random_shift_1d, ComposeTransforms, roll_1d
from ECGDataset import ECGDataset
import random
import math

def Tnoise_powerline(fs=100, N=1000,C=1,fn=50.,K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    #C *= 0.333 #adjust default scale
    t = torch.arange(0,N/fs,1./fs)
    
    signal = torch.zeros(N)
    phi1 = random.uniform(0,2*math.pi)
    for k in range(1,K+1):
        ak = random.uniform(0,1)
        signal += C*ak*torch.cos(2*math.pi*k*fn*t+phi1)
    signal = C*signal[:,None]
    if(channels>1):
        channel_gains = torch.empty(channels).uniform_(-1,1)
        signal = signal*channel_gains[None]
    return signal

def Tnoise_baseline_wander(fs=100, N=1000, C=1.0, fc=0.5, fdelta=0.01,channels=1,independent_channels=False):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if(fdelta is None):# 0.1
        fdelta = fs/N

    K = int((fc/fdelta)+0.5)
    t = torch.arange(0, N/fs, 1./fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2*math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2*math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C*res

class ChannelResize():
    def __init__(self, magnitude_range=(0.5, 2)):
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range))

    def __call__(self, wave):
        channels, len_wave = wave.shape
        resize_factors = torch.exp(torch.empty(channels).uniform_(*self.log_magnitude_range)) 
        resize_factors = resize_factors.repeat(len_wave).view(wave.T.shape).T 
        wave = resize_factors * wave
        return wave
    
class GaussianNoise():
    def __init__(self, prob=1.0, scale=0.01):
        self.scale = scale
        self.prob = prob
    
    def __call__(self, wave):
        if random.random() < self.prob:
            wave += self.scale * torch.randn(wave.shape)
        return wave
    
class BaselineShift():
    def __init__(self, prob=1.0, scale=1.0):
        self.prob = prob
        self.scale = scale

    def __call__(self, wave):
        if random.random() < self.prob:
            shift = torch.randn(1)
            wave = wave + self.scale * shift
        return wave
    
class BaselineWander():
    def __init__(self, prob=1.0, freq=1000):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            wander = Tnoise_baseline_wander(fs=self.freq, N=len_wave) 
            wander = wander.repeat(channels).view(wave.shape)
            wave = wave + wander
        return wave

class PowerlineNoise():
    def __init__(self, prob=1.0, freq=1000):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            noise = Tnoise_powerline(fs=self.freq, N=len_wave, channels=channels).T 
            wave = wave + noise
        return wave

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wave):
        for t in self.transforms:
            wave = t(wave)
        return wave



class ECGDataLoader(L.LightningDataModule):

    def __init__(self, csv_file, data_dir, fold_train, fold_test, batch_size, num_workers,
                 split_ratio=0.9, sample_before=100, sample_after=100):
        super().__init__()
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.fold_train = fold_train
        self.fold_test = fold_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.sample_before = sample_before
        self.sample_after = sample_after

        # ✅ define transform to apply only on training set
        self.train_transform = ComposeTransforms([
            GaussianNoise(prob=0.2),
            ChannelResize(),
            BaselineShift(prob=0.2),
            BaselineWander(prob=0.2),
            PowerlineNoise(prob=0.2),
        ])

    def setup(self, stage=None):
        full_dataset = ECGDataset(
            csv_file=self.csv_file,
            data_dir=self.data_dir,
            fold_list=self.fold_train,
            sample_before=self.sample_before,
            sample_after=self.sample_after,
            transform=None  # set transform later
        )

        total_size = len(full_dataset)
        train_size = int(total_size * self.split_ratio)
        val_size = total_size - train_size

        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # ✅ Gán transform chỉ cho train dataset
        train_subset.dataset.transform = self.train_transform
        val_subset.dataset.transform = None

        self.train_dataset = train_subset
        self.val_dataset = val_subset

        self.test_dataset = ECGDataset(
            csv_file=self.csv_file,
            data_dir=self.data_dir,
            fold_list=self.fold_test,
            sample_before=self.sample_before,
            sample_after=self.sample_after,
            transform=None  # no transform on test
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )