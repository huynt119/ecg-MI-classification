import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=8, kernel_sizes=[5,3]):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, growth_rate, kernel_size=kernel_sizes[0],
                               padding="same")
        self.bn2 = nn.BatchNorm1d(in_channels + growth_rate)
        self.conv2 = nn.Conv1d(in_channels + growth_rate, growth_rate, kernel_size=kernel_sizes[1],
                               padding="same")

    def forward(self, x):
        # First Composite Function
        out = self.bn1(x)
        out = self.lrelu(out)
        out = self.conv1(out)
        x = torch.cat([x, out], dim=1)

        # Second Composite Function
        out = self.bn2(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        x = torch.cat([x, out], dim=1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class DACB(nn.Module):
    def __init__(self):
        super(DACB, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)

        self.dense1 = DenseBlock(16)
        self.transition1 = TransitionLayer(32)
        self.se1 = SEBlock(64)

        self.dense2 = DenseBlock(64)
        self.transition2 = TransitionLayer(80)
        self.se2 = SEBlock(64)

        self.skip = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1),  
            nn.BatchNorm1d(64), 
            nn.LeakyReLU(0.01)
        )
        self.lrelu = nn.LeakyReLU(0.01)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)

        d = self.dense1(x)
        d = self.transition1(d)
        d = self.se1(d)

        d = self.dense2(d)
        d = self.transition2(d)
        d = self.se2(d)
        
        skip = self.skip(x)
        out = self.lrelu(torch.cat((d, skip), dim=2))
        out = self.gap(out)
        
        return out


class MCDANNNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MCDANNNet, self).__init__()
        self.channels = nn.ModuleList([DACB() for _ in range(12)])  # 12 module per 12 leads
        self.pos_encoder = PositionalEncoding(d_model=64, max_len=12)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12, num_classes)
        )

    def forward(self, x):
        x = x[:, :, ::2]  # Downsample 0.5

        # Z-score normalization 
        mean = x.mean(dim=2, keepdim=True)  
        std = x.std(dim=2, keepdim=True)    
        x = (x - mean) / (std + 1e-8)       
        
        features = []
        for i, channel in enumerate(self.channels):
            lead = x[:, i, :].unsqueeze(1)  
            feat = channel(lead).squeeze(-1) 
            features.append(feat)
        features = torch.stack(features, dim=1)  # [batch, 12, 64]
        features = self.pos_encoder(features)    # Thêm positional encoding
        attn_out, _ = self.attn(features, features, features)
        combined = attn_out.reshape(attn_out.size(0), -1)  # [batch, 12*64]
        return self.classifier(combined)

class MCDANN(L.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = MCDANNNet(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)   
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.learning_rate/100)
        return [optimizer], [scheduler]