import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ECGDataLoader import ECGDataLoader
import random
from pathlib import Path
import shap
import json
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

# Import model directly
from ECGModel.MCDANN import MCDANN

class SHAPVisualizer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.attention_weights = []
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention weights"""
        # Hook for attention weights
        def attention_hook(module, input, output):
            if isinstance(output, tuple):
                _, attention_weights = output
                if attention_weights is not None:
                    self.attention_weights.append(attention_weights.detach().cpu())
        
        # Register hooks for attention
        if hasattr(self.model.model, 'lead_attention'):
            handle = self.model.model.lead_attention.register_forward_hook(attention_hook)
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def calculate_shap_values(self, input_data, num_samples=100):
        """Calculate SHAP values using Integrated Gradients with multiple baselines
        
        Args:
            input_data: Input tensor [1, num_leads, signal_length]
            num_samples: Number of samples for integration
            
        Returns:
            shap_values: SHAP values with same shape as input
        """
        print(f"🔬 Using Enhanced Integrated Gradients with multiple baselines...")
        
        # Move to device and detach from any existing graph
        input_data = input_data.detach().to(self.device)
        
        # Get model prediction for target class
        with torch.no_grad():
            output = self.model(input_data)
            pred_class = torch.argmax(output, dim=1).item()
        
        # Create multiple baselines for more robust estimation
        baselines = []
        
        # 1. Zero baseline
        baselines.append(torch.zeros_like(input_data))
        
        # 2. Mean baseline (per lead)
        mean_baseline = input_data.mean(dim=2, keepdim=True).expand_as(input_data)
        baselines.append(mean_baseline)
        
        # 3. Gaussian noise baselines (3 samples)
        for _ in range(3):
            noise_baseline = torch.randn_like(input_data) * input_data.std()
            baselines.append(noise_baseline)
        
        # Accumulate attributions from all baselines
        all_attributions = []
        
        for baseline_idx, baseline in enumerate(baselines):
            baseline = baseline.to(self.device)
            integrated_grads = torch.zeros_like(input_data).to(self.device)
            
            # Use more integration steps for better accuracy
            steps = min(num_samples, 100)
            
            for i in range(steps + 1):
                # Interpolate between baseline and input
                alpha = i / steps
                
                # Create interpolated input as a new leaf tensor
                interpolated = baseline + alpha * (input_data - baseline)
                interpolated = interpolated.detach().clone()
                interpolated.requires_grad = True
                
                # Forward pass
                output = self.model(interpolated)
                
                # Get score for predicted class
                score = output[0, pred_class]
                
                # Backward pass
                score.backward()
                
                # Accumulate gradients
                if interpolated.grad is not None:
                    integrated_grads += interpolated.grad / steps
            
            # Calculate attributions for this baseline
            attributions = (input_data - baseline) * integrated_grads
            all_attributions.append(attributions)
        
        # Average attributions across all baselines
        final_attributions = torch.stack(all_attributions).mean(dim=0)
        
        # Convert to numpy
        shap_values = final_attributions.detach().cpu().numpy()
        
        print(f"✅ Enhanced Integrated Gradients calculated with {len(baselines)} baselines and {steps + 1} steps each")
        
        return shap_values
    
    def visualize_attention(self, sample_idx=0, save_dir='visualizations', metadata=None):
        """Visualize attention weights between leads"""
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.attention_weights:
            print("No attention weights captured!")
            return None
        
        # Get the first attention weight matrix
        attn = self.attention_weights[sample_idx]  # [batch, num_heads, seq_len, seq_len]
        
        # Handle different attention weight dimensions
        if attn.dim() == 4:
            # [batch, num_heads, seq_len, seq_len] -> [seq_len, seq_len]
            attn = attn.mean(dim=(0, 1))
        elif attn.dim() == 3:
            # [batch, seq_len, seq_len] -> [seq_len, seq_len]
            attn = attn.squeeze(0)
        elif attn.dim() == 2:
            # Already [seq_len, seq_len]
            pass
        
        # Convert to numpy and ensure 2D
        attn = attn.numpy()
        if attn.ndim != 2:
            print(f"Warning: Unexpected attention shape {attn.shape}, attempting to reshape...")
            if attn.ndim == 3 and attn.shape[0] == 1:
                attn = attn.squeeze(0)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        sns.heatmap(attn, 
                    xticklabels=lead_names[:attn.shape[1]], 
                    yticklabels=lead_names[:attn.shape[0]],
                    cmap='YlOrRd', 
                    annot=True, 
                    fmt='.3f',
                    cbar_kws={'label': 'Attention Weight'})
        
        # Create title with patient info and r-peak
        label_dict = {0: "Nhồi máu cơ tim", 1: "Khỏe mạnh"}
        if metadata:
            patient_num = metadata.get('patient_number', 'N/A')
            gt_label = metadata.get('gt_label', 0)
            pred_label = metadata.get('pred_label', 0)
            r_peak_idx = metadata.get('r_peak_index', 'N/A')
            record_name = metadata.get('record_name', 'N/A')
            title = f'Bệnh nhân {patient_num} - {record_name} (R-peak: {r_peak_idx})\nThực tế: {label_dict[gt_label]} | Dự đoán: {label_dict[pred_label]}'
        else:
            title = 'Cross-Lead Attention Weights'
        
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel('Key Leads', fontsize=12)
        plt.ylabel('Query Leads', fontsize=12)
        plt.tight_layout()
        
        # Generate filename with record_name
        if metadata:
            patient_num = metadata.get('patient_number', 'N/A')
            r_peak_idx = metadata.get('r_peak_index', 'N/A')
            record_name = metadata.get('record_name', 'unknown')
            save_path = os.path.join(save_dir, f'patient{patient_num}_{record_name}_{r_peak_idx}_att.png')
        else:
            save_path = os.path.join(save_dir, f'attention_heatmap_sample_{sample_idx}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Attention heatmap saved to: {save_path}")
        
        # Return attention matrix for finding top leads
        return attn
    
    def visualize_shap_top_leads(self, signal, shap_values, pred_label, gt_label, 
                                 save_dir, patient_num, r_peak_idx, record_name,
                                 top_lead_indices, smooth_sigma=2):
        """Visualize ECG signal with SHAP values overlay for top 2 leads only
        
        Args:
            signal: ECG signal [num_leads, signal_length]
            shap_values: SHAP values (list or array)
            pred_label: Predicted label
            gt_label: Ground truth label
            save_dir: Directory to save visualization
            patient_num: Patient number
            r_peak_idx: R-peak index
            record_name: Record name
            top_lead_indices: List of top 2 lead indices
            smooth_sigma: Gaussian smoothing parameter for SHAP values
        """
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        label_dict = {0: "Nhồi máu cơ tim", 1: "Khỏe mạnh"}
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[pred_label]
        else:
            shap_vals = shap_values
        
        # Handle numpy array dimensions
        if isinstance(shap_vals, np.ndarray):
            print(f"📊 Raw SHAP values shape: {shap_vals.shape}")
            # Remove batch dimension if present
            if shap_vals.ndim == 3 and shap_vals.shape[0] == 1:
                shap_vals = shap_vals[0]
            # If still 3D, it's already [num_leads, signal_length]
            elif shap_vals.ndim == 3:
                print(f"⚠️ Unexpected 3D shape, taking first slice")
                shap_vals = shap_vals[0]
        
        print(f"📊 Processed SHAP values shape: {shap_vals.shape}")
        print(f"📊 Signal shape: {signal.shape}")
        
        # Ensure shapes match
        if shap_vals.shape != signal.shape:
            print(f"⚠️ Shape mismatch! Interpolating SHAP values...")
            shap_vals_new = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                shap_vals_new[i] = np.interp(
                    np.linspace(0, 1, signal.shape[1]),
                    np.linspace(0, 1, shap_vals.shape[1]),
                    shap_vals[i]
                )
            shap_vals = shap_vals_new
            print(f"📊 Interpolated SHAP values shape: {shap_vals.shape}")
        
        # Apply Gaussian smoothing to SHAP values for smoother visualization
        print(f"🔧 Applying Gaussian smoothing with sigma={smooth_sigma}...")
        shap_vals_smooth = np.zeros_like(shap_vals)
        for i in range(shap_vals.shape[0]):
            shap_vals_smooth[i] = gaussian_filter1d(shap_vals[i], sigma=smooth_sigma)
        
        # Create 2 rows layout for top 2 leads
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Set background color to light pink/salmon (ECG paper color)
        fig.patch.set_facecolor('#FFE4E1')
        
        # Get signal length
        signal_length = signal.shape[1]
        
        # Calculate grid spacing for 1000Hz sampling rate
        minor_grid_x = 40   # 1mm horizontal (0.04s)
        major_grid_x = 200  # 5mm horizontal (0.2s)
        minor_grid_y = 0.1  # 1mm vertical = 0.1mV
        major_grid_y = 0.5  # 5mm vertical = 0.5mV
        
        # Find y-range for selected leads
        selected_signals = signal[top_lead_indices]
        y_min_global = selected_signals.min()
        y_max_global = selected_signals.max()
        y_min_rounded = np.floor(y_min_global / minor_grid_y) * minor_grid_y
        y_max_rounded = np.ceil(y_max_global / minor_grid_y) * minor_grid_y
        
        # Calculate aspect ratio to make grid squares
        aspect_ratio = minor_grid_x / minor_grid_y
        
        for plot_idx, lead_idx in enumerate(top_lead_indices):
            ax = axes[plot_idx]
            lead_name = lead_names[lead_idx]
            
            # Set background color for each subplot
            ax.set_facecolor('#FFE4E1')
            
            # Draw minor grid (1mm = 0.1mV) - thinner, lighter lines
            ax.set_xticks(np.arange(0, signal_length + minor_grid_x, minor_grid_x), minor=True)
            ax.set_yticks(np.arange(y_min_rounded, y_max_rounded + minor_grid_y, minor_grid_y), minor=True)
            ax.grid(which='minor', color='#FF9999', linestyle='-', linewidth=0.3, alpha=0.5)
            
            # Draw major grid (5mm = 0.5mV) - thicker, darker lines
            ax.set_xticks(np.arange(0, signal_length + major_grid_x, major_grid_x), minor=False)
            ax.set_yticks(np.arange(y_min_rounded, y_max_rounded + major_grid_y, major_grid_y), minor=False)
            ax.grid(which='major', color='#FF6666', linestyle='-', linewidth=0.8, alpha=0.7)
            
            # Get smoothed SHAP values for this lead
            x = np.arange(len(signal[lead_idx]))
            shap_lead = shap_vals_smooth[lead_idx]
            
            # Normalize SHAP values (0 to 1) with percentile clipping to remove outliers
            shap_abs = np.abs(shap_lead)
            shap_95 = np.percentile(shap_abs, 95)  # Use 95th percentile instead of max
            shap_normalized = np.clip(shap_abs / (shap_95 + 1e-8), 0, 1)
            
            # Apply threshold to reduce noise
            threshold = 0.05
            shap_normalized[shap_normalized < threshold] = 0
            
            # Create color array for line segments
            from matplotlib.collections import LineCollection
            
            # Create points for line segments
            points = np.array([x, signal[lead_idx]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create colors with smooth gradient
            colors = np.zeros((len(x) - 1, 4))  # RGBA
            
            for j in range(len(x) - 1):
                importance = shap_normalized[j]
                
                if importance > 0.5:
                    # Orange for important regions
                    colors[j] = [1.0, 0.5, 0.0, 1.0]  # Orange
                else:
                    # Blue for base signal
                    colors[j] = [0.0, 0.2, 0.8, 1.0]  # Blue
            
            # Create LineCollection with colors and thicker line
            lc = LineCollection(segments, colors=colors, linewidth=1.2, zorder=3)
            ax.add_collection(lc)
            
            # Set limits
            ax.set_xlim(0, signal_length)
            ax.set_ylim(y_min_rounded - 0.2, y_max_rounded + 0.2)
            
            # Set aspect ratio to make grid squares
            ax.set_aspect(aspect_ratio, adjustable='box')
            
            # Remove minor tick labels
            ax.tick_params(which='major', labelsize=8)
            ax.tick_params(which='minor', labelleft=False, labelbottom=False)
            
            # Format y-axis labels to show millivolts
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
            
            ax.set_title(f'Lead {lead_name} (Top {plot_idx + 1} Attention)', 
                        fontweight='bold', fontsize=12, pad=8)
            ax.set_ylabel('Amplitude (mV)', fontsize=10)
        
        # Add x-label for bottom
        axes[-1].set_xlabel('Time (ms)', fontsize=10)
        axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        # Create title
        title = (f'SHAP Values - Top 2 Leads - Bệnh nhân {patient_num} - {record_name} (R-peak: {r_peak_idx})\n'
                f'Thực tế: {label_dict[gt_label]} | Dự đoán: {label_dict[pred_label]}')
        
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.998)
        
        # Add custom legend outside the plot, next to top lead (top subplot)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0.0, 0.2, 0.8), label='Basic ECG signal'),
            Patch(facecolor=(1.0, 0.5, 0.0), label='Important region')
        ]
        axes[0].legend(handles=legend_elements, 
                      loc='center left', 
                      bbox_to_anchor=(1.02, 0.5),
                      fontsize=9,
                      frameon=True,
                      fancybox=True,
                      shadow=True)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.995])
        
        save_path = os.path.join(save_dir, 
                                f'patient{patient_num}_{record_name}_{r_peak_idx}_shap_top2.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FFE4E1')
        plt.close()
        print(f"✅ SHAP overlay (top 2 leads) saved to: {save_path}")


def extract_record_name(pt_path):
    """Extract record name from pt_path
    Example: processed_ptb_all/051/s0179lre/signal.pt -> s0179lre
    """
    path_parts = Path(pt_path).parts
    # Get the parent directory name (second to last part)
    if len(path_parts) >= 2:
        return path_parts[-2]
    return "unknown"


def visualize_shap_checkpoint(
    checkpoint_path,
    label_file='ptb_fold.csv',
    data_dir='../',
    batch_size=1,
    num_workers=2,
    split_ratio=0.8,
    sample_before=198,
    sample_after=400,
    num_classes=2,
    learning_rate=1e-3,
    save_dir='shap_visualizations',
    target_patients=None,
    num_shap_samples=200,
    smooth_sigma=3
):
    """Visualize SHAP values and attention weights for specific patients
    
    Args:
        target_patients: List of patient numbers to visualize
        num_shap_samples: Number of samples for SHAP explainer
        smooth_sigma: Gaussian smoothing parameter
    """
    
    # Set random seed
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # Extract fold from checkpoint name
    ckpt_file = Path(checkpoint_path).name
    fold_str = ckpt_file.split('-')[0].replace('[', '').replace(']', '')
    fold = int(fold_str)
    
    print(f"\n🔎 Loading checkpoint: {ckpt_file} (fold {fold})")
    
    # Setup dataloader
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
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = MCDANN(num_classes=num_classes, learning_rate=learning_rate)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Create visualizer
    visualizer = SHAPVisualizer(model, device)
    visualizer.register_hooks()
    
    # Create save directory
    fold_save_dir = os.path.join(save_dir, f'fold_{fold}')
    os.makedirs(fold_save_dir, exist_ok=True)
    
    # Get test samples
    test_loader = dataloader.test_dataloader()
    
    # Collect all records for target patients
    patient_records = defaultdict(list)
    
    print(f"\n📊 Collecting data for patients {target_patients}...")
    
    for batch in test_loader:
        inputs, gt_labels, patient_numbers, r_peak_indices, pt_paths = batch
        
        patient_num = patient_numbers[0].item() if torch.is_tensor(patient_numbers[0]) else patient_numbers[0]
        r_peak_idx = r_peak_indices[0].item() if torch.is_tensor(r_peak_indices[0]) else r_peak_indices[0]
        pt_path = pt_paths[0] if isinstance(pt_paths, list) else pt_paths
        
        if target_patients is None or patient_num in target_patients:
            record_name = extract_record_name(pt_path)
            
            patient_records[patient_num].append({
                'r_peak_idx': r_peak_idx,
                'inputs': inputs.clone(),
                'gt_label': gt_labels[0].item(),
                'pt_path': pt_path,
                'record_name': record_name
            })
    
    print(f"\n🎨 Processing {len(patient_records)} patients...")
    
    # Process each patient
    for patient_num in sorted(patient_records.keys()):
        all_rpeaks = patient_records[patient_num]
        print(f"\n👤 Processing Patient {patient_num} ({len(all_rpeaks)} r-peaks)")
        
        # Create patient directories
        patient_base_dir = os.path.join(fold_save_dir, f'patient_{patient_num}')
        att_dir = os.path.join(patient_base_dir, 'attention')
        shap_dir = os.path.join(patient_base_dir, 'shap')
        os.makedirs(att_dir, exist_ok=True)
        os.makedirs(shap_dir, exist_ok=True)
        
        # Select first r-peak for visualization
        rpeak_data = all_rpeaks[0]
        inputs = rpeak_data['inputs'].to(device)
        
        # Clear previous attention weights
        visualizer.attention_weights = []
        
        # Get prediction and capture attention
        with torch.no_grad():
            outputs = model(inputs)
            pred_label = torch.argmax(outputs, dim=1)[0].item()
            probs = torch.softmax(outputs, dim=1)[0]
        
        print(f"   R-peak: {rpeak_data['r_peak_idx']}, Record: {rpeak_data['record_name']}")
        print(f"   Ground Truth: {'MI' if rpeak_data['gt_label'] == 0 else 'Healthy'}")
        print(f"   Predicted: {'MI' if pred_label == 0 else 'Healthy'} ({probs[pred_label].item():.4f})")
        
        # Visualize attention and get top leads
        metadata = {
            'patient_number': patient_num,
            'r_peak_index': rpeak_data['r_peak_idx'],
            'record_name': rpeak_data['record_name'],
            'gt_label': rpeak_data['gt_label'],
            'pred_label': pred_label
        }
        
        attn_matrix = visualizer.visualize_attention(
            sample_idx=0,
            save_dir=att_dir,
            metadata=metadata
        )
        
        # Find top 2 leads with highest attention (average across rows)
        if attn_matrix is not None:
            # Average attention received by each lead (column sum)
            lead_importance = attn_matrix.mean(axis=0)
            top_2_indices = np.argsort(lead_importance)[-2:][::-1]  # Get top 2, descending
            
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            print(f"   Top 2 leads by attention: {[lead_names[i] for i in top_2_indices]}")
            print(f"   Attention scores: {[f'{lead_importance[i]:.4f}' for i in top_2_indices]}")
            
            # Calculate SHAP values
            print(f"\n🔬 Calculating SHAP values...")
            shap_values = visualizer.calculate_shap_values(inputs, num_samples=num_shap_samples)
            
            # Visualize SHAP for top 2 leads
            visualizer.visualize_shap_top_leads(
                signal=inputs[0].cpu().numpy(),
                shap_values=shap_values,
                pred_label=pred_label,
                gt_label=rpeak_data['gt_label'],
                save_dir=shap_dir,
                patient_num=patient_num,
                r_peak_idx=rpeak_data['r_peak_idx'],
                record_name=rpeak_data['record_name'],
                top_lead_indices=top_2_indices,
                smooth_sigma=smooth_sigma
            )
            
            # Save metadata
            metadata_full = {
                'patient_number': patient_num,
                'r_peak_index': rpeak_data['r_peak_idx'],
                'record_name': rpeak_data['record_name'],
                'pt_path': rpeak_data['pt_path'],
                'gt_label': rpeak_data['gt_label'],
                'pred_label': pred_label,
                'confidence': probs[pred_label].item(),
                'fold': fold,
                'top_2_leads': {
                    'indices': [int(i) for i in top_2_indices],
                    'names': [lead_names[i] for i in top_2_indices],
                    'attention_scores': [float(lead_importance[i]) for i in top_2_indices]
                },
                'num_shap_samples': num_shap_samples,
                'smooth_sigma': smooth_sigma
            }
            
            with open(os.path.join(patient_base_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata_full, f, indent=4)
    
    visualizer.remove_hooks()
    
    print(f"\n✅ All visualizations saved to: {fold_save_dir}")


if __name__ == "__main__":
    checkpoint_path = "./runs/20251024_085235/checkpoints/[4]-ecg-swa.ckpt"
    
    # Visualize specific patients
    visualize_shap_checkpoint(
        checkpoint_path=checkpoint_path,
        target_patients=[1, 32],  # Specify patient numbers
        save_dir='shap_visualizations',
        num_shap_samples=100,  # More samples = better estimation
        smooth_sigma=5 # Adjust smoothing (1-5, higher = smoother)
    )