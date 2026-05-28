```python
import os
import time
import copy
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ==================== Basic Configuration ====================
RANDOM_SEED = 2025
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Combined configuration: 3-class for SEED, batch size dynamically set to 128 or 64 based on GPU memory
num_classes = 3
batch_size = 128 if torch.cuda.is_available() else 64
img_rows, img_cols, num_chan = 8, 9, 4
num_windows = 6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'visualizations', 'RMA_Net_SEED')
CKPT_DIR = os.path.join(SAVE_DIR, 'checkpoints')
CACHE_DIR = os.path.join(SAVE_DIR, 'cache')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SCI_COLORS = {
    'blue': '#4C78A8', 'teal': '#72B7B2', 'green': '#54A24B', 'gold': '#E6C76E',
    'red': '#E45756', 'purple': '#B279A2', 'gray': '#7F7F7F', 'navy': '#3B5BA5', 'coral': '#F58579'
}
BAND_COLORS = [SCI_COLORS['blue'], SCI_COLORS['teal'], SCI_COLORS['gold'], SCI_COLORS['red']]

# Training hyperparameters
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
EARLY_STOP = 10
DROPOUT = 0.5
LAMBDA_REC = 0.30
LAMBDA_CONS = 0.20
LAMBDA_REG = 0.01

# Mask simulation parameters
RANDOM_MISSING_MAX = 0.30
BLOCK_MISSING_PROB = 0.50
MAX_BLOCKS = 2
BLOCK_SIZE_RANGE = (2, 4)

USE_SAVED_CACHE = False
FORCE_RETRAIN = True
RESUME_FROM_CHECKPOINT = False

FULL_CACHE_PATH = os.path.join(CACHE_DIR, 'rma_net_seed_cache.pkl')


# ==================== SEED Data Loading ====================
def load_seed_arrays(x_path, y_path):
    """Load .npy files for SEED dataset."""
    falx = np.load(x_path)
    y = np.load(y_path)
    return falx, y

def build_seed_labels(y_raw):
    """Build one-hot labels from raw SEED labels (3 classes)."""
    one_y = np.array([y_raw[:1126]] * 3).reshape((-1,))
    one_y = one_y - np.min(one_y)
    one_y = one_y.astype(np.int64)
    return F.one_hot(torch.tensor(one_y, dtype=torch.long), num_classes=num_classes).float()

def load_data(subject_index, falx, one_y):
    """
    Load data for a specific subject (15 subjects, each with 3 sessions).
    Returns shape: [Batch, T=6, C=4, H=8, W=9]
    """
    one_falx_1 = falx[subject_index * 3: subject_index * 3 + 3]
    one_falx_1 = one_falx_1.reshape((-1, num_windows, img_rows, img_cols, 5))
    one_falx = one_falx_1[:, :, :, :, 1:5]

    X = one_falx.transpose([0, 1, 4, 2, 3]).reshape((-1, num_windows, num_chan, img_rows, img_cols))
    y = one_y.clone().numpy() if isinstance(one_y, torch.Tensor) else np.copy(one_y)

    return X.astype(np.float32), y.astype(np.float32)


# ==================== Utility Functions ====================
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def init_results_state():
    return {
        'all_acc': [], 'all_precision': [], 'all_recall': [], 'all_f1': [],
        'all_confusion_matrices': [], 'all_losses_per_subject': [], 'all_stability': [],
        'all_preds': [], 'all_labels': [], 'all_probs': [],
        'all_temporal_weights': [], 'all_band_attention': [], 'all_spatial_maps': [],
        'all_mask_examples': [], 'processed_subjects': []
    }

def save_results_cache(state, path):
    with open(path, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

def masked_mse(pred, target, missing_mask):
    diff = (pred - target) * missing_mask
    return torch.mean(diff.pow(2))

def compute_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)
    return acc, precision, recall, f1, cm

def get_class_names(n_classes):
    if n_classes == 2: return ['Low Valence', 'High Valence']
    if n_classes == 3: return ['Negative', 'Neutral', 'Positive']
    return [f'Class {i}' for i in range(n_classes)]


# ==================== Mask Simulation ====================
def generate_random_block_mask(batch_size, h, w, p_random_max=0.3,
                               block_missing_prob=0.5, max_blocks=2,
                               block_size_range=(2, 4), device='cpu'):
    masks = []
    for _ in range(batch_size):
        p_r = random.uniform(0.0, p_random_max)
        m_r = torch.bernoulli(torch.full((h, w), 1.0 - p_r, device=device))

        m_b = torch.ones((h, w), device=device)
        if random.random() < block_missing_prob:
            num_blocks = random.randint(1, max_blocks)
            for _ in range(num_blocks):
                bh = random.randint(block_size_range[0], min(block_size_range[1], h))
                bw = random.randint(block_size_range[0], min(block_size_range[1], w))
                sy = random.randint(0, h - bh)
                sx = random.randint(0, w - bw)
                m_b[sy:sy + bh, sx:sx + bw] = 0.0
        masks.append(m_r * m_b)
    return torch.stack(masks, dim=0)

DEAP_REGION_BLOCKS = {
    'frontal': [(0, 0), (4, 9)],
    'central': [(2, 0), (6, 9)],
    'parietal_occipital': [(5, 0), (8, 9)],
    'left': [(0, 0), (8, 4)],
    'right': [(0, 5), (8, 9)],
}

DEAP_CHANNEL_KEEP_POINTS = {
    'channels8': [(0, 3), (0, 5), (4, 2), (4, 6), (6, 2), (6, 6), (7, 1), (7, 7)],
    'channels16': [(0, 3), (0, 5), (1, 2), (1, 6), (2, 2), (2, 6), (3, 3), (3, 5),
                   (4, 2), (4, 4), (4, 6), (5, 3), (5, 5), (6, 2), (6, 6), (7, 4)],
    'channels24': [(0, 3), (0, 5), (1, 2), (1, 6), (2, 0), (2, 2), (2, 4), (2, 6), (2, 8),
                   (3, 1), (3, 3), (3, 5), (3, 7), (4, 0), (4, 2), (4, 4), (4, 6), (4, 8),
                   (5, 1), (5, 3), (5, 5), (5, 7), (6, 2), (6, 6)]
}

DEAP_BLOCK_SPECS = {
    'small': [(2, 3), (2, 2)],
    'medium': [(3, 4), (3, 3)],
    'large': [(4, 5), (4, 4)],
}

def build_eval_mask(batch_size, h, w, mode='full', ratio=0.0, device='cpu', region='frontal', block_size='small'):
    if mode == 'full': return torch.ones((batch_size, h, w), device=device)
    masks = []
    for _ in range(batch_size):
        m = torch.ones((h, w), device=device)
        if mode == 'random':
            keep_prob = max(0.0, min(1.0, 1.0 - ratio))
            m = torch.bernoulli(torch.full((h, w), keep_prob, device=device))
        elif mode == 'region':
            if region not in DEAP_REGION_BLOCKS: region = 'frontal'
            (y1, x1), (y2, x2) = DEAP_REGION_BLOCKS[region]
            m[y1:y2, x1:x2] = 0.0
        elif mode == 'block':
            if block_size not in DEAP_BLOCK_SPECS: block_size = 'small'
            dims = DEAP_BLOCK_SPECS[block_size]
            bh, bw = random.choice(dims)
            sy = max(0, (h - bh) // 2)
            sx = max(0, (w - bw) // 2)
            m[sy:sy + bh, sx:sx + bw] = 0.0
        elif mode in DEAP_CHANNEL_KEEP_POINTS:
            m.zero_()
            for yy, xx in DEAP_CHANNEL_KEEP_POINTS[mode]:
                if 0 <= yy < h and 0 <= xx < w: m[yy, xx] = 1.0
        else: raise ValueError(f'Unsupported eval mask mode: {mode}')
        masks.append(m)
    return torch.stack(masks, dim=0)


# ==================== RMA-Net Modules ====================
class ReliabilityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x_masked, mask_keep):
        u = x_masked.mean(dim=1, keepdim=True)
        v = (x_masked - u).abs().mean(dim=1, keepdim=True)
        m = mask_keep.unsqueeze(1)
        q = torch.cat([u, v, m], dim=1)
        r = self.net(q)
        rho_t = r.mean(dim=(2, 3), keepdim=True)
        return r, rho_t


class ChannelReconstructor(nn.Module):
    def __init__(self, in_channels=num_chan + 1, out_channels=num_chan):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU())

        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_masked, r):
        y = torch.cat([x_masked, r], dim=1)
        e1 = self.enc1(y)
        e2 = self.enc2(e1)
        z = self.enc3(e2)

        d2 = self.dec2(z)
        if d2.shape[-2:] != e2.shape[-2:]: d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = d2 + e2

        d1 = self.dec1(d2)
        if d1.shape[-2:] != e1.shape[-2:]: d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = d1 + e1

        x_hat = self.out_conv(d1)
        return x_hat


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class QualityGuidedFrequencyAttention(nn.Module):
    def __init__(self, channels=num_chan, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, channels)

    def forward(self, x_bar, r):
        g = (x_bar * r).mean(dim=(2, 3))
        q = F.gelu(self.fc1(g))
        alpha = F.softmax(self.fc2(q), dim=1)
        x_f = x_bar * alpha.unsqueeze(-1).unsqueeze(-1)
        return x_f, alpha


class MultiScaleResidualBackbone(nn.Module):
    def __init__(self, in_channels=num_chan):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.GELU()
        )
        self.block1 = ResidualBlock(32, 64, stride=1)
        self.block2 = ResidualBlock(64, 96, stride=2)
        self.block3 = ResidualBlock(96, 128, stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class QualityGuidedSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8), nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, feat, r):
        s = feat.mean(dim=1, keepdim=True)
        r_small = F.interpolate(r, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        p = torch.cat([s, r_small], dim=1)
        a_s = self.net(p)
        x_s = feat * a_s
        return x_s, a_s


class QualityGuidedTemporalAttention(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=64, eps_rho=1e-6):
        super().__init__()
        self.eps_rho = eps_rho
        self.W_h = nn.Linear(feature_dim, hidden_dim)
        self.w_r = nn.Linear(1, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h_seq, rho_seq):
        B, T, _ = h_seq.shape
        if rho_seq.dim() == 2: rho_seq = rho_seq.unsqueeze(-1)
        elif rho_seq.dim() == 3 and rho_seq.shape[1] == 1 and rho_seq.shape[2] == T: rho_seq = rho_seq.transpose(1, 2)
        elif rho_seq.dim() != 3: raise ValueError(f"rho_seq shape error: {rho_seq.shape}")

        rho_seq = rho_seq.contiguous().view(B, T, 1)
        rho_bar = rho_seq.mean(dim=1, keepdim=True)
        rho_tilde = rho_seq / (rho_bar + self.eps_rho)

        h_proj = self.W_h(h_seq)
        r_proj = self.w_r(rho_tilde)
        a = torch.tanh(h_proj + r_proj)
        e = self.v(a).squeeze(-1)
        alpha = F.softmax(e, dim=1)
        z = torch.sum(alpha.unsqueeze(-1) * h_seq, dim=1)
        return z, alpha


class RMANet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.reliability = ReliabilityEstimator()
        self.reconstructor = ChannelReconstructor(in_channels=num_chan + 1, out_channels=num_chan)
        self.freq_att = QualityGuidedFrequencyAttention(channels=num_chan, hidden_dim=16)
        self.backbone = MultiScaleResidualBackbone(in_channels=num_chan)
        self.spatial_att = QualityGuidedSpatialAttention()
        self.temporal_att = QualityGuidedTemporalAttention(feature_dim=128, hidden_dim=64)

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.last_band_attention = None
        self.last_spatial_attention = None
        self.last_temporal_attention = None

    def forward_once(self, x, inject_mask=True, eval_mask_keep=None):
        b, t, c, h, w = x.shape

        if eval_mask_keep is not None:
            mask_keep = eval_mask_keep
        elif inject_mask:
            mask_keep = generate_random_block_mask(
                batch_size=b, h=h, w=w,
                p_random_max=RANDOM_MISSING_MAX, block_missing_prob=BLOCK_MISSING_PROB,
                max_blocks=MAX_BLOCKS, block_size_range=BLOCK_SIZE_RANGE, device=x.device,
            )
        else:
            mask_keep = torch.ones((b, h, w), device=x.device)

        mask_expand = mask_keep.unsqueeze(1).unsqueeze(1)
        x_masked = x * mask_expand

        time_features, rho_seq, recon_list, reliab_list, band_weights, spatial_weights = [], [], [], [], [], []

        for ti in range(t):
            x_t = x[:, ti]
            x_masked_t = x_masked[:, ti]
            mask_t = mask_keep

            r_t, rho_t = self.reliability(x_masked_t, mask_t)
            x_hat_t = self.reconstructor(x_masked_t, r_t)
            x_bar_t = x_masked_t + (1.0 - mask_t.unsqueeze(1)) * x_hat_t

            x_f_t, alpha_f_t = self.freq_att(x_bar_t, r_t)
            feat_t = self.backbone(x_f_t)
            x_s_t, a_s_t = self.spatial_att(feat_t, r_t)
            h_t = F.adaptive_avg_pool2d(x_s_t, output_size=1).flatten(1)

            time_features.append(h_t)
            rho_seq.append(rho_t.view(rho_t.size(0), 1))
            recon_list.append(x_hat_t)
            reliab_list.append(r_t)
            band_weights.append(alpha_f_t)
            spatial_weights.append(a_s_t)

        h_seq = torch.stack(time_features, dim=1)
        rho_seq = torch.stack(rho_seq, dim=1)
        z, alpha_t = self.temporal_att(h_seq, rho_seq)
        logits = self.classifier(z)

        x_hat = torch.stack(recon_list, dim=1)
        r_all = torch.stack(reliab_list, dim=1)
        band_weights = torch.stack(band_weights, dim=1)
        spatial_weights = torch.stack(spatial_weights, dim=1)

        self.last_band_attention = band_weights.detach()
        self.last_spatial_attention = spatial_weights.detach()
        self.last_temporal_attention = alpha_t.detach()

        return {
            'logits': logits, 'x_hat': x_hat, 'r': r_all,
            'mask_keep': mask_keep, 'x_masked': x_masked,
            'band_attention': band_weights, 'spatial_attention': spatial_weights,
            'temporal_attention': alpha_t,
        }

    def forward(self, x, inject_mask=True, eval_mask_keep=None):
        return self.forward_once(x, inject_mask=inject_mask, eval_mask_keep=eval_mask_keep)


# ==================== Visualization Functions ====================
def safe_plot(plot_func, name, save_path=None):
    try:
        if save_path is not None: os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plot_func()
        print(f"  [OK] {name} saved")
        return True
    except Exception as e:
        print(f"  [SKIP] {name} failed: {str(e)[:120]}...")
        return False

def plot_accuracy_bar(short_name, all_acc, save_path):
    def _plot():
        plt.figure(figsize=(14, 6))
        bars = plt.bar(short_name, all_acc, color=SCI_COLORS['blue'], alpha=0.82, edgecolor='black', linewidth=0.6)
        mean_acc = np.mean(all_acc)
        plt.axhline(y=mean_acc, color='r', linestyle='--', label=f'Mean: {mean_acc:.4f}')
        y_max = max(all_acc) if len(all_acc) > 0 else 1.0
        plt.ylim(0, y_max * 1.16 if y_max > 0 else 1.0)
        for bar, acc in zip(bars, all_acc):
            plt.text(bar.get_x() + bar.get_width()/2, acc + max(y_max*0.012, 0.008), f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        plt.title('Accuracy per Subject')
        plt.xlabel('Subject'); plt.ylabel('Accuracy'); plt.xticks(rotation=45); plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    safe_plot(_plot, 'Accuracy Bar', save_path)

def plot_boxplot(all_accs, all_f1s, save_path):
    def _plot():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].boxplot(all_accs, patch_artist=True, boxprops=dict(facecolor=SCI_COLORS['teal'], alpha=0.72), medianprops=dict(color='red', linewidth=2))
        axes[0].set_ylabel('Accuracy'); axes[0].set_title('Accuracy Distribution'); axes[0].set_xticklabels(['Model']); axes[0].grid(True, alpha=0.3)
        x = np.ones(len(all_accs)) + np.random.normal(0, 0.04, len(all_accs)); axes[0].scatter(x, all_accs, alpha=0.6, c='blue', s=30)
        axes[1].boxplot(all_f1s, patch_artist=True, boxprops=dict(facecolor=SCI_COLORS['coral'], alpha=0.72), medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('F1 Score'); axes[1].set_title('F1 Score Distribution'); axes[1].set_xticklabels(['Model']); axes[1].grid(True, alpha=0.3)
        x = np.ones(len(all_f1s)) + np.random.normal(0, 0.04, len(all_f1s)); axes[1].scatter(x, all_f1s, alpha=0.6, c='red', s=30)
        plt.suptitle('Model Performance Distribution'); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    safe_plot(_plot, 'Boxplot', save_path)

def plot_loss_curves(all_losses_per_subject, short_name, save_path):
    def _plot():
        num_subjects = len(all_losses_per_subject)
        fig_width = max(16, min(26, 12 + num_subjects * 0.25))
        plt.figure(figsize=(fig_width, 8))
        for i, losses in enumerate(all_losses_per_subject):
            label = f'Subject {short_name[i]}' if i < len(short_name) else f'Subject {i+1:02d}'
            plt.plot(losses, label=label, alpha=0.8, linewidth=1.3)
        plt.title('Loss over Epochs (All Subjects)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True); plt.subplots_adjust(right=0.78)
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    safe_plot(_plot, 'Loss Curves', save_path)

def plot_confusion_matrix_heatmap(cm, save_path, class_names=None):
    def _plot():
        names = class_names if class_names is not None else get_class_names(cm.shape[0])
        cm_int = np.asarray(cm, dtype=np.int32)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, square=True, cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix'); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    safe_plot(_plot, 'Confusion Matrix', save_path)

def plot_roc_curves(all_labels, all_probs, num_classes, save_path):
    def _plot():
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        all_labels_one_hot = np.eye(num_classes)[all_labels_arr]
        plt.figure(figsize=(8, 6))
        valid_curve_num = 0
        for i in range(num_classes):
            y_true = all_labels_one_hot[:, i]
            if len(np.unique(y_true)) < 2: continue
            fpr, tpr, _ = roc_curve(y_true, all_probs_arr[:, i]); roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})'); valid_curve_num += 1
        if valid_curve_num == 0: raise ValueError('No valid ROC curves can be computed')
        plt.plot([0, 1], [0, 1], 'k--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves'); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    safe_plot(_plot, 'ROC Curves', save_path)

def plot_temporal_attention_weights(temporal_weights, save_path):
    def _plot():
        mean_temporal = np.asarray(temporal_weights).mean(axis=0)
        std_temporal = np.asarray(temporal_weights).std(axis=0)
        plt.figure(figsize=(8.8, 5.4))
        x = np.arange(1, len(mean_temporal) + 1)
        bars = plt.bar(x, mean_temporal, yerr=std_temporal, capsize=5, color=SCI_COLORS['blue'], alpha=0.82, edgecolor='black', linewidth=0.6)
        plt.xlabel('Time Window'); plt.ylabel('Attention Weight'); plt.title('Temporal Attention Weights'); plt.xticks(x); plt.grid(True, alpha=0.25, axis='y')
        y_max = float(np.max(mean_temporal + std_temporal)) if len(mean_temporal) > 0 else 1.0
        plt.ylim(0, y_max * 1.20 if y_max > 0 else 1.0)
        for bar, mean_val, std_val in zip(bars, mean_temporal, std_temporal):
            plt.text(bar.get_x()+bar.get_width()/2, mean_val + std_val + max(y_max*0.02, 0.01), f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout(); plt.savefig(save_path, dpi=180, bbox_inches='tight'); plt.close()
    safe_plot(_plot, 'Temporal Attention Weights', save_path)

def plot_frequency_importance(band_values, save_path):
    def _plot():
        values = np.asarray(band_values, dtype=np.float32).reshape(-1)
        total = float(np.sum(values)); values = values / total if total > 0 else values
        bands = ['Theta', 'Alpha', 'Beta', 'Gamma']
        plt.figure(figsize=(8.5, 5.2))
        bars = plt.bar(bands, values, color=BAND_COLORS, alpha=0.88, edgecolor='black', linewidth=0.6)
        y_max = float(np.max(values)) if len(values) > 0 else 1.0; plt.ylim(0, y_max * 1.16 if y_max > 0 else 1.0)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x()+bar.get_width()/2, val + max(y_max*0.02, 0.008), f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        plt.xlabel('Frequency Band'); plt.ylabel('Normalized Importance'); plt.title('Frequency Band Importance'); plt.grid(True, alpha=0.25, axis='y'); plt.tight_layout(); plt.savefig(save_path, dpi=170, bbox_inches='tight'); plt.close()
    safe_plot(_plot, 'Frequency Importance', save_path)

def plot_spatial_attention_map(spatial_weights, save_path):
    def _plot():
        spatial_arr = np.asarray(spatial_weights, dtype=np.float32)
        if spatial_arr.ndim != 2: raise ValueError(f'Expected 2D map, got shape {spatial_arr.shape}')
        center = float(np.median(spatial_arr)); centered = spatial_arr - center
        vmax = float(np.percentile(np.abs(centered), 95)) if centered.size > 0 else 1.0
        if np.isclose(vmax, 0.0): vmax = max(float(np.max(np.abs(centered))), 1.0)
        plt.figure(figsize=(7.2, 5.8))
        im = plt.imshow(centered, cmap='RdBu_r', interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax)
        cs = plt.contour(centered, levels=6, colors='k', linewidths=0.45, alpha=0.55); plt.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
        plt.title('Spatial Discriminative Attention Map'); plt.xlabel('Width'); plt.ylabel('Height'); cbar = plt.colorbar(im); cbar.set_label('Centered Attention')
        plt.tight_layout(); plt.savefig(save_path, dpi=190, bbox_inches='tight'); plt.close()
    safe_plot(_plot, 'Spatial Attention Map', save_path)

def plot_classwise_metrics(all_labels, all_preds, save_path):
    def _plot():
        classes = np.unique(np.concatenate([np.asarray(all_labels), np.asarray(all_preds)]))
        names = [get_class_names(num_classes)[int(c)] if int(c) < len(get_class_names(num_classes)) else f'Class {int(c)}' for c in classes]
        precisions = precision_score(all_labels, all_preds, average=None, labels=classes, zero_division=0)
        recalls = recall_score(all_labels, all_preds, average=None, labels=classes, zero_division=0)
        f1s = f1_score(all_labels, all_preds, average=None, labels=classes, zero_division=0)
        x = np.arange(len(classes)); width = 0.24
        plt.figure(figsize=(10, 5.5))
        plt.bar(x - width, precisions, width=width, label='Precision', color='#4C78A8')
        plt.bar(x, recalls, width=width, label='Recall', color='#72B7B2')
        plt.bar(x + width, f1s, width=width, label='F1-score', color='#E45756')
        for xpos, vals in [(x - width, precisions), (x, recalls), (x + width, f1s)]:
            for xi, yi in zip(xpos, vals): plt.text(xi, yi + 0.01, f'{yi:.2f}', ha='center', va='bottom', fontsize=8)
        plt.ylim(0, min(1.08, max(float(np.max([precisions, recalls, f1s])), 0.1) * 1.18)); plt.xticks(x, names); plt.ylabel('Score'); plt.title('Class-wise Precision / Recall / F1'); plt.grid(True, alpha=0.25, axis='y'); plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=170, bbox_inches='tight'); plt.close()
    safe_plot(_plot, 'Classwise Metrics', save_path)

def plot_subject_metrics_heatmap(subjects, acc_values, f1_values, save_path):
    def _plot():
        data = np.vstack([acc_values, f1_values]); plt.figure(figsize=(max(18, len(subjects) * 0.58), 4.3))
        sns.heatmap(data, annot=True, fmt='.3f', annot_kws={'size': 8}, cmap='YlGnBu', cbar_kws={'label': 'Score'}, xticklabels=subjects, yticklabels=['Accuracy', 'F1-score'], linewidths=0.35, linecolor='white')
        plt.xticks(rotation=0, fontsize=10); plt.yticks(rotation=90, fontsize=11); plt.title('Subject-wise Performance Heatmap'); plt.tight_layout(); plt.savefig(save_path, dpi=190, bbox_inches='tight'); plt.close()
    safe_plot(_plot, 'Subject Metrics Heatmap', save_path)

def plot_stability_summary(all_stability, save_path):
    def _plot():
        keys = ['full','random_10','random_20','random_30','random_50',
                'block_small','block_medium','block_large',
                'region_frontal','region_central','region_parietal_occipital',
                'channels8','channels16','channels24']
        labels = ['Full','R10','R20','R30','R50','B-S','B-M','B-L','Frontal','Central','ParOcc','8ch','16ch','24ch']
        acc_vals = [np.mean([x[k]['acc'] for x in all_stability]) for k in keys]
        f1_vals = [np.mean([x[k]['f1'] for x in all_stability]) for k in keys]
        x = np.arange(len(keys)); width = 0.36
        plt.figure(figsize=(16, 6.2))
        plt.bar(x-width/2, acc_vals, width, label='Accuracy', color=SCI_COLORS['blue'], alpha=0.85)
        plt.bar(x+width/2, f1_vals, width, label='F1', color=SCI_COLORS['coral'], alpha=0.85)
        for xi, yi in zip(x-width/2, acc_vals): plt.text(xi, yi+0.008, f'{yi:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        for xi, yi in zip(x+width/2, f1_vals): plt.text(xi, yi+0.008, f'{yi:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        plt.xticks(x, labels, rotation=20); plt.ylabel('Score'); plt.title('Stability under Incomplete Inputs'); plt.grid(True, axis='y', alpha=0.2); plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=180, bbox_inches='tight'); plt.close()
    safe_plot(_plot, 'Stability Summary', save_path)

def plot_mask_examples(mask_examples, save_path):
    def _plot():
        if len(mask_examples) == 0: raise ValueError('No mask examples')
        n = min(4, len(mask_examples)); fig, axes = plt.subplots(1, n, figsize=(3.5*n, 3.3))
        axes = np.atleast_1d(axes)
        for ax, item in zip(axes, mask_examples[:n]):
            ax.imshow(item, cmap='gray', vmin=0, vmax=1); ax.set_title('Keep Mask'); ax.axis('off')
        plt.suptitle('Incomplete-input Mask Examples'); plt.tight_layout(); plt.savefig(save_path, dpi=180, bbox_inches='tight'); plt.close()
    safe_plot(_plot, 'Mask Examples', save_path)


def collect_attention_and_probs(model, loader):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    temporal_list, band_list, spatial_list, mask_examples = [], [], [], []
    with torch.no_grad():
        for bi, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs, inject_mask=False)
            logits = out['logits']; probs = F.softmax(logits, dim=1); preds = logits.argmax(dim=1)
            preds_all.extend(preds.cpu().numpy().tolist()); labels_all.extend(labels.cpu().numpy().tolist()); probs_all.extend(probs.cpu().numpy().tolist())
            if out['temporal_attention'] is not None: temporal_list.append(out['temporal_attention'].cpu().numpy())
            if out['band_attention'] is not None: band_list.append(out['band_attention'].mean(dim=1).cpu().numpy())
            if out['spatial_attention'] is not None: spatial_list.append(out['spatial_attention'].mean(dim=(0,1,2)).cpu().numpy())
            if bi < 4: mask_examples.append(out['mask_keep'][0].cpu().numpy())
    temporal = np.vstack(temporal_list) if temporal_list else None
    band = np.vstack(band_list).mean(axis=0) if band_list else None
    spatial = np.mean(np.stack(spatial_list, axis=0), axis=0) if spatial_list else None
    return {'preds': preds_all, 'labels': labels_all, 'probs': probs_all, 'temporal': temporal, 'band': band, 'spatial': spatial, 'mask_examples': mask_examples}


# ==================== Loss and Training ====================
def compute_total_loss(model, x, labels, training=True):
    out_masked = model(x, inject_mask=training)
    logits_masked = out_masked['logits']
    x_hat = out_masked['x_hat']
    mask_keep = out_masked['mask_keep']
    r = out_masked['r']

    cls_loss = F.cross_entropy(logits_masked, labels)

    missing_mask = (1.0 - mask_keep).unsqueeze(1).unsqueeze(1)
    rec_loss = masked_mse(x_hat, x, missing_mask)

    with torch.no_grad() if training else torch.enable_grad():
        out_full = model(x, inject_mask=False)
    p_full = F.softmax(out_full['logits'], dim=1)
    log_p_masked = F.log_softmax(logits_masked, dim=1)
    cons_loss = F.kl_div(log_p_masked, p_full, reduction='batchmean')

    mask_prior = mask_keep.unsqueeze(1).unsqueeze(1).expand_as(r)
    reg_loss = F.mse_loss(r, mask_prior)

    total_loss = cls_loss + LAMBDA_REC * rec_loss + LAMBDA_CONS * cons_loss + LAMBDA_REG * reg_loss
    return total_loss, {'cls': float(cls_loss), 'rec': float(rec_loss), 'cons': float(cons_loss), 'reg': float(reg_loss)}

def train_one_fold(X_train, y_train, X_test, y_test, fold_id=0):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.argmax(1)))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.argmax(1)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    model = RMANet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_acc = 0.0
    best_state = None
    trigger_times = 0
    epoch_losses = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, _ = compute_total_loss(model, inputs, labels, training=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _ = compute_total_loss(model, inputs, labels, training=True)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / max(len(train_loader), 1)
        epoch_losses.append(epoch_loss)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs, inject_mask=False)
                val_preds.extend(out['logits'].argmax(dim=1).cpu().numpy().tolist())
                val_labels.extend(labels.cpu().numpy().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)

        print(f'    Epoch {epoch + 1:03d} | loss={epoch_loss:.4f} | val_acc={val_acc:.4f}')

        if val_acc > best_acc:
            best_acc, best_state, trigger_times = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            trigger_times += 1
            if trigger_times >= EARLY_STOP: break

    if best_state is not None: model.load_state_dict(best_state)
    analysis = collect_attention_and_probs(model, test_loader)
    acc, precision, recall, f1, cm = compute_metrics(analysis['labels'], analysis['preds'])

    fold_ckpt_path = os.path.join(CKPT_DIR, f'best_fold_{fold_id + 1}.pth')
    torch.save({'model_state_dict': model.state_dict()}, fold_ckpt_path)

    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm,
        'loss_curve': epoch_losses, 'ckpt': fold_ckpt_path,
        **analysis
    }

def evaluate_under_missing_condition(model, test_loader, mode='full', ratio=0.0, region='frontal', block_size='small'):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            b, _, _, h, w = inputs.shape
            eval_mask_keep = build_eval_mask(b, h, w, mode=mode, ratio=ratio, region=region, block_size=block_size, device=inputs.device)
            out = model(inputs, inject_mask=False, eval_mask_keep=eval_mask_keep)
            preds_all.extend(torch.argmax(out['logits'], dim=1).cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())
    acc, precision, recall, f1, cm = compute_metrics(labels_all, preds_all)
    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm}

def train_subject(X, y, n_splits=10):
    labels_np = y.argmax(1)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(labels_np)), labels_np)):
        print(f'\n{"=" * 60}\nFold {fold + 1}/{n_splits}\n{"=" * 60}')
        result = train_one_fold(X[train_idx], y[train_idx], X[test_idx], y[test_idx], fold_id=fold)

        test_dataset = TensorDataset(torch.FloatTensor(X[test_idx]), torch.LongTensor(y[test_idx].argmax(1)))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

        model_eval = RMANet(num_classes=num_classes).to(device)
        model_eval.load_state_dict(torch.load(result['ckpt'], map_location=device)['model_state_dict'])

        result['stability'] = {
            'full': evaluate_under_missing_condition(model_eval, test_loader, mode='full'),
            'random_10': evaluate_under_missing_condition(model_eval, test_loader, mode='random', ratio=0.10),
            'random_20': evaluate_under_missing_condition(model_eval, test_loader, mode='random', ratio=0.20),
            'random_30': evaluate_under_missing_condition(model_eval, test_loader, mode='random', ratio=0.30),
            'random_50': evaluate_under_missing_condition(model_eval, test_loader, mode='random', ratio=0.50),
            'block_small': evaluate_under_missing_condition(model_eval, test_loader, mode='block', block_size='small'),
            'block_medium': evaluate_under_missing_condition(model_eval, test_loader, mode='block', block_size='medium'),
            'block_large': evaluate_under_missing_condition(model_eval, test_loader, mode='block', block_size='large'),
            'region_frontal': evaluate_under_missing_condition(model_eval, test_loader, mode='region', region='frontal'),
            'region_central': evaluate_under_missing_condition(model_eval, test_loader, mode='region', region='central'),
            'region_parietal_occipital': evaluate_under_missing_condition(model_eval, test_loader, mode='region', region='parietal_occipital'),
            'channels8': evaluate_under_missing_condition(model_eval, test_loader, mode='channels8'),
            'channels16': evaluate_under_missing_condition(model_eval, test_loader, mode='channels16'),
            'channels24': evaluate_under_missing_condition(model_eval, test_loader, mode='channels24'),
        }

        print(f"Fold {fold + 1} -> Acc: {result['acc']:.4f}, F1: {result['f1']:.4f}")
        fold_results.append(result)

    mean_acc = float(np.mean([r['acc'] for r in fold_results]))
    mean_precision = float(np.mean([r['precision'] for r in fold_results]))
    mean_recall = float(np.mean([r['recall'] for r in fold_results]))
    mean_f1 = float(np.mean([r['f1'] for r in fold_results]))
    sum_cm = np.sum(np.stack([r['cm'] for r in fold_results], axis=0), axis=0).astype(int)

    max_len = max(len(r['loss_curve']) for r in fold_results)
    padded_losses = [r['loss_curve'] + [r['loss_curve'][-1]] * (max_len - len(r['loss_curve'])) for r in fold_results]
    mean_losses = np.mean(np.array(padded_losses), axis=0).tolist()

    stability_keys = ['full','random_10','random_20','random_30','random_50','block_small','block_medium','block_large','region_frontal','region_central','region_parietal_occipital','channels8','channels16','channels24']
    stability_summary = {key: {'acc': float(np.mean([r['stability'][key]['acc'] for r in fold_results])), 'f1': float(np.mean([r['stability'][key]['f1'] for r in fold_results]))} for key in stability_keys}

    return {
        'acc': mean_acc, 'precision': mean_precision, 'recall': mean_recall, 'f1': mean_f1, 'cm': sum_cm,
        'loss_curve': mean_losses, 'fold_results': fold_results, 'stability': stability_summary,
        'preds': sum([r['preds'] for r in fold_results], []), 'labels': sum([r['labels'] for r in fold_results], []),
        'probs': sum([r['probs'] for r in fold_results], []),
        'temporal': np.vstack([r['temporal'] for r in fold_results if r['temporal'] is not None]) if any(r['temporal'] is not None for r in fold_results) else None,
        'band': np.mean(np.stack([r['band'] for r in fold_results if r['band'] is not None], axis=0), axis=0) if any(r['band'] is not None for r in fold_results) else None,
        'spatial': np.mean(np.stack([r['spatial'] for r in fold_results if r['spatial'] is not None], axis=0), axis=0) if any(r['spatial'] is not None for r in fold_results) else None,
        'mask_examples': sum([r['mask_examples'] for r in fold_results], [])
    }


# ==================== Main ====================
if __name__ == '__main__':
    # TODO: Replace with your actual paths to the SEED dataset .npy files
    # Example: seed_x_path = "/path/to/your/SEED_DE0.5s/t6x_89.npy"
    seed_x_path = "path/to/your/SEED_DE0.5s/t6x_89.npy"
    seed_y_path = "path/to/your/SEED_DE0.5s/t6y_89.npy"

    # SEED has 15 subjects
    subject_list = [f"{i:02d}" for i in range(1, 16)]
    falx, seed_y_raw = load_seed_arrays(seed_x_path, seed_y_path)
    one_y = build_seed_labels(seed_y_raw)

    start = time.time()

    print(f'\n{"=" * 80}')
    print('RMA-Net + SEED Dataset')
    print(f'seed_x_path = {seed_x_path}')
    print(f'seed_y_path = {seed_y_path}')
    print(f'batch_size  = {batch_size}')
    print(f'device      = {device}')
    print(f'{"=" * 80}\n')

    if USE_SAVED_CACHE and (not FORCE_RETRAIN) and os.path.exists(FULL_CACHE_PATH):
        with open(FULL_CACHE_PATH, 'rb') as f: state = pickle.load(f)
        print('Loaded cached results.')
    else:
        state = init_results_state()
        processed_subjects = set(state['processed_subjects']) if RESUME_FROM_CHECKPOINT else set()

        for sub_idx, sub in enumerate(subject_list):
            if sub in processed_subjects:
                print(f'[RESUME] Skip subject {sub}')
                continue

            print(f'\n{"#" * 80}')
            print(f'Training Subject {sub} ({sub_idx + 1}/{len(subject_list)})')
            print(f'{"#" * 80}')

            X, y = load_data(sub_idx, falx, one_y)
            result = train_subject(X, y, n_splits=10)

            state['processed_subjects'].append(sub)
            state['all_acc'].append(result['acc'])
            state['all_precision'].append(result['precision'])
            state['all_recall'].append(result['recall'])
            state['all_f1'].append(result['f1'])
            state['all_confusion_matrices'].append(result['cm'])
            state['all_losses_per_subject'].append(result['loss_curve'])
            state['all_stability'].append(result['stability'])
            state['all_preds'].extend(result['preds'])
            state['all_labels'].extend(result['labels'])
            state['all_probs'].extend(result['probs'])

            if result['temporal'] is not None: state['all_temporal_weights'].append(result['temporal'])
            if result['band'] is not None: state['all_band_attention'].append(result['band'])
            if result['spatial'] is not None: state['all_spatial_maps'].append(result['spatial'])
            state['all_mask_examples'].extend(result['mask_examples'])

            print(f"[Subject {sub}] Acc={result['acc']:.4f}, Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, F1={result['f1']:.4f}")
            save_results_cache(state, FULL_CACHE_PATH)

    print(f'\n{"=" * 80}')
    print('Final Subject-wise Results')
    print(f'Accuracy : {np.mean(state["all_acc"]):.4f} ± {np.std(state["all_acc"]):.4f}')
    print(f'Precision: {np.mean(state["all_precision"]):.4f}')
    print(f'Recall   : {np.mean(state["all_recall"]):.4f}')
    print(f'F1-score : {np.mean(state["all_f1"]):.4f}')
    print(f'Processed subjects: {len(state["processed_subjects"])}')

    if len(state['all_stability']) > 0:
        keys = ['full','random_10','random_20','random_30','random_50','block_small','block_medium','block_large','region_frontal','region_central','region_parietal_occipital','channels8','channels16','channels24']
        print('-' * 80)
        print('Stability summary:')
        for key in keys: print(f'{key:>26s} -> Acc={np.mean([x[key]["acc"] for x in state["all_stability"]]):.4f}, F1={np.mean([x[key]["f1"] for x in state["all_stability"]]):.4f}')

    print('\n' + '=' * 50 + '\nGenerating Visualizations...\n' + '=' * 50)

    if len(state['all_acc']) > 0:
        plot_accuracy_bar(subject_list[:len(state['all_acc'])], state['all_acc'], os.path.join(SAVE_DIR, 'accuracy_per_subject.png'))
        plot_subject_metrics_heatmap(subject_list[:len(state['all_acc'])], np.array(state['all_acc']), np.array(state['all_f1']), os.path.join(SAVE_DIR, 'subject_metrics_heatmap.png'))
        plot_boxplot(state['all_acc'], state['all_f1'], os.path.join(SAVE_DIR, 'boxplot.png'))
        plot_loss_curves(state['all_losses_per_subject'], subject_list[:len(state['all_losses_per_subject'])], os.path.join(SAVE_DIR, 'loss_curves.png'))

    if len(state['all_confusion_matrices']) > 0: plot_confusion_matrix_heatmap(np.sum(np.stack(state['all_confusion_matrices'], axis=0), axis=0).astype(int), os.path.join(SAVE_DIR, 'confusion_matrix.png'), class_names=get_class_names(num_classes))
    if len(state['all_labels']) > 0 and len(state['all_probs']) > 0:
        plot_roc_curves(state['all_labels'], state['all_probs'], num_classes, os.path.join(SAVE_DIR, 'roc_curves.png'))
        plot_classwise_metrics(state['all_labels'], state['all_preds'], os.path.join(SAVE_DIR, 'classwise_metrics.png'))
    if len(state['all_temporal_weights']) > 0: plot_temporal_attention_weights(np.vstack(state['all_temporal_weights']), os.path.join(SAVE_DIR, 'temporal_attention_weights.png'))
    if len(state['all_band_attention']) > 0: plot_frequency_importance(np.mean(np.stack(state['all_band_attention'], axis=0), axis=0), os.path.join(SAVE_DIR, 'frequency_importance.png'))
    if len(state['all_spatial_maps']) > 0: plot_spatial_attention_map(np.mean(np.stack(state['all_spatial_maps'], axis=0), axis=0), os.path.join(SAVE_DIR, 'spatial_attention_map.png'))
    if len(state['all_stability']) > 0: plot_stability_summary(state['all_stability'], os.path.join(SAVE_DIR, 'stability_summary.png'))
    if len(state['all_mask_examples']) > 0: plot_mask_examples(state['all_mask_examples'], os.path.join(SAVE_DIR, 'mask_examples.png'))

    print(f'Save dir: {SAVE_DIR}')
    print(f'Total time: {(time.time() - start) / 60:.2f} min\n{"=" * 80}')
```
