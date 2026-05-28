# CG
# RMA-Net: Reliability‑guided Masked Attention Network for EEG Emotion Recognition

This repository contains the PyTorch implementation of **RMA‑Net**, a deep learning framework designed for robust EEG‑based emotion recognition under incomplete input conditions. The model is introduced in **Chapter 5** of the associated thesis/publication and is evaluated on two widely used affective EEG datasets: **DEAP** and **SEED**.

## Overview

RMA‑Net jointly addresses two challenges:
- **Missing electrode signals** (e.g., poor contact, artefacts) through a trainable mask simulation, reliability estimation, and channel reconstruction.
- **Multi‑scale spatio‑temporal feature learning** via quality‑guided frequency, spatial, and temporal attention mechanisms.

The architecture follows a modular design:
1. **Hybrid mask generation** – simulates random and block‑wise missing patterns.
2. **Reliability estimator** – predicts a per‑channel confidence map.
3. **Channel reconstructor** – inpaints missing channels in a learned encoder‑decoder.
4. **Quality‑guided attention** – refines features using reliability scores in frequency, spatial, and temporal domains.
5. **Multi‑scale residual backbone** – extracts discriminative EEG representations.
6. **Classifier** – maps the final temporal‑aggregated feature to emotion categories.

## Repository Structure

```
.
├── rma_net_seed.py          # Training script for SEED dataset (3-class)
├── rma_net_deap.py          # Training script for DEAP dataset (2-class valence/arousal)
├── visualizations/          # Output directory for plots and checkpoints
│   ├── RMA_Net_SEED/        # SEED results
│   └── RMA_Net/             # DEAP results
└── README.md
```

## Datasets

### DEAP
The DEAP script (`rma_net_deap.py`) expects preprocessed data in `.mat` files, one per subject (e.g., `DE_s01.mat`), containing:
- `data`: EEG segments shaped `[trials, channels, time_points]` (the code transposes and reshapes into `[samples, 6 windows, 4 bands, 8×9 grid]`)
- `valence_labels` / `arousal_labels`: binary labels per trial

**Note:** The preprocessing pipeline (baseline removal, segmentation, DE feature extraction) should be applied *beforehand*. The provided script assumes that input data has already been transformed into the 4‑band (θ, α, β, γ) × 8×9 grid representation.

### SEED
The SEED script (`rma_net_seed.py`) loads two `.npy` files containing the pre‑extracted features and labels for 15 subjects. The code automatically reshapes them into `[samples, 6 windows, 4 bands, 8×9 grid]` and uses a 3‑class label scheme (Negative / Neutral / Positive).

## Requirements

- Python 3.8+
- PyTorch ≥ 1.10 (with CUDA support recommended)
- NumPy, SciPy
- scikit‑learn
- Matplotlib, Seaborn
- (optional) TensorBoard / wandb for logging

Install the required packages:

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn
```

## Usage

### 1. Prepare your dataset

Place the DEAP or SEED files in a known directory. For DEAP, the folder should contain 32 files named `DE_s01.mat` … `DE_s32.mat`. For SEED, you need two files, e.g., `t6x_89.npy` and `t6y_89.npy`.

### 2. Configure paths

Open the corresponding script and modify the dataset path at the bottom (`if __name__ == '__main__':` block):

- For **DEAP** (`rma_net_deap.py`):
  ```python
  dataset_dir = '/path/to/your/DEAP/processed_data/with_base_0.5/'
  task_flag = 'a'   # 'v' for valence, 'a' for arousal
  ```

- For **SEED** (`rma_net_seed.py`):
  ```python
  seed_x_path = '/path/to/your/t6x_89.npy'
  seed_y_path = '/path/to/your/t6y_89.npy'
  ```

### 3. Run training

```bash
python rma_net_deap.py   # train on DEAP (arousal by default)
# or
python rma_net_seed.py   # train on SEED (3-class)
```

The script performs:
- 10‑fold stratified cross‑validation for each subject.
- Early stopping and model checkpointing.
- Evaluation under multiple incomplete‑input scenarios (random missing, block missing, regional occlusion, reduced channel sets).
- Saves all metrics, attention weights, and visualizations to the `visualizations/` folder.

### 4. Results

After completion, you will find:
- **Checkpoints** – best model for each fold (`best_fold_*.pth`)
- **Plots** – accuracy bar, boxplot, loss curves, confusion matrix, ROC curves, class‑wise metrics, temporal/frequency/spatial attention maps, stability summary, and mask examples.
- **Cache** – a pickle file storing all experimental results, allowing you to resume or quickly regenerate plots without retraining.

## Key Hyperparameters

All important parameters are defined at the top of each script and can be easily adjusted:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LR` | 1e‑3 | Learning rate |
| `WEIGHT_DECAY` | 1e‑5 | L2 regularisation |
| `MAX_EPOCHS` | 100 | Maximum training epochs |
| `EARLY_STOP` | 10 | Patience for early stopping |
| `DROPOUT` | 0.5 | Dropout rate in classifier |
| `LAMBDA_REC` | 0.30 | Weight of reconstruction loss |
| `LAMBDA_CONS` | 0.20 | Weight of consistency loss |
| `LAMBDA_REG` | 0.01 | Weight of reliability regularisation |
| `RANDOM_MISSING_MAX` | 0.30 | Max random missing ratio during training |

## Visualisation Examples

The code automatically generates publication‑quality figures (using a consistent scientific colour palette). Some examples:

- **Subject‑wise accuracy bar chart**
- **Model performance boxplot (Accuracy & F1)**
- **Confusion matrix, ROC curves, class‑wise metrics**
- **Temporal, frequency, and spatial attention weight visualisations**
- **Stability under incomplete inputs** (random, block, region, reduced channels)

All outputs are saved in the `visualizations/` subdirectory.

## Citing

If you use this code or the RMA‑Net model in your research, please cite the original thesis/publication:

> [Author(s).] *Thesis Title or Paper Title*. University / Conference, Year.

*(Replace with the actual citation once available.)*

## License

This project is released under the MIT License. See `LICENSE` file for details.

## Contact

For questions or collaboration, please open an issue on the GitHub repository or contact the corresponding author of the associated publication.

**Disclaimer:** This code is provided for research purposes. The authors are not responsible for any misuse or data privacy issues. Please ensure you have the rights to use the DEAP / SEED datasets under their respective licenses.
