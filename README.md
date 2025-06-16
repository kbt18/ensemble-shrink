# NdCNN - CIFAR-10 Classification with N-Dimensional Linear Layers

A PyTorch implementation of a basic Convolutional Neural Network (CNN) for CIFAR-10 image classification that incorporates N-dimensional linear layer (`NdLinear`) instead of the standard fully connected layer for reduced parameter count.

## Requirements

- Python 3.13.4

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv ndcnn_env
source ndcnn_env/bin/activate  # On Windows: ndcnn_env\Scripts\activate
```

### 2. Install Dependencies

Install the requirements:

```bash
pip install -r requirements.txt
```

**Note**: The `ndlinear` package must be available in your Python environment. If it's a custom package, ensure it's properly installed or included in your project directory.

### 3. Clone the NdCNN Repo 

Clone the NdLinear repo into the project root and rename it:

```bash
git clone https://github.com/ensemble-core/NdLinear.git 
mv NdLinear ndlinear
```

## Usage

### Basic Training

Run the script with default parameters:

```bash
python main.py
```

### Model Architecture

The `NdCNN` model consists of:

- **Feature Extraction**: 3 convolutional blocks with BatchNorm, ReLU, and MaxPooling
  - Conv2d(3→32) → BatchNorm → ReLU → MaxPool
  - Conv2d(32→64) → BatchNorm → ReLU → MaxPool  
  - Conv2d(64→128) → BatchNorm → ReLU → MaxPool
- **Global Average Pooling**: Reduces 4×4×128 feature maps to 1×1×128
- **Classification**: `NdLinear` layer (128 → 10 classes)

### Training Configuration

- **Dataset**: CIFAR-10 (50K training + 10K test images)
- **Validation Split**: 5K images from training set
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Patience=10 epochs, min_delta=0.001

## Output

The script provides real-time training progress:

```
Starting training...
Epoch 1: Loss = 1.8234, Val Acc = 0.3240
Epoch 2: Loss = 1.5678, Val Acc = 0.4156
...
Early stopping triggered at epoch 45
Best validation accuracy: 0.7832

Training complete. Evaluating on test data...
Final test accuracy: 0.7756
```

## File Structure

```
project/
├── main.py      # Main training script
├── ndlinear
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── data/               # CIFAR-10 dataset (auto-downloaded)
    └── cifar-10-batches-py/
```

## Key Components

### EarlyStopping Class
- Monitors validation accuracy to prevent overfitting
- Saves best model weights automatically
- Configurable patience and minimum improvement threshold

### NdLinear Integration
- Replaces traditional `nn.Linear` layers
- Maintains compatibility with standard PyTorch workflows

## Customization

### Modify Hyperparameters

Edit the following variables in `main()`:

```python
batch_size = 64          # Batch size for training
learning_rate = 0.001    # Adam optimizer learning rate
patience = 10            # Early stopping patience
```

### Adjust Model Architecture

Modify the `NdCNN` class to experiment with:
- Different kernel sizes
- Additional convolutional layers
- Alternative activation functions
- Dropout layers for regularization