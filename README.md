# Haimian: Intraday Stock Prediction


**Haimian** is a flexible and modular framework for intraday stock price prediction, built on PyTorch Lightning. It supports **multi-task learning**, **time-series forecasting**, **DeepFM-style recommendation models**, and **tabular deep learning**. The project is designed for scalability, enabling easy customization of models, datasets, and training pipelines.

## ✨ Features

- **Multi-Task Learning**: Predict multiple targets (e.g., price regression, trend classification) simultaneously.
- **Time-Series Support**: Handle sequential stock data with configurable window lengths.
- **DeepFM Integration**: Incorporate DeepFM-style models for recommendation-like feature interactions.
- **Tabular Deep Learning**: Leverage deep learning for structured tabular data.
- **PyTorch Lightning**: Standardized training with minimal boilerplate code.
- **OmegaConf**: Structured and flexible configuration management.
- **Modular Design**: Easily extend models, datasets, and pipelines.

## 📂 Project Structure

```plaintext
haimian/
│
├── configs/                  # Configuration files
│   └── config.yaml           # Hyperparameter configuration (YAML)
│
├── data/                     # Data handling
│   ├── __init__.py
│   └── data_module.py        # LightningDataModule for stock data
│
├── models/                   # Model definitions
│   ├── __init__.py
│   ├── base_model.py         # Abstract base LightningModule
│   └── specific_model.py      # Specific model implementations (e.g., DeepFM, LSTM)
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── logging.py            # Logging utilities
│   └── helpers.py            # Helper functions
│
├── train.py                  # Main training script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch Lightning 2.0+
- OmegaConf for configuration management
- Pandas, NumPy, and other dependencies (see `requirements.txt`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/haimian.git
   cd haimian
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Ensure your stock data is in a Pandas DataFrame format with columns for features (continuous and categorical), targets, and category labels.
   - Example data format:
     ```python
     import pandas as pd
     data = pd.DataFrame({
         "time": [1, 2, 3, 4, 5, 6],
         "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
         "volume": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
         "trend": [0, 1, 0, 1, 0, 1],
         "sector": [1, 2, 1, 2, 1, 2],
         "category": [7, 7, 7, 6, 7, 7]
     })
     ```

### Usage

1. **Configure the model**:
   - Edit `configs/config.yaml` to specify dataset and model parameters:
     ```yaml
     data:
       continuous_cols: ["price", "volume"]
       categorical_cols: ["sector"]
       target_cols: ["trend"]
       task_types:
         trend: "classification"
       window_len: 2
       batch_size: 32
       split_type: "time"0
     model:
       hidden_dim: 64
       num_classes: 2
     trainer:
       max_epochs: 10
     ```

2. **Run the training script**:
   ```bash
   python train.py --config configs/config.yaml
   ```

3. **Customize models**:
   - Define new models in `models/specific_model.py` by inheriting from `BaseModel`.
   - Follow the `embedding + backbone + head` structure, with heads supporting multi-task outputs (e.g., dictionary of task-specific predictions).

### Example

```python
# train.py
from omegaconf import OmegaConf
import pytorch_lightning as pl
from data.data_module import StockDataModule
from models.specific_model import StockPredictor

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Initialize data module
data_module = StockDataModule(
    train=data,
    config=config.data,
    validation=None,
    seed=42,
)

# Initialize model
model = StockPredictor(
    continuous_dim=len(config.data.continuous_cols),
    categorical_dim=len(config.data.categorical_cols) if config.data.categorical_cols else 0,
    window_len=config.data.window_len,
    hidden_dim=config.model.hidden_dim,
    num_classes=config.model.num_classes,
)

# Train
trainer = pl.Trainer(**config.trainer)
trainer.fit(model, data_module)
```

## 🛠️ Code Details

### Core Components

- **HaimianModel**: The top-level class orchestrating data loading, model initialization, and training logic. It integrates `StockDataModule` and specific models.
- **BaseModel**: An abstract `pl.LightningModule` providing standardized training, validation, and logging methods. All models inherit from it.
- **Specific Models**: Concrete models (e.g., LSTM, DeepFM) defined in `models/specific_model.py`. Follow the `embedding + backbone + head` paradigm, with heads outputting a dictionary for multi-task learning.

### Data Handling

- **`StockDataModule`**: A `pl.LightningDataModule` for loading and preprocessing stock data. Supports:
  - Time-series windows for sequential data.
  - Multi-task targets (regression/classification).
  - Random or time-based train/validation splits.
  - Configurable via `DataConfig` and OmegaConf.

### Model Architecture

- **Embedding Layer**: Handles categorical features (e.g., sector) with embeddings.
- **Backbone**: Processes sequential/tabular data (e.g., LSTM for time-series, DeepFM for feature interactions).
- **Head**: Task-specific output layers, returning a dictionary of predictions (e.g., `{"trend": logits}`).

### Configuration

- Parameters are managed using **OmegaConf** with YAML files.
- `DataConfig` defines data  `StockDataModule` parameters, supporting direct initialization or file-based loading.

## 📚 Documentation

- **DataConfig**: See `data/data_module.py` for configuration details.
- **Model Customization**: Extend `BaseModel` in `models/base_model.py` for new models.
- **Training**: Use `train.py` to run experiments.



## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

## 🙌 Acknowledgments

- [PyTorch Lightning](https://www.pytorchlightning.ai/) for simplifying training.
- [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management.
- The open-source community for inspiration and tools.

---
