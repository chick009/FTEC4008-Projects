# FTEC_4008 – Financial Time Series Classification with Sentiment Embeddings and SAE

## AI Disclaimer
‘I acknowledge the use of ChatGPT & Grok to generate the relevant README with my supervision‘

## Project Overview
This repository implements an end-to-end pipeline for:

- Merging financial OHLCV time series data with sentiment embeddings (FinBERT).
- Preprocessing and dataset preparation.
- Training transformer-based classification models (`PatchTSMixer`) under various configurations (with/without sentiment, masking).
- Extracting hidden representations and analyzing them via a Sparse Autoencoder (SAE).
- Supervised classification on SAE embeddings.

A detailed project report is available in `FTEC_4008_Projects_v1.pdf`.

## Directory Structure
```
.
├── FTEC_4008_Dataset_Preparations.ipynb    # Merge and preprocess data
├── FTEC_4008_Classification_Training.ipynb  # Transformer classifier evaluation & SAE analysis
├── FTEC4008_SAE_Trainings.ipynb             # Combined pipeline: dataset prep, hidden-state extraction, SAE training
├── ohlcv.csv                                # Raw OHLCV time series data
├── is_easy_1_5_20_v2.csv                    # Difficulty annotations for 1, 5, 20-day horizons
├── is_easy_5_20_v2.csv                      # Difficulty annotations for 5, 20-day horizons
├── merged_dataset.csv                       # (Optional) Pre-computed merged dataset
├── Execution_Proof/                         # Example outputs: plots, classification reports
└── README.md                                # Project overview and instructions
```

## Requirements
- Python 3.8+
- Jupyter Notebook or JupyterLab
- pandas
- numpy
- torch (PyTorch)
- transformers
- scikit-learn
- matplotlib
- seaborn
- tqdm
- collections
- chronos-forecasting
- transformers==4.40.1

Install dependencies via pip:
```bash
pip install pandas numpy torch transformers scikit-learn matplotlib seaborn tqdm
```

> **Tip:** You can also create a `requirements.txt` with the above packages and run:
> ```bash
> pip install -r requirements.txt
> ```

## Usage

### 1. Google Colab (Recommended)
1. Open the notebook in Colab:
   - Go to `File → Open notebook` → `GitHub` and paste the repository URL, or upload the `.ipynb` files.
2. Mount your Google Drive to access large data files:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ensure data paths in the notebooks point to `/content/drive/MyDrive/financial_data/`.
4. Run the notebooks in the following order:
   1. **Data Preparation**: `FTEC_4008_Dataset_Preparations.ipynb`
   2. **Classification Training & Analysis**: `FTEC_4008_Classification_Training.ipynb`
   3. **SAE Training & Activation Analysis**: `FTEC4008_SAE_Trainings.ipynb`

### 2. Local Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/chick009/FTEC4008-Projects
   ```
2. Install dependencies as shown above.
3. Start Jupyter Lab/Notebook:
   ```bash
   jupyter notebook
   ```
4. Execute the notebooks in the same order:
   1. `FTEC_4008_Dataset_Preparations.ipynb`
   2. `FTEC_4008_Classification_Training.ipynb`
   3. `FTEC4008_SAE_Trainings.ipynb`

## Data Files
- **`ohlcv.csv`**: Contains daily open, high, low, close, volume data for selected stocks.
- **`is_easy_*.csv`**: Labels indicating easy/hard forecast cases based on price movement thresholds.
- **`finbert_emb_v2.csv`**: Mapping of dates and stocks to precomputed FinBERT embeddings (`.pt` files). You can access the precomputed embeddings via https://drive.google.com/drive/folders/1ptIf0xGgmRqxDpSb0kLXo_9ssdaZuCuY?usp=sharing 

> The notebooks will load these files, merge them into a unified DataFrame, and save intermediate outputs (e.g., `merged_dataset.csv`) to speed up subsequent runs.

## Outputs & Results
- **Classification Reports**: Precision, recall, F1-score for each condition (with/without sentiment, masking).
- **SAE Analysis**: Clustering and downstream classification metrics (random forest).
- All plots and reports are saved under `Execution_Proof/` or as specified in each notebook.

---
*For any questions or issues, please raise an issue in the repository or contact the project maintainer.*