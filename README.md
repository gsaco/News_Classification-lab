# News Classification Lab

Fine-tuning Transformer Models for AG News Classification and Evaluation

## 📋 Project Overview

This project implements and compares three state-of-the-art transformer models (RoBERTa, DeBERTa, and ModernBERT) for news classification using the AG News dataset. The project also includes a bonus task that applies these models to classify Peruvian news articles from RPP.

### AG News Categories:
- 0: World
- 1: Sports
- 2: Business
- 3: Science/Technology

## 🏗️ Repository Structure

```
News_Classification-lab/
├── data/                      # Data files
│   ├── rpp_news_50.csv       # RPP news articles for bonus task
│   ├── rpp_classified.json   # LLM classifications (bonus)
│   └── .gitkeep
├── notebooks/                 # Jupyter notebooks
│   └── agnews_train_eval.ipynb  # Main training and evaluation notebook
├── outputs/                   # Results and visualizations
│   ├── model_f1_comparison.png
│   ├── confusion_matrices.png
│   ├── rpp_category_distribution.png
│   ├── model_vs_llm_comparison.png (if LLM task completed)
│   └── .gitkeep
├── src/                       # Source code (optional scripts)
│   └── .gitkeep
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- GPU recommended for faster training (works with CPU but slower)
- 8GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gsaco/News_Classification-lab.git
cd News_Classification-lab
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebook

#### Local Jupyter:
```bash
jupyter notebook notebooks/agnews_train_eval.ipynb
```

#### Google Colab:
1. Upload the notebook to Google Colab
2. Mount your Google Drive or upload the data files
3. Update file paths to match Colab structure
4. Run all cells

**Note**: Training may take 1-3 hours depending on hardware. The notebook saves checkpoints so you can resume if interrupted.

## 📊 Methodology

### Data Preparation
- **Dataset**: AG News (127,600 samples)
- **Split**: 70% train / 15% validation / 15% test
- **Validation Strategy**: Used only train + validation for model tuning; test set reserved for final evaluation

### Models Trained
1. **RoBERTa-base** (roberta-base)
   - Optimized BERT with dynamic masking
   - 125M parameters
   
2. **DeBERTa-v3-base** (microsoft/deberta-v3-base)
   - Disentangled attention mechanism
   - Enhanced mask decoder
   - 183M parameters
   
3. **ModernBERT-base** (answerdotai/ModernBERT-base)
   - Recent architecture with modern optimizations
   - Improved efficiency and performance

### Training Configuration
- **Max Sequence Length**: 256 tokens
- **Batch Size**: 16 (train) / 32 (eval)
- **Epochs**: 3 with early stopping
- **Optimizer**: AdamW with warmup
- **Learning Rate**: Default (5e-5)
- **Metric**: F1-score (macro)

### Evaluation Metrics
- F1-score (macro): Equal weight to all classes
- F1-score (weighted): Weighted by class support
- Confusion matrices
- Per-class precision, recall, F1

## 📈 Results

The notebook generates several visualizations:
- **F1-score comparison chart**: Bar plot comparing all three models
- **Confusion matrices**: One for each model
- **Classification reports**: Detailed per-class metrics

Results are saved to:
- `outputs/model_comparison.csv`: Numerical results
- `outputs/model_f1_comparison.png`: Visual comparison
- `outputs/confusion_matrices.png`: All confusion matrices

## 🎁 Bonus Task: RPP News Classification

The bonus task applies the trained models to 50 real-world Peruvian news articles from RPP.

### Workflow:
1. **LLM Classification** (optional): Use GPT or another LLM to classify RPP articles as ground truth
2. **Model Predictions**: Apply all three trained models to RPP articles
3. **Comparison**: Analyze agreement between models and LLM (if available)
4. **Analysis**: Discuss domain shift and model behavior

### Key Considerations:
- **Language**: RPP articles are in Spanish; AG News is English
- **Domain Shift**: Peruvian news vs. international news
- **Cultural Context**: Local references may affect classification
- **Cross-lingual Transfer**: Models need multilingual capabilities

## 📝 Key Findings

### Model Performance on AG News:
*(Results will vary based on training - check notebook outputs)*

- All models achieve high F1-scores (typically >0.90)
- ModernBERT often shows slight improvements due to recent optimizations
- DeBERTa's disentangled attention helps with context understanding

### RPP News Analysis:
- Model agreement varies by article clarity
- Sports articles tend to have highest agreement
- World/Business news may overlap due to international economics
- Domain adaptation recommended for production Spanish news classification

## 🔧 Customization

### Training Your Own Model:
```python
model_results = train_and_evaluate_model(
    model_name='your-model-name',
    dataset_splits=splits,
    output_dir='../outputs/your-model',
    num_epochs=3,
    batch_size=16
)
```

### Using Different Datasets:
Replace the AG News loading section with your own dataset:
```python
dataset = load_dataset("your-dataset")
# Ensure it has 'text' and 'label' columns
```

## 📚 Dependencies

Core packages (see `requirements.txt` for versions):
- `transformers`: Hugging Face Transformers
- `datasets`: Hugging Face Datasets
- `torch`: PyTorch
- `scikit-learn`: Metrics and evaluation
- `matplotlib`, `seaborn`: Visualization
- `pandas`, `numpy`: Data manipulation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Hyperparameter optimization
- Additional model architectures
- Cross-lingual evaluation
- Ensemble methods
- Real-time inference API

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- **AG News Dataset**: Zhang et al., 2015
- **Hugging Face**: For transformers library and model hub
- **RPP**: For news articles used in bonus task

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project for demonstrating transformer-based text classification. Model performance may vary based on hardware and random initialization.
