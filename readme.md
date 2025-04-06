# Transformer-Based Neural Embeddings for Semantic Preservation Measurement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.12+-green.svg)](https://huggingface.co/transformers/)

Official implementation of our research paper: [Transformer-Based Neural Embeddings for Semantic Preservation Measurement in Cross-Metamodel Transformations](https://arxiv.org/abs/xxxx.xxxxx)

## 📑 Abstract

Model transformation validation approaches typically focus on structural correctness while neglecting semantic preservation measurement, especially in cross-metamodel transformations. Our research introduces a novel approach using transformer-based neural embeddings to quantify semantic preservation across different modeling languages. Experiments on 134 diverse transformations demonstrate our approach delivers a statistically significant **73.72% improvement** in backward assessment score.

![Framework Overview](docs/images/framework-overview.png)

## 🚀 Key Features

- **Transformer-Based Neural Embeddings**: Leverages DistilBERT to capture deep semantic relationships between model elements
- **Token Pair Representation**: Novel approach for representing model elements and their metamodel relationships
- **Bidirectional Assessment Framework**: Evaluates both structural correctness and semantic preservation
- **Cross-Metamodel Focus**: Specifically addresses semantic preservation challenges when transforming between different modeling languages
- **Extensive Evaluation**: Tested on 134 diverse transformations from the ModelSet benchmark

## 📊 Key Results

| Approach          | Quality Score | Improvement |
| ----------------- | ------------- | ----------- |
| Baseline          | 0.1480        | -           |
| Neural Embeddings | 0.2485        | +67.88%     |
| Auto-regression   | 0.1484        | +0.26%      |

Our framework successfully identifies semantic preservation issues that rule-based approaches miss, particularly for elements without direct metamodel equivalents.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.12+

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/username/semantic-preservation.git
cd semantic-preservation

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the ModelSet dataset
wget https://figshare.com/s/5a6c02fa8ed20782935c -O modelset.zip
unzip modelset.zip -d modelset-dataset
```

## 📁 Repository Structure

```
.
├── src/
│   ├── semantic_preservation.py      # Main framework implementation
│   ├── token_pair.py                 # Token pair representation and analysis
│   ├── autoregression.py             # Auto-regression model implementation
│   ├── embedding.py                  # Transformer-based neural embedding model
│   ├── modelset_loader.py            # ModelSet dataset handler
│   └── rdbms_transformation.py       # UML/Ecore to RDBMS transformation
├── tests/
│   ├── test_modelset.py              # General testing with ModelSet
│   └── test_rdbms.py                 # RDBMS transformation testing
├── data/
│   └── sample_models/                # Sample models for quick testing
├── results/                          # Directory for storing experiment results
├── docs/
│   └── images/                       # Documentation images
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🧪 Usage

### Running RDBMS Transformation Experiments

```bash
python tests/test_rdbms.py --model-folder modelset-dataset/txt --max-models 15 --output-dir results
```

### Running General ModelSet Experiments

```bash
python tests/test_modelset.py --model-folder modelset-dataset/txt --max-pairs 20 --output-dir results
```

### Advanced Configuration

```bash
python tests/test_rdbms.py \
  --model-folder modelset-dataset/txt \
  --max-models 50 \
  --embedding-size 768 \
  --embedding-model "distilbert-base-uncased" \
  --batch-size 16 \
  --epochs 10 \
  --learning-rate 2e-5 \
  --beta 0.7 \
  --alpha 0.5 \
  --output-dir results/rdbms_large_scale
```

### API Usage

```python
from src.semantic_preservation import SemanticPreservationMeasurement
from src.embedding import TransformerEmbedding
from src.token_pair import TokenPairExtractor

# Initialize components
embedding_model = TransformerEmbedding("distilbert-base-uncased")
token_extractor = TokenPairExtractor()
measurement = SemanticPreservationMeasurement(embedding_model, token_extractor)

# Measure semantic preservation
source_model = load_model("path/to/source.uml")
target_model = load_model("path/to/target.rdbms")
quality_score, forward_score, backward_score = measurement.assess(source_model, target_model)

print(f"Quality Score: {quality_score:.4f}")
print(f"Forward Assessment Score: {forward_score:.4f}")
print(f"Backward Assessment Score: {backward_score:.4f}")
```

## 🔍 Example: UML to Relational Transformation

Our approach successfully identifies semantic preservation issues in UML to Relational transformations:

**UML Class Model:**

```
Class: Person
Attributes:
  - id: Integer {primaryKey}
  - name: String
  - dateOfBirth: Date
Operations:
  - calculateAge(): Integer {derivedFrom=dateOfBirth}
```

**Relational Schema:**

```
Table: PERSON
Columns:
  - ID: INTEGER (Primary Key)
  - NAME: VARCHAR
  - DATE_OF_BIRTH: DATE
```

**Token Pair Similarity Matrix:**
| UML Element | RDBMS Element | Structural Similarity | Embedding Similarity |
|-------------|---------------|------------------------|----------------------|
| Person (Class) | PERSON (Table) | 0.83 | 0.92 |
| id (Attribute) | ID (Column) | 0.87 | 0.95 |
| calculateAge (Operation) | - | 0.22 (best match) | 0.38 (to VIEW_AGE) |

Our transformer-based embeddings detect the semantic relationship between `calculateAge` and potential implementations like views, even when not explicitly present.

## 📈 Visualization Tools

The repository includes several visualization tools:

```bash
# Generate token pair similarity heatmap
python scripts/visualize_similarity.py --input results/rdbms_experiment --output visualizations/heatmap.png

# Generate quality score comparison chart
python scripts/visualize_results.py --input results/comparison_experiment --output visualizations/comparison.png
```

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{author2025transformer,
  title={Transformer-Based Neural Embeddings for Semantic Preservation Measurement in Cross-Metamodel Transformations},
  author={Author, Names},
  journal={Journal Name},
  volume={X},
  number={Y},
  pages={Z},
  year={2025},
  publisher={Publisher Name}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- We thank the creators of the ModelSet dataset for providing a comprehensive collection of models
- This research was supported by [Your Institution/Grant]
- Computational resources were provided by [Computing Center]
