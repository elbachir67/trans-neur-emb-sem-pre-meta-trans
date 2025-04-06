# Transformer-Based Neural Embeddings for Semantic Preservation Measurement in Cross-Metamodel Transformations

This repository contains the implementation for our paper "Transformer-Based Neural Embeddings for Semantic Preservation Measurement in Cross-Metamodel Transformations".

## Overview

Our research introduces a novel approach for measuring semantic preservation in model transformations, particularly focusing on cross-metamodel scenarios. We combine neural embeddings with auto-regression to improve the accuracy of semantic preservation assessment.

## Key Features

- **Token Pair Analysis**: Representation of model elements and their metamodel relationships
- **Neural Embeddings**: Use of DistilBERT to capture semantic relationships between model elements
- **Auto-Regression**: Leveraging historical transformation patterns for improved assessment
- **Cross-Metamodel Focus**: Specifically designed to address semantic preservation challenges when transforming between different modeling languages
- **ModelSet Integration**: Support for evaluating transformations using the ModelSet dataset

## Repository Structure

```
.
├── semantic_preservation.py - Main framework implementation
├── token_pair.py - Token pair representation and analysis
├── autoregression.py - Auto-regression model implementation
├── embedding.py - Neural embedding model
├── modelset_loader.py - ModelSet dataset handler
├── rdbms_transformation.py - UML/Ecore to RDBMS transformation
├── test_modelset.py - General testing with ModelSet
├── test_rdbms.py - RDBMS transformation testing
└── results/ - Directory for storing experiment results
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.12+

# Install required packages

pip install torch transformers matplotlib numpy pandas scikit-learn networkx

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/username/semantic-preservation.git
   cd semantic-preservation
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the ModelSet dataset:
   ```bash
   # Download from the official source
   wget https://figshare.com/s/5a6c02fa8ed20782935c -O modelset.zip
   unzip modelset.zip -d modelset-dataset
   ```

## Usage

### Running RDBMS Transformation Experiments

The RDBMS transformation experiments demonstrate semantic preservation measurement for UML/Ecore to Relational Database transformations:

```bash
python test_rdbms.py --model-folder modelset-dataset/txt --max-models 15 --output-dir results
```

### Running General ModelSet Experiments

For broader experimentation with different types of model transformations:

```bash
python test_modelset.py --model-folder modelset-dataset/txt --max-pairs 20 --output-dir results
```

### Command Line Arguments

- `--model-folder`: Path to the ModelSet dataset folder
- `--max-models`: Maximum number of models to select for transformations
- `--embedding-size`: Dimension of neural embeddings (default: 128)
- `--output-dir`: Directory to save results (default: results)

## Results

demonstrated that transformer-based neural embeddings provide a substantial 73.72% improvement in backward assessment score (semantic preservation measurement) and 67.88% improvement in overall quality assessment compared to traditional approaches. For comparison, an auto- regressive approach with historical context provided only a minimal 0.26% improvement, highlighting the exceptional effectiveness of transformer-based models for this task

The framework successfully identifies semantic preservation issues that rule-based approaches miss, particularly for elements without direct metamodel equivalents.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{authorname2025neural,
  title={Transformer-Based Neural Embeddings for Semantic Preservation Measurement in Cross-Metamodel Transformations},
  author={Author, Names},
  journal={Journal Name},
  volume={X},
  number={Y},
  pages={Z},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- We thank the creators of the ModelSet dataset for providing a comprehensive collection of models
