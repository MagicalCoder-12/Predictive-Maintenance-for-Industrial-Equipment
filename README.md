# Predictive Maintenance Analysis

This project implements a machine learning solution for predictive maintenance using Random Forest classification, achieving **99% accuracy**.

## Overview

The project analyzes sensor data from industrial equipment to predict potential failures before they occur. The model processes various features including temperature, rotational speed, torque, and tool wear to classify whether a failure will happen.

## Project Structure

- **`predictive_maintenance.py`**: Refactored Python module containing the main analysis pipeline
- **`main.ipynb`**: Original Jupyter notebook with exploratory analysis
- **`predictive_maintenance.csv`**: Dataset containing sensor readings and failure labels

## Features

The refactored code provides:

- **Object-Oriented Design**: Encapsulated in `PredictiveMaintenanceAnalyzer` class for better maintainability
- **Data Preprocessing**: Automatic handling of categorical encoding and feature scaling
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) integration
- **Feature Importance Analysis**: Visualization of feature contributions using Random Forest
- **Model Evaluation**: Comprehensive metrics including classification report, confusion matrix, and cross-validation
- **Reusable Components**: Modular methods that can be used independently

## Usage

### Running the Complete Pipeline

```bash
python predictive_maintenance.py
```

### Using as a Module

```python
from predictive_maintenance import PredictiveMaintenanceAnalyzer

# Initialize analyzer
analyzer = PredictiveMaintenanceAnalyzer('predictive_maintenance.csv')

# Preprocess data
analyzer.preprocess_data()

# Handle class imbalance
X_resampled, y_resampled = analyzer.apply_smote()

# Split data
X_train, X_test, y_train, y_test = analyzer.split_data()

# Train and evaluate
analyzer.train_and_evaluate(X_train, y_train, X_test, y_test)

# Cross-validation
analyzer.perform_cross_validation(X_train, y_train, X_test, y_test)
```

## Model Performance

- **Test Accuracy**: 99.85%
- **Cross-Validation Mean Accuracy**: 99.91%
- **Precision (Failure Class)**: 99%
- **Recall (Failure Class)**: 97%

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

## Key Improvements from Original Notebook

1. **Better Code Organization**: Converted from notebook to modular Python script
2. **Type Hints**: Added type annotations for better IDE support and documentation
3. **Docstrings**: Comprehensive documentation for all classes and methods
4. **Error Handling**: Added validation checks (e.g., model must be trained before cross-validation)
5. **Reusability**: Class-based design allows for easy extension and testing
6. **Separation of Concerns**: Each method has a single, well-defined responsibility

## License

This project is for educational purposes.
