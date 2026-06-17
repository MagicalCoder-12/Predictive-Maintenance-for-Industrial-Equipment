"""
Predictive Maintenance Analysis Module

This module provides functionality for analyzing predictive maintenance data,
including data preprocessing, feature engineering, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


class PredictiveMaintenanceAnalyzer:
    """
    A class to handle predictive maintenance data analysis and modeling.
    
    Attributes:
        df (pd.DataFrame): The main dataframe containing the dataset.
        label_encoders (dict): Dictionary to store label encoders for categorical columns.
        scaler (StandardScaler): Scaler for numerical features.
        model (RandomForestClassifier): Trained random forest model.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the analyzer with data from a CSV file.
        
        Args:
            filepath: Path to the CSV file containing maintenance data.
        """
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.load_data(filepath)
    
    def load_data(self, filepath: str) -> None:
        """Load data from CSV file."""
        self.df = pd.read_csv(filepath)
    
    def explore_data(self) -> None:
        """
        Perform initial data exploration.
        
        Displays dataframe info, null values, and basic statistics.
        """
        print("Dataset Info:")
        print(self.df.info())
        print("\nNull Values:")
        print(self.df.isnull().sum())
        print("\nBasic Statistics:")
        print(self.df.describe())
    
    def preprocess_data(self) -> None:
        """
        Preprocess the data by removing unnecessary columns,
        encoding categorical variables, and scaling numerical features.
        """
        # Drop unnecessary columns
        if 'UDI' in self.df.columns:
            self.df.drop(columns=['UDI', 'Product ID'], inplace=True)
        
        # Encode categorical variables
        categorical_cols = ['Type', 'Failure Type']
        for col in categorical_cols:
            if col in self.df.columns:
                encoder = LabelEncoder()
                self.df[col] = encoder.fit_transform(self.df[col])
                self.label_encoders[col] = encoder
        
        # Scale numerical features
        numerical_cols = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
    
    def analyze_target_distribution(self) -> None:
        """Analyze and display the target variable distribution."""
        print("Target Distribution:")
        print(self.df['Target'].value_counts(normalize=True))
    
    def apply_smote(self, sampling_strategy: float = 0.3, random_state: int = 42):
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            sampling_strategy: Ratio of minority class to majority class after resampling.
            random_state: Random seed for reproducibility.
            
        Returns:
            tuple: Resampled features (X) and target (y).
        """
        X = self.df.drop(columns=['Target'])
        y = self.df['Target']
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print("New Target Distribution after SMOTE:")
        print(pd.Series(y_resampled).value_counts(normalize=True))
        
        return X_resampled, y_resampled
    
    def plot_feature_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_estimators: int = 100,
        random_state: int = 42
    ) -> None:
        """
        Train a Random Forest and plot feature importance.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            n_estimators: Number of trees in the forest.
            random_state: Random seed for reproducibility.
        """
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf_model.fit(X, y)
        
        feature_importance = rf_model.feature_importances_
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=X.columns, y=feature_importance)
        plt.xticks(rotation=45)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.title("Feature Importance from Random Forest")
        plt.tight_layout()
        plt.show()
    
    def remove_low_importance_features(self, low_importance_cols: list) -> None:
        """
        Remove features with low importance.
        
        Args:
            low_importance_cols: List of column names to remove.
        """
        self.df.drop(columns=low_importance_cols, inplace=True)
        print("Updated Dataset Statistics:")
        print(self.df.describe())
    
    def split_data(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42,
        stratify: bool = True
    ):
        """
        Split data into training and testing sets.
        
        Args:
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.
            stratify: Whether to stratify the split.
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = self.df.drop(columns=['Target'])
        y = self.df['Target']
        
        stratify_param = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Testing set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> RandomForestClassifier:
        """
        Train a Random Forest model and evaluate its performance.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Testing features.
            y_test: Testing target.
            n_estimators: Number of trees in the forest.
            random_state: Random seed for reproducibility.
            
        Returns:
            RandomForestClassifier: Trained model.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return self.model
    
    def perform_cross_validation(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        cv: int = 5
    ) -> None:
        """
        Perform cross-validation on the trained model.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Testing features (optional).
            y_test: Testing target (optional).
            cv: Number of cross-validation folds.
        """
        if self.model is None:
            raise ValueError("Model must be trained before cross-validation.")
        
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=cv, 
            scoring="accuracy"
        )
        
        print(f"Cross-validation accuracy scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean()}")
        
        if X_test is not None and y_test is not None:
            print(f"Test set accuracy: {self.model.score(X_test, y_test)}")


def main():
    """Main function to run the predictive maintenance analysis pipeline."""
    # Initialize analyzer
    analyzer = PredictiveMaintenanceAnalyzer('predictive_maintenance.csv')
    
    # Explore data
    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    analyzer.explore_data()
    
    # Preprocess data
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    analyzer.preprocess_data()
    analyzer.analyze_target_distribution()
    
    # Apply SMOTE for class imbalance
    print("\n" + "=" * 60)
    print("HANDLING CLASS IMBALANCE (SMOTE)")
    print("=" * 60)
    X_resampled, y_resampled = analyzer.apply_smote()
    
    # Analyze feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    analyzer.plot_feature_importance(X_resampled, y_resampled)
    
    # Remove low importance features
    print("\n" + "=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)
    analyzer.remove_low_importance_features(['Type', 'Process temperature [K]'])
    
    # Split data
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = analyzer.split_data()
    
    # Train and evaluate model
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    analyzer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Cross-validation
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    analyzer.perform_cross_validation(X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
