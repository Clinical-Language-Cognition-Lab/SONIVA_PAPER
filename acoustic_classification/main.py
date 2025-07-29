"""
Acoustic Feature Classification for Medical Diagnosis
====================================================

This script implements machine learning models (SVM, Random Forest, Neural Network) 
for classifying post-stroke conditions based on acoustic features from openSMILE.

First Author: [Giulia Sanguedolce]
Paper: [SONIVA: Speech recOgNItion Validation in
Aphasia]
Date: [29/07/2025]

Dependencies:
- pandas>=1.3.0
- numpy>=1.21.0
- torch>=1.9.0
- scikit-learn>=1.0.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
- imbalanced-learn>=0.8.0
- openpyxl>=3.0.0 (for Excel file reading)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.over_sampling import SMOTE
import argparse
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


class DataLoader:
    """Handles data loading and preprocessing."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.validate_data_path()
    
    def validate_data_path(self):
        """Validate that the data file exists."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load_data(self):
        """
        Load data from Excel file.
        
        Returns:
            tuple: (X, y, groups, feature_columns)
        """
        try:
            df = pd.read_excel(self.data_path)
            print(f"Loaded data with shape: {df.shape}")
            
            # Remove filename column if present
            if 'filename' in df.columns:
                df = df.drop(columns=['filename'])
            
            # Validate required columns
            required_cols = ['Label', 'ID']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Extract labels, groups, and features
            y = (df['Label'] == 'Patient').astype(int).to_numpy()
            groups = df['ID']
            feature_cols = [col for col in df.columns if col not in ['Label', 'ID']]
            X = df[feature_cols].to_numpy()
            
            print(f"Features: {len(feature_cols)}")
            print(f"Samples: {len(X)}")
            print(f"Classes distribution: Controls={np.sum(y==0)}, Patients={np.sum(y==1)}")
            print(f"Unique subjects: {len(groups.unique())}")
            
            return X, y, groups, feature_cols
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            sys.exit(1)


class CrossValidator:
    """Handles cross-validation procedures."""
    
    def __init__(self, n_splits=9, random_state=RANDOM_SEED):
        self.n_splits = n_splits
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)
    
    def run_cross_validation(self, X, y, groups, model, scaler=None):
        """
        Run stratified group k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            groups: Group identifiers
            model: ML model to evaluate
            scaler: Feature scaler (optional)
            
        Returns:
            tuple: (fitted_model, cv_reports, cv_accuracies)
        """
        kfold = StratifiedGroupKFold(
            n_splits=self.n_splits, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        cv_reports, cv_accuracies = [], []
        
        print(f"\nRunning {self.n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y, groups=groups)):
            print(f'\nFold {fold + 1}/{self.n_splits}')
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Apply scaling if provided
            if scaler is not None:
                scaler_fold = type(scaler)()
                X_train_fold = scaler_fold.fit_transform(X_train_fold)
                X_val_fold = scaler_fold.transform(X_val_fold)
            
            # Apply SMOTE to balance training data
            X_train_resampled, y_train_resampled = self.smote.fit_resample(
                X_train_fold, y_train_fold
            )
            
            # Train model
            model.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate
            y_val_pred = model.predict(X_val_fold)
            acc = balanced_accuracy_score(y_val_fold, y_val_pred)
            
            report = classification_report(
                y_val_fold, y_val_pred, 
                target_names=['Controls', 'Patient'], 
                output_dict=True
            )
            
            cv_reports.append(report)
            cv_accuracies.append(acc)
            
            print(f"Fold {fold + 1} Balanced Accuracy: {acc:.4f}")
        
        # Print summary
        mean_acc = np.mean(cv_accuracies)
        std_acc = np.std(cv_accuracies)
        print(f"\nCross-validation Results:")
        print(f"Mean Balanced Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        return model, cv_reports, cv_accuracies


class Evaluator:
    """Handles model evaluation and visualization."""
    
    @staticmethod
    def plot_confusion_matrix(cm, title, save_path=None):
        """Plot and optionally save confusion matrix."""
        plt.figure(figsize=(6, 5))
        plt.rcParams.update({'font.size': 12})
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Controls', 'Patients'],
            yticklabels=['Controls', 'Patients']
        )
        
        plt.title(f'{title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name, save_dir=None):
        """Evaluate model on test set."""
        y_pred = model.predict(X_test)
        
        print(f"\n{'='*50}")
        print(f"Test Set Evaluation - {model_name}")
        print(f"{'='*50}")
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=['Controls', 'Patient']
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm, 
            index=['Controls', 'Patients'], 
            columns=['Controls', 'Patients']
        )
        print(f"\nConfusion Matrix ({model_name}):")
        print(cm_df)
        
        # Plot confusion matrix
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
        
        Evaluator.plot_confusion_matrix(cm, model_name, save_path)
        
        # Return metrics for saving
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        return {
            'model': model_name,
            'balanced_accuracy': balanced_acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=['Controls', 'Patient'], 
                output_dict=True
            )
        }


class ImprovedAcousticNet(nn.Module):
    """
    Improved Neural Network for acoustic feature classification.
    
    Architecture:
    - Input layer with batch normalization
    - Two hidden layers with layer/batch normalization
    - Dropout for regularization
    - Sigmoid output for binary classification
    """
    
    def __init__(self, input_size, dropout_rate=0.4):
        super(ImprovedAcousticNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, layer):
        """Initialize layer weights using Kaiming normal initialization."""
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            layer.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output_layer(x)


class RegularizedFocalLoss(nn.Module):
    """
    Focal Loss with L1 regularization for handling class imbalance.
    
    Args:
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        l1_factor: L1 regularization strength
    """
    
    def __init__(self, alpha=2.0, gamma=1.5, l1_factor=0.0005):
        super(RegularizedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l1_factor = l1_factor
    
    def forward(self, inputs, targets, model):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        
        # Calculate class weights
        pos_weight = (targets == 0).float().sum() / targets.float().sum()
        alpha_weight = torch.where(
            targets == 1, 
            pos_weight, 
            torch.tensor([1.0]).to(targets.device)
        )
        
        # Focal loss
        focal_loss = alpha_weight * (1 - pt)**self.gamma * bce_loss
        
        # L1 regularization
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        
        return focal_loss.mean() + self.l1_factor * l1_loss


class NeuralNetworkTrainer:
    """Handles neural network training and evaluation."""
    
    def __init__(self, epochs=120, batch_size=64, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, save_dir=None):
        """Train neural network and evaluate on test set."""
        
        # Apply SMOTE
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_resampled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_resampled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Initialize model, loss, optimizer
        input_size = X_train.shape[1]
        model = ImprovedAcousticNet(input_size).to(self.device)
        criterion = RegularizedFocalLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=15
        )
        
        print(f"\nTraining Neural Network for {self.epochs} epochs...")
        
        # Training loop
        train_losses = []
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1), model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).cpu().numpy()
            y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
        
        # Evaluate and save results
        results = Evaluator.evaluate_model(
            type('MockModel', (), {'predict': lambda self, X: y_pred})(),
            X_test, y_test, "Neural Network", save_dir
        )
        
        # Save model if requested
        if save_dir:
            model_path = os.path.join(save_dir, "neural_network_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'train_losses': train_losses
            }, model_path)
            print(f"Model saved to: {model_path}")
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Acoustic Feature Classification for Medical Diagnosis"
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True,
        help='Path to the Excel data file'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['SVM', 'RF', 'NN', 'all'], 
        default='SVM',
        help='Model to train: SVM, RF, NN, or all'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--cv_folds', 
        type=int, 
        default=9,
        help='Number of cross-validation folds'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    
    # Load data
    data_loader = DataLoader(args.data_path)
    X, y, groups, feature_cols = data_loader.load_data()
    
    # Train-test split using StratifiedGroupKFold
    kfold = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=False)
    train_idx, test_idx = next(kfold.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]
    
    # Print data split information
    print(f"\n{'='*50}")
    print("DATA SPLIT INFORMATION")
    print(f"{'='*50}")
    print(f"Training subjects: {len(groups_train.unique())}")
    print(f"Test subjects: {len(groups_test.unique())}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Feature scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize cross-validator
    cv = CrossValidator(n_splits=args.cv_folds)
    
    # Results storage
    all_results = {}
    
    # Train models
    models_to_train = [args.model] if args.model != 'all' else ['SVM', 'RF', 'NN']
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"TRAINING {model_type}")
        print(f"{'='*60}")
        
        if model_type == 'SVM':
            model = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
            trained_model, cv_reports, cv_accs = cv.run_cross_validation(
                X_train_scaled, y_train, groups_train, model
            )
            results = Evaluator.evaluate_model(
                trained_model, X_test_scaled, y_test, 
                "Support Vector Machine", output_dir
            )
            results['cv_accuracies'] = cv_accs
            all_results['SVM'] = results
            
        elif model_type == 'RF':
            model = RandomForestClassifier(
                n_estimators=200, 
                random_state=RANDOM_SEED
            )
            trained_model, cv_reports, cv_accs = cv.run_cross_validation(
                X_train_scaled, y_train, groups_train, model
            )
            results = Evaluator.evaluate_model(
                trained_model, X_test_scaled, y_test, 
                "Random Forest", output_dir
            )
            results['cv_accuracies'] = cv_accs
            all_results['RF'] = results
            
        elif model_type == 'NN':
            trainer = NeuralNetworkTrainer()
            results = trainer.train_and_evaluate(
                X_train_scaled, y_train, X_test_scaled, y_test, output_dir
            )
            all_results['NN'] = results
    
    # Save all results
    results_file = os.path.join(output_dir, 'results_summary.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save experimental setup
    setup_info = {
        'data_file': args.data_path,
        'models_trained': models_to_train,
        'cv_folds': args.cv_folds,
        'random_seed': RANDOM_SEED,
        'feature_count': len(feature_cols),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_subjects': len(groups_train.unique()),
        'test_subjects': len(groups_test.unique()),
        'timestamp': timestamp
    }
    
    setup_file = os.path.join(output_dir, 'experimental_setup.json')
    with open(setup_file, 'w') as f:
        json.dump(setup_info, f, indent=2)
    
    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {results_file}")
    print(f"Setup file: {setup_file}")


if __name__ == "__main__":
    main()
