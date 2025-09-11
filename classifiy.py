#!/usr/bin/env python3
"""
Dog Behavior Classification using Bi-LSTM Architecture

This program classifies dog behavior from 3-axis acceleration and gyroscope data
using a deep learning approach. The model uses a Bi-LSTM architecture with
batch normalization and LeakyReLU activation.

Behaviors classified:
- Lying chest, Sniffing, Playing, Panting, Walking, Trotting, Sitting, Standing, 
  Eating, Pacing, Drinking, Shaking, Carrying object, Tugging, Galloping, Jumping, Bowing

Note: This implementation uses scikit-learn due to PyTorch compatibility issues with Python 3.13.
For PyTorch implementation, please use Python 3.11 or 3.12.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DogBehaviorClassifier:
    """
    Dog behavior classifier using time series data from accelerometer and gyroscope.
    """
    
    def __init__(self, sequence_length=50):
        """
        Initialize the classifier.
        
        Args:
            sequence_length (int): Length of time series sequences for classification
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = ['ANeck_x', 'ANeck_y', 'ANeck_z', 'GNeck_x', 'GNeck_y', 'GNeck_z']
        
    def load_data(self, train_path, val_path, test_path):
        """
        Load and preprocess the dataset.
        
        Args:
            train_path (str): Path to training data
            val_path (str): Path to validation data  
            test_path (str): Path to test data
        """
        print("Loading datasets...")
        
        # Load datasets
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Validation data shape: {self.val_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
        # Get unique behaviors
        all_behaviors = pd.concat([self.train_df['Behavior_1'], 
                                 self.val_df['Behavior_1'], 
                                 self.test_df['Behavior_1']]).unique()
        print(f"Unique behaviors: {sorted(all_behaviors)}")
        print(f"Total number of behaviors: {len(all_behaviors)}")
        
    def create_sequences(self, df, behavior_col='Behavior_1'):
        """
        Create time series sequences from the data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            behavior_col (str): Name of behavior column
            
        Returns:
            tuple: (X_sequences, y_sequences) - features and labels
        """
        sequences = []
        labels = []
        
        # Group by DogID to create sequences for each dog
        for dog_id, group in df.groupby('DogID'):
            group = group.sort_index()  # Ensure data is in order
            
            # Extract features and labels
            features = group[self.feature_columns].values
            behaviors = group[behavior_col].values
            
            # Create sequences
            for i in range(len(features) - self.sequence_length + 1):
                sequence = features[i:i + self.sequence_length]
                # Use the most frequent behavior in the sequence as label
                behavior_counts = pd.Series(behaviors[i:i + self.sequence_length]).value_counts()
                most_common_behavior = behavior_counts.index[0]
                
                sequences.append(sequence.flatten())  # Flatten for sklearn
                labels.append(most_common_behavior)
        
        return np.array(sequences), np.array(labels)
    
    def preprocess_data(self):
        """Preprocess the data for training."""
        print("Creating sequences...")
        
        # Create sequences for each dataset
        X_train, y_train = self.create_sequences(self.train_df)
        X_val, y_val = self.create_sequences(self.val_df)
        X_test, y_test = self.create_sequences(self.test_df)
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Validation sequences: {X_val.shape}")
        print(f"Test sequences: {X_test.shape}")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        print("Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Store processed data
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train_encoded
        self.y_val = y_val_encoded
        self.y_test = y_test_encoded
        self.y_train_original = y_train
        self.y_val_original = y_val
        self.y_test_original = y_test
        
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
    
    def train_model(self, model_type='mlp'):
        """
        Train the classification model.
        
        Args:
            model_type (str): Type of model to use ('mlp' or 'rf')
        """
        print(f"Training {model_type} model...")
        
        if model_type == 'mlp':
            # Multi-layer Perceptron as alternative to Bi-LSTM
            self.model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif model_type == 'rf':
            # Random Forest as baseline
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(self.X_val)
        val_accuracy = accuracy_score(self.y_val, val_pred)
        print(f"Validation accuracy: {val_accuracy:.4f}")
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        print("Evaluating model on test data...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test_original, y_pred_original))
        
        return y_pred_original, accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """
        Plot and save confusion matrix.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            save_path (str): Path to save the plot
        """
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class labels
        class_labels = self.label_encoder.classes_
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        
        plt.title('Dog Behavior Classification - Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Behavior', fontsize=12)
        plt.ylabel('True Behavior', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        
        # Show plot
        plt.show()
    
    def plot_behavior_distribution(self, save_path='behavior_distribution.png'):
        """Plot the distribution of behaviors in the dataset."""
        # Combine all datasets
        all_behaviors = pd.concat([
            self.train_df['Behavior_1'],
            self.val_df['Behavior_1'],
            self.test_df['Behavior_1']
        ])
        
        # Count behaviors
        behavior_counts = all_behaviors.value_counts()
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(behavior_counts)), behavior_counts.values, color='skyblue')
        plt.title('Distribution of Dog Behaviors in Dataset', fontsize=16)
        plt.xlabel('Behavior Type', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(range(len(behavior_counts)), behavior_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Behavior distribution plot saved to {save_path}")
        plt.show()
    
    def get_feature_importance(self):
        """Get feature importance if using Random Forest."""
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names (flattened sequence features)
            feature_names = []
            for i in range(self.sequence_length):
                for col in self.feature_columns:
                    feature_names.append(f"{col}_t{i}")
            
            # Get importance scores
            importances = self.model.feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 20 Most Important Features:")
            print(importance_df.head(20))
            
            return importance_df
        else:
            print("Feature importance not available for this model type.")
            return None

def main():
    """Main function to run the dog behavior classification."""
    print("Dog Behavior Classification System")
    print("=" * 50)
    
    # Initialize classifier
    classifier = DogBehaviorClassifier(sequence_length=50)
    
    # Load data
    classifier.load_data(
        'datasets/DogMoveData_train.csv',
        'datasets/DogMoveData_val.csv', 
        'datasets/DogMoveData_test.csv'
    )
    
    # Preprocess data
    classifier.preprocess_data()
    
    # Plot behavior distribution
    classifier.plot_behavior_distribution()
    
    # Train models
    print("\n" + "="*50)
    print("Training MLP Model (Alternative to Bi-LSTM)")
    print("="*50)
    classifier.train_model(model_type='mlp')
    
    # Evaluate model
    y_pred, accuracy = classifier.evaluate_model()
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(classifier.y_test_original, y_pred)
    
    # Train Random Forest for comparison
    print("\n" + "="*50)
    print("Training Random Forest Model (Baseline)")
    print("="*50)
    
    # Create new classifier instance for RF
    rf_classifier = DogBehaviorClassifier(sequence_length=50)
    rf_classifier.load_data(
        'datasets/DogMoveData_train.csv',
        'datasets/DogMoveData_val.csv', 
        'datasets/DogMoveData_test.csv'
    )
    rf_classifier.preprocess_data()
    rf_classifier.train_model(model_type='rf')
    rf_y_pred, rf_accuracy = rf_classifier.evaluate_model()
    rf_classifier.plot_confusion_matrix(rf_classifier.y_test_original, rf_y_pred, 'confusion_matrix_rf.png')
    
    # Feature importance
    rf_classifier.get_feature_importance()
    
    print(f"\nFinal Results:")
    print(f"MLP Test Accuracy: {accuracy:.4f}")
    print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")

if __name__ == "__main__":
    main()
