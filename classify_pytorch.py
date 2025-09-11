#!/usr/bin/env python3
"""
Dog Behavior Classification using PyTorch Bi-LSTM Architecture

This program classifies dog behavior from 3-axis acceleration and gyroscope data
using a Bi-LSTM deep learning model with batch normalization and LeakyReLU activation.

Behaviors classified:
- Lying chest, Sniffing, Playing, Panting, Walking, Trotting, Sitting, Standing, 
  Eating, Pacing, Drinking, Shaking, Carrying object, Tugging, Galloping, Jumping, Bowing

Note: This implementation requires PyTorch. For Python 3.13 compatibility issues,
please use Python 3.11 or 3.12, or install PyTorch from source.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Please install PyTorch for Python 3.11/3.12 or build from source.")
    PYTORCH_AVAILABLE = False

# Set random seeds for reproducibility
if PYTORCH_AVAILABLE:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
np.random.seed(42)

class DogBehaviorDataset(Dataset):
    """PyTorch Dataset for dog behavior classification."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class BiLSTMClassifier(nn.Module):
    """
    Bi-LSTM model for dog behavior classification.
    
    Architecture:
    - Bidirectional LSTM layer
    - Batch normalization
    - LeakyReLU activation
    - Global average pooling
    - Dense layer with LeakyReLU
    - Final dense layer for classification
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
        
        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Global average pooling (implemented as adaptive avg pool)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        
        if x.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Transpose for batch normalization: (batch_size, features, sequence_length)
        lstm_out = lstm_out.transpose(1, 2)
        
        # Batch normalization
        lstm_out = self.batch_norm(lstm_out)
        
        # LeakyReLU activation
        lstm_out = self.leaky_relu(lstm_out)
        
        # Global average pooling
        pooled = self.global_avg_pool(lstm_out)  # (batch_size, features, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, features)
        
        # Dense layers
        x = self.fc1(pooled)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class DogBehaviorClassifierPyTorch:
    """
    PyTorch-based dog behavior classifier using Bi-LSTM architecture.
    """
    
    def __init__(self, sequence_length=50, hidden_size=64, num_layers=2, learning_rate=0.001):
        """
        Initialize the classifier.
        
        Args:
            sequence_length (int): Length of time series sequences
            hidden_size (int): Hidden size of LSTM
            num_layers (int): Number of LSTM layers
            learning_rate (float): Learning rate for optimizer
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch.")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_columns = ['ANeck_x', 'ANeck_y', 'ANeck_z', 'GNeck_x', 'GNeck_y', 'GNeck_z']
        
        print(f"Using device: {self.device}")
    
    def load_data(self, train_path, val_path, test_path):
        """Load and preprocess the dataset."""
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
        """Create time series sequences from the data."""
        sequences = []
        labels = []
        
        # Group by DogID to create sequences for each dog
        for dog_id, group in df.groupby('DogID'):
            group = group.sort_index()
            
            # Extract features and labels
            features = group[self.feature_columns].values
            behaviors = group[behavior_col].values
            
            # Create sequences
            for i in range(len(features) - self.sequence_length + 1):
                sequence = features[i:i + self.sequence_length]
                # Use the most frequent behavior in the sequence as label
                behavior_counts = pd.Series(behaviors[i:i + self.sequence_length]).value_counts()
                most_common_behavior = behavior_counts.index[0]
                
                sequences.append(sequence)
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
        
        # Scale features (reshape for scaling, then reshape back)
        print("Scaling features...")
        original_shape = X_train.shape
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(original_shape)
        
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
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
    
    def create_data_loaders(self, batch_size=32):
        """Create PyTorch data loaders."""
        # Create datasets
        train_dataset = DogBehaviorDataset(self.X_train, self.y_train)
        val_dataset = DogBehaviorDataset(self.X_val, self.y_val)
        test_dataset = DogBehaviorDataset(self.X_test, self.y_test)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def train_model(self, num_epochs=50, batch_size=32):
        """Train the Bi-LSTM model."""
        print("Creating data loaders...")
        self.create_data_loaders(batch_size)
        
        # Initialize model
        input_size = len(self.feature_columns)
        num_classes = len(self.label_encoder.classes_)
        
        self.model = BiLSTMClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        print(f"Training Bi-LSTM model for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate average losses and accuracy
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, '
                      f'Val Accuracy: {val_accuracy:.4f}')
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, val_accuracies)
        
        print(f"Training completed. Final validation accuracy: {val_accuracies[-1]:.4f}")
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("Training curves saved to training_curves.png")
        plt.show()
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        print("Evaluating model on test data...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Convert back to original labels
        y_pred_original = self.label_encoder.inverse_transform(all_predictions)
        y_true_original = self.label_encoder.inverse_transform(all_labels)
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_original, y_pred_original))
        
        return y_pred_original, accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix_pytorch.png'):
        """Plot and save confusion matrix."""
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class labels
        class_labels = self.label_encoder.classes_
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        
        plt.title('Dog Behavior Classification - Bi-LSTM Confusion Matrix', fontsize=16)
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

def main():
    """Main function to run the PyTorch dog behavior classification."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch is not available. Please install PyTorch for Python 3.11/3.12.")
        print("Alternatively, use the scikit-learn version in classify.py")
        return
    
    print("Dog Behavior Classification System - PyTorch Bi-LSTM")
    print("=" * 60)
    
    # Initialize classifier
    classifier = DogBehaviorClassifierPyTorch(
        sequence_length=50,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001
    )
    
    # Load data
    classifier.load_data(
        'datasets/DogMoveData_train.csv',
        'datasets/DogMoveData_val.csv', 
        'datasets/DogMoveData_test.csv'
    )
    
    # Preprocess data
    classifier.preprocess_data()
    
    # Train model
    classifier.train_model(num_epochs=50, batch_size=32)
    
    # Evaluate model
    y_pred, accuracy = classifier.evaluate_model()
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(classifier.y_test_original, y_pred)
    
    print(f"\nFinal Results:")
    print(f"Bi-LSTM Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
