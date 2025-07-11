#!/usr/bin/env python3
"""
Binary Defense Detection - Convolutional Neural Network (CNN)
Goal: Detect whether defense mechanisms are active in the network using deep learning
Complementary approach to Random Forest for comparison and ensemble possibilities
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class CNNDefenseDetector:
    def __init__(self, data_path="./simulations/features/"):
        """
        CNN-based binary defense detector: with defense (1) or without defense (0)
        """
        self.data_path = data_path
        self.cnn_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.history = None
        self.input_shape = None

    def load_simulation_data(self):
        """
        Load data and group scenarios with defense into one category
        Same as Random Forest approach for consistency
        """
        all_data = []

        # Binary mapping: defense present or not
        scenarios = {
            'baseline': 0,  # No defense
            'defense': 1,  # Defense active
            'defense_vs_attack': 1  # Defense active (attack doesn't matter for this binary task)
        }

        for scenario, binary_label in scenarios.items():
            scenario_path = os.path.join(self.data_path, scenario)
            csv_files = glob.glob(os.path.join(scenario_path, "*.csv"))

            print(f"Loading {len(csv_files)} files from {scenario} scenario -> Label: {binary_label}")

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['defense_active'] = binary_label
                    df['original_scenario'] = scenario
                    df['file_source'] = os.path.basename(csv_file)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")

        if not all_data:
            raise ValueError("No data files found! Check your data path.")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Show distribution
        defense_counts = combined_df['defense_active'].value_counts()
        print(f"\nDataset distribution:")
        print(f"No Defense (0): {defense_counts[0]} samples")
        print(f"Defense Active (1): {defense_counts[1]} samples")
        print(f"Total records loaded: {len(combined_df)}")

        return combined_df

    def preprocess_data_for_cnn(self, df):
        """
        Preprocess the simulation data specifically for CNN input
        Creates structured input that CNNs can process effectively
        """
        # Pivot the data to have metrics as columns
        df_pivot = df.pivot_table(
            index=['defense_active', 'original_scenario', 'file_source'],
            columns='Metric',
            values='Value',
            aggfunc='first'
        ).reset_index()

        # Clean column names
        df_pivot.columns.name = None

        # Separate features and binary labels
        feature_cols = [col for col in df_pivot.columns
                        if col not in ['defense_active', 'original_scenario', 'file_source']]

        X = df_pivot[feature_cols]
        y = df_pivot['defense_active'].values

        # Handle missing values
        X = X.fillna(X.median())

        # Store feature names
        self.feature_names = feature_cols

        print(f"Features available for CNN: {len(feature_cols)}")
        print(f"Binary classification samples: {Counter(y)}")

        return X.values, y, df_pivot

    def create_cnn_architecture(self, input_shape, dropout_rate=0.3):
        """
        Create CNN architecture optimized for network metrics analysis
        """
        model = keras.Sequential([
            # Input layer - reshape 1D feature vector to 2D for CNN processing
            layers.Input(shape=input_shape),
            layers.Reshape((input_shape[0], 1)),  # Convert to (features, 1) for 1D CNN
            
            # First CNN block - detect local patterns in metrics
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout_rate),
            
            # Second CNN block - detect higher-level patterns
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout_rate),
            
            # Third CNN block - capture complex relationships
            layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),  # Global pooling to reduce dimensionality
            
            # Dense layers for final classification
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            
            # Output layer for binary classification
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def train_cnn_model(self, X, y, test_size=0.2, validation_size=0.2, random_state=42):
        """
        Train CNN for binary defense detection with comprehensive evaluation
        """
        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Store input shape
        self.input_shape = (X_train_scaled.shape[1],)

        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Validation set: {X_val_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        print(f"Input shape: {self.input_shape}")

        # Create model
        self.cnn_model = self.create_cnn_architecture(self.input_shape)

        # Compile model with class weights to handle imbalance
        class_weight = {
            0: len(y_train) / (2 * np.sum(y_train == 0)),
            1: len(y_train) / (2 * np.sum(y_train == 1))
        }

        self.cnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        # Print model architecture
        print("\nCNN Model Architecture:")
        self.cnn_model.summary()

        # Callbacks for training
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            'best_cnn_defense_model.h5', monitor='val_loss', save_best_only=True
        )

        # Train model
        print("\nTraining CNN model...")
        self.history = self.cnn_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )

        # Load best model
        self.cnn_model = keras.models.load_model('best_cnn_defense_model.h5')

        # Evaluate on test set
        test_loss, test_accuracy, test_precision, test_recall = self.cnn_model.evaluate(
            X_test_scaled, y_test, verbose=0
        )

        # Get predictions and probabilities
        y_pred_proba = self.cnn_model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate AUC
        auc_score = roc_auc_score(y_test, y_pred_proba)

        print(f"\nTest Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test AUC: {auc_score:.4f}")

        return {
            'model': self.cnn_model,
            'history': self.history,
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'auc_score': auc_score,
            'class_weight': class_weight
        }

    def evaluate_cnn_model(self, results):
        """
        Comprehensive evaluation for CNN defense detection
        """
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']

        print("\n" + "=" * 60)
        print("CNN BINARY DEFENSE DETECTION - EVALUATION RESULTS")
        print("=" * 60)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['No Defense', 'Defense Active']))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("\nConfusion Matrix Interpretation:")
        print(f"True Negatives (correctly identified no defense): {cm[0, 0]}")
        print(f"False Positives (incorrectly detected defense): {cm[0, 1]}")
        print(f"False Negatives (missed defense): {cm[1, 0]}")
        print(f"True Positives (correctly detected defense): {cm[1, 1]}")

        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nKey Metrics:")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"AUC Score: {results['auc_score']:.4f}")
        print(f"Precision: {precision:.4f} (of all defense predictions, how many were correct)")
        print(f"Recall: {recall:.4f} (of all actual defenses, how many were detected)")
        print(f"F1-Score: {f1:.4f} (harmonic mean of precision and recall)")

        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def plot_training_history(self):
        """
        Plot CNN training history
        """
        if self.history is None:
            print("Model not trained yet!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, results):
        """
        Plot ROC curve for CNN binary classification
        """
        y_test = results['y_test']
        y_pred_proba = results['y_pred_proba']

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = results['auc_score']

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'CNN ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - CNN Defense Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"At optimal threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")

        return optimal_threshold

    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix heatmap for CNN binary classification
        """
        plt.figure(figsize=(8, 6))
        labels = ['No Defense', 'Defense Active']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - CNN Defense Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def predict_defense_presence(self, new_data, threshold=0.5):
        """
        Predict if defense mechanisms are active in new network data using CNN
        """
        if self.cnn_model is None:
            raise ValueError("CNN model not trained yet!")

        # Scale new data
        new_data_scaled = self.scaler.transform(new_data)

        # Get predictions and probabilities
        defense_probabilities = self.cnn_model.predict(new_data_scaled).flatten()

        # Custom threshold predictions
        custom_predictions = (defense_probabilities >= threshold).astype(int)

        # Convert to readable labels
        labels = ['No Defense', 'Defense Active']
        pred_labels = [labels[pred] for pred in custom_predictions]

        results = []
        for i, (pred, prob) in enumerate(zip(pred_labels, defense_probabilities)):
            results.append({
                'prediction': pred,
                'defense_probability': prob,
                'confidence': 'High' if prob > 0.8 or prob < 0.2 else 'Medium' if prob > 0.6 or prob < 0.4 else 'Low'
            })

        return results

    def get_feature_importance_cnn(self, results):
        """
        Approximate feature importance for CNN using permutation importance
        """
        if self.cnn_model is None:
            print("Model not trained yet!")
            return None

        X_test = results['X_test']
        y_test = results['y_test']

        # Baseline accuracy
        baseline_accuracy = self.cnn_model.evaluate(X_test, y_test, verbose=0)[1]
        
        feature_importance = []
        
        print("Calculating CNN feature importance (this may take a while)...")
        
        for i, feature_name in enumerate(self.feature_names):
            # Create a copy of test data
            X_test_permuted = X_test.copy()
            
            # Permute the feature
            np.random.shuffle(X_test_permuted[:, i])
            
            # Calculate new accuracy
            permuted_accuracy = self.cnn_model.evaluate(X_test_permuted, y_test, verbose=0)[1]
            
            # Importance is the drop in accuracy
            importance = baseline_accuracy - permuted_accuracy
            feature_importance.append(importance)
            
            if i % 3 == 0:  # Progress indicator
                print(f"Processed {i+1}/{len(self.feature_names)} features")

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_cnn_feature_importance(self, importance_df, top_n=15):
        """
        Plot feature importance from CNN permutation analysis
        """
        if importance_df is None:
            print("Feature importance not calculated yet!")
            return None

        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)

        sns.barplot(data=top_features, x='importance', y='feature', palette='plasma')
        plt.title(f'Top {top_n} Most Important Features for CNN Defense Detection')
        plt.xlabel('Feature Importance (Accuracy Drop)')
        plt.ylabel('Network Metrics')
        plt.tight_layout()
        plt.show()

        print(f"\nTop {min(top_n, len(importance_df))} CNN Defense Detection Features:")
        print("-" * 50)
        for i, row in top_features.iterrows():
            print(f"{i + 1:2d}. {row['feature']:<30}: {row['importance']:.4f}")

        return top_features

    def save_cnn_model_results(self, results, importance_df=None, output_dir="./cnn_defense_results/"):
        """
        Save CNN model results and analysis
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save feature importance if available
        if importance_df is not None:
            importance_df.to_csv(
                os.path.join(output_dir, 'cnn_feature_importance.csv'), index=False
            )

        # Save model architecture
        with open(os.path.join(output_dir, 'cnn_model_architecture.txt'), 'w') as f:
            self.cnn_model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Save model info
        with open(os.path.join(output_dir, 'cnn_model_info.txt'), 'w') as f:
            f.write("CNN BINARY DEFENSE DETECTION MODEL\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Test accuracy: {results['test_accuracy']:.4f}\n")
            f.write(f"Test precision: {results['test_precision']:.4f}\n")
            f.write(f"Test recall: {results['test_recall']:.4f}\n")
            f.write(f"Test AUC: {results['auc_score']:.4f}\n")
            f.write(f"Number of features: {len(self.feature_names)}\n")
            f.write(f"Input shape: {self.input_shape}\n")
            f.write(f"Training epochs: {len(self.history.history['loss'])}\n\n")
            
            if importance_df is not None:
                f.write("Top 10 most important features:\n")
                for i, row in importance_df.head(10).iterrows():
                    f.write(f"{i + 1:2d}. {row['feature']:<25}: {row['importance']:.4f}\n")

        # Save the trained model
        self.cnn_model.save(os.path.join(output_dir, 'cnn_defense_model.h5'))

    def cross_validate_cnn(self, X, y, cv_folds=5):
        """
        Perform cross-validation for CNN model
        """
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Training fold {fold + 1}/{cv_folds}...")
            
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Scale data
            scaler_cv = StandardScaler()
            X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
            X_val_cv_scaled = scaler_cv.transform(X_val_cv)
            
            # Create and train model
            model_cv = self.create_cnn_architecture((X_train_cv_scaled.shape[1],))
            model_cv.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train with early stopping
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            model_cv.fit(
                X_train_cv_scaled, y_train_cv,
                validation_data=(X_val_cv_scaled, y_val_cv),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            val_loss, val_accuracy = model_cv.evaluate(X_val_cv_scaled, y_val_cv, verbose=0)
            cv_scores.append(val_accuracy)
            
            print(f"Fold {fold + 1} accuracy: {val_accuracy:.4f}")
        
        cv_scores = np.array(cv_scores)
        print(f"\nCross-validation results:")
        print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores


def main():
    """
    Main execution function for CNN binary defense detection
    """
    print("🧠 CNN BINARY DEFENSE DETECTION 🧠")
    print("=" * 70)
    print("Goal: Detect if defense mechanisms are active using Deep Learning")
    print("=" * 70)

    # Initialize detector
    detector = CNNDefenseDetector("./simulations/features/")

    try:
        # Step 1: Load data
        print("\nStep 1: Loading simulation data...")
        df = detector.load_simulation_data()

        # Step 2: Preprocess data for CNN
        print("\nStep 2: Preprocessing data for CNN...")
        X, y, df_pivot = detector.preprocess_data_for_cnn(df)

        # Step 3: Cross-validation (optional, can be skipped for faster execution)
        # cv_scores = detector.cross_validate_cnn(X, y)

        # Step 4: Train CNN model
        print("\nStep 4: Training CNN model...")
        results = detector.train_cnn_model(X, y)

        # Step 5: Evaluate model
        print("\nStep 5: Evaluating CNN model...")
        eval_results = detector.evaluate_cnn_model(results)

        # Step 6: Plot results
        print("\nStep 6: Generating visualizations...")
        detector.plot_training_history()
        detector.plot_confusion_matrix(eval_results['confusion_matrix'])
        optimal_threshold = detector.plot_roc_curve(results)

        # Step 7: Calculate feature importance (optional - takes time)
        print("\nStep 7: Calculating feature importance...")
        importance_df = detector.get_feature_importance_cnn(results)
        if importance_df is not None:
            top_features = detector.plot_cnn_feature_importance(importance_df)

        # Step 8: Save results
        print("\nStep 8: Saving CNN model results...")
        detector.save_cnn_model_results(results, importance_df)

        print("\n" + "=" * 70)
        print("CNN DEFENSE DETECTION ANALYSIS COMPLETE!")
        print("=" * 70)
        print("Key Results:")
        print(f"• CNN achieved {results['test_accuracy']:.1%} accuracy in detecting defenses")
        print(f"• AUC Score: {results['auc_score']:.3f} (closer to 1.0 = better)")
        print(f"• Precision: {eval_results['precision']:.3f} (false positive rate)")
        print(f"• Recall: {eval_results['recall']:.3f} (defense detection rate)")
        print(f"• F1-Score: {eval_results['f1_score']:.3f}")

        if importance_df is not None:
            print(f"\nTop 3 Defense Indicators (CNN):")
            for i, row in importance_df.head(3).iterrows():
                print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")

        print(f"\nOptimal detection threshold: {optimal_threshold:.3f}")
        print("\nCNN model saved and ready for defense detection in new network simulations!")

        return detector, results

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()