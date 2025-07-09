#!/usr/bin/env python3
"""
Binary Defense Detection - Random Forest
Goal: Detect whether defense mechanisms are active in the network or not (regardless of attacks)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob


class BinaryDefenseDetector:
    def __init__(self, data_path="./simulations/features/"):
        """
        Binary defense detector: with defense (1) or without defense (0)
        """
        self.data_path = data_path
        self.rf_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None

    def load_simulation_data(self):
        """
        Load data and group scenarios with defense into one category
        """
        all_data = []

        # New mapping: only 2 categories
        scenarios = {
            'baseline': 0,  # No defense
            'defense': 1,  # Defense active
            'defense_vs_attack': 1  # Defense active (doesn't matter if attack is also present)
        }

        for scenario, binary_label in scenarios.items():
            scenario_path = os.path.join(self.data_path, scenario)
            csv_files = glob.glob(os.path.join(scenario_path, "*.csv"))

            print(f"Loading {len(csv_files)} files from {scenario} scenario -> Label: {binary_label}")

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['defense_active'] = binary_label  # 0 = no defense, 1 = defense active
                    df['original_scenario'] = scenario  # Keep original name for tracking
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

    def preprocess_data(self, df):
        """
        Preprocess the simulation data for binary classification
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
        y = df_pivot['defense_active']  # Binary: 0 or 1

        # Handle missing values
        X = X.fillna(X.median())

        # Store feature names
        self.feature_names = feature_cols

        print(f"Features available: {len(feature_cols)}")
        print(f"Binary classification: {y.value_counts().to_dict()}")

        return X, y, df_pivot

    def train_binary_random_forest(self, X, y, test_size=0.3, random_state=42):
        """
        Train Random Forest for binary defense detection
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Starting hyperparameter tuning for binary classification...")

        # Hyperparameter tuning for binary classification
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced']  # Handle class imbalance if needed
        }

        # Grid search with cross-validation
        rf_base = RandomForestClassifier(random_state=random_state)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, scoring='roc_auc',  # Use AUC for binary classification
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train_scaled, y_train)

        # Best model
        self.rf_model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")

        # Train final model with best parameters
        self.rf_model.fit(X_train_scaled, y_train)

        # Predictions and probabilities
        y_pred = self.rf_model.predict(X_test_scaled)
        y_pred_proba = self.rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability of defense

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc_score:.4f}")

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'model': self.rf_model,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'best_params': grid_search.best_params_
        }

    def evaluate_binary_model(self, results):
        """
        Comprehensive evaluation for binary defense detection
        """
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']

        print("\n" + "=" * 60)
        print("BINARY DEFENSE DETECTION - EVALUATION RESULTS")
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
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUC Score: {results['auc_score']:.4f}")
        print(f"Precision: {precision:.4f} (of all defense predictions, how many were correct)")
        print(f"Recall: {recall:.4f} (of all actual defenses, how many were detected)")
        print(f"F1-Score: {f1:.4f} (harmonic mean of precision and recall)")

        # Cross-validation score
        cv_scores = cross_val_score(self.rf_model, results['X_train'],
                                    results['y_train'], cv=5, scoring='roc_auc')
        print(f"\nCross-validation AUC scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm,
            'cv_scores': cv_scores,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def plot_roc_curve(self, results):
        """
        Plot ROC curve for binary classification
        """
        y_test = results['y_test']
        y_pred_proba = results['y_pred_proba']

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = results['auc_score']

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Defense Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"At optimal threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")

        return optimal_threshold

    def plot_binary_confusion_matrix(self, cm):
        """
        Plot confusion matrix heatmap for binary classification
        """
        plt.figure(figsize=(8, 6))
        labels = ['No Defense', 'Defense Active']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Binary Defense Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def predict_defense_presence(self, new_data, threshold=0.5):
        """
        Predict if defense mechanisms are active in new network data
        """
        if self.rf_model is None:
            raise ValueError("Model not trained yet!")

        # Scale new data
        new_data_scaled = self.scaler.transform(new_data)

        # Get predictions and probabilities
        predictions = self.rf_model.predict(new_data_scaled)
        probabilities = self.rf_model.predict_proba(new_data_scaled)

        # Custom threshold predictions
        defense_probabilities = probabilities[:, 1]
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

    def analyze_defense_indicators(self, df_pivot):
        """
        Analyze which network metrics best indicate defense presence
        """
        print("\n" + "=" * 60)
        print("DEFENSE INDICATORS ANALYSIS")
        print("=" * 60)

        # Select only numeric columns for analysis (exclude text columns)
        numeric_cols = df_pivot.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'defense_active']  # Exclude the target variable

        # Group by defense status using only numeric columns
        defense_stats = df_pivot.groupby('defense_active')[numeric_cols].agg(['mean', 'std']).round(4)

        # Key metrics that should differ between defense/no-defense
        key_metrics = [
            'PacketDeliveryRatio', 'PacketLossRatio', 'EndToEndDelay', 'Jitter'
            'Throughput', 'AverageHopCount', 'HELLOPacketsPerSec', 'TCPacketsPerSec',
            'RoutingOverhead', 'NormalizedRoutingLoad', 'MACLayerOverhead', 'AverageTCPacketRows',
            'MIDPacketsPerSec', 'HNAPacketsPerSec', 'MACDataPacketsPerSec', 'MACControlPacketsPerSec'
        ]

        # Filter to only include metrics that exist in our data
        available_metrics = [metric for metric in key_metrics if metric in numeric_cols]

        print(f"\nAnalyzing {len(available_metrics)} key defense indicators:")
        print("=" * 40)

        for metric in available_metrics:
            try:
                no_def_mean = defense_stats.loc[0, (metric, 'mean')]
                def_mean = defense_stats.loc[1, (metric, 'mean')]
                difference = def_mean - no_def_mean

                print(f"\n{metric}:")
                print(f"  No Defense:   {no_def_mean:.4f}")
                print(f"  With Defense: {def_mean:.4f}")
                print(f"  Difference:   {difference:+.4f}")

                if abs(no_def_mean) > 0:  # Avoid division by zero
                    percent_change = (difference / abs(no_def_mean)) * 100
                    if abs(percent_change) > 10:  # More than 10% change
                        direction = "increases" if difference > 0 else "decreases"
                        print(f"  → Defense {direction} this metric by {abs(percent_change):.1f}%!")

            except KeyError:
                print(f"\nMetric '{metric}' not found in data")
                continue

        # Show all available metrics if the key ones are missing
        if not available_metrics:
            print(f"\nKey metrics not found. Available numeric columns:")
            for col in numeric_cols[:10]:  # Show first 10
                print(f"  - {col}")
            if len(numeric_cols) > 10:
                print(f"  ... and {len(numeric_cols) - 10} more columns")

    def save_binary_model_results(self, results, output_dir="./binary_defense_results/"):
        """
        Save binary model results and analysis
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save feature importance
        self.feature_importance.to_csv(
            os.path.join(output_dir, 'binary_feature_importance.csv'), index=False
        )

        # Save model info
        with open(os.path.join(output_dir, 'binary_model_info.txt'), 'w') as f:
            f.write("BINARY DEFENSE DETECTION MODEL\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Best parameters: {results['best_params']}\n")
            f.write(f"Test accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Test AUC: {results['auc_score']:.4f}\n")
            f.write(f"Number of features: {len(self.feature_names)}\n\n")
            f.write("Top 10 most important features:\n")
            for i, row in self.feature_importance.head(10).iterrows():
                f.write(f"{i + 1:2d}. {row['feature']:<25}: {row['importance']:.4f}\n")

    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance from Random Forest for binary classification
        """
        if self.feature_importance is None:
            print("Model not trained yet!")
            return None

        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)

        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features for Defense Detection')
        plt.xlabel('Feature Importance')
        plt.ylabel('Network Metrics')
        plt.tight_layout()
        plt.show()

        print(f"\nTop {min(top_n, len(self.feature_importance))} Defense Detection Features:")
        print("-" * 50)
        for i, row in top_features.iterrows():
            print(f"{i + 1:2d}. {row['feature']:<30}: {row['importance']:.4f}")

        return top_features


def main():
    """
    Main execution function for binary defense detection
    """
    print("🛡️  BINARY DEFENSE DETECTION WITH RANDOM FOREST 🛡️")
    print("=" * 70)
    print("Goal: Detect if defense mechanisms are active (regardless of attacks)")
    print("=" * 70)

    # Initialize detector
    detector = BinaryDefenseDetector("./simulations/features/")

    try:
        # Step 1: Load data
        print("\nStep 1: Loading simulation data...")
        df = detector.load_simulation_data()

        # Step 2: Preprocess data
        print("\nStep 2: Preprocessing data for binary classification...")
        X, y, df_pivot = detector.preprocess_data(df)

        # Step 3: Analyze defense indicators
        detector.analyze_defense_indicators(df_pivot)

        # Step 4: Train binary model
        print("\nStep 4: Training Binary Random Forest model...")
        results = detector.train_binary_random_forest(X, y)

        # Step 5: Evaluate model
        print("\nStep 5: Evaluating binary model...")
        eval_results = detector.evaluate_binary_model(results)

        # Step 6: Plot results
        print("\nStep 6: Generating visualizations...")
        top_features = detector.plot_feature_importance()
        detector.plot_binary_confusion_matrix(eval_results['confusion_matrix'])
        optimal_threshold = detector.plot_roc_curve(results)

        # Step 7: Save results
        print("\nStep 7: Saving binary model results...")
        detector.save_binary_model_results(results)

        print("\n" + "=" * 70)
        print("BINARY DEFENSE DETECTION ANALYSIS COMPLETE!")
        print("=" * 70)
        print("Key Results:")
        print(f"• Model achieved {results['accuracy']:.1%} accuracy in detecting defenses")
        print(f"• AUC Score: {results['auc_score']:.3f} (closer to 1.0 = better)")
        print(f"• Precision: {eval_results['precision']:.3f} (false positive rate)")
        print(f"• Recall: {eval_results['recall']:.3f} (defense detection rate)")

        print(f"\nTop 3 Defense Indicators:")
        for i, row in top_features.head(3).iterrows():
            print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")

        print(f"\nOptimal detection threshold: {optimal_threshold:.3f}")
        print("\nThis model can now detect defense presence in new network simulations!")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()