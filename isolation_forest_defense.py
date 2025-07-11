#!/usr/bin/env python3
"""
Defense Detection using Isolation Forest
Goal: Use anomaly detection to identify defense mechanisms in network data
Approach: Train on baseline (no defense) data, then detect defense scenarios as anomalies
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats


class IsolationForestDefenseDetector:
    def __init__(self, data_path="./simulations/features/"):
        """
        Isolation Forest for defense detection through anomaly detection
        """
        self.data_path = data_path
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.baseline_stats = None
        self.contamination_rate = 0.1  # Expected proportion of anomalies

    def load_simulation_data(self):
        """
        Load data with separate handling for baseline and defense scenarios
        """
        all_data = []
        
        scenarios = {
            'baseline': 0,  # Normal behavior (train on this)
            'defense': 1,   # Defense active (detect as anomaly)
            'defense_vs_attack': 1  # Defense + attack (detect as anomaly)
        }

        for scenario, label in scenarios.items():
            scenario_path = os.path.join(self.data_path, scenario)
            csv_files = glob.glob(os.path.join(scenario_path, "*.csv"))

            print(f"Loading {len(csv_files)} files from {scenario} scenario -> Label: {label}")

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['defense_active'] = label
                    df['scenario'] = scenario
                    df['file_source'] = os.path.basename(csv_file)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")

        if not all_data:
            raise ValueError("No data files found! Check your data path.")

        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Show distribution
        scenario_counts = combined_df['scenario'].value_counts()
        print(f"\nDataset distribution:")
        for scenario, count in scenario_counts.items():
            print(f"{scenario}: {count} samples")
        print(f"Total records loaded: {len(combined_df)}")

        return combined_df

    def preprocess_data(self, df):
        """
        Preprocess data for Isolation Forest anomaly detection
        """
        # Pivot the data to have metrics as columns
        df_pivot = df.pivot_table(
            index=['defense_active', 'scenario', 'file_source'],
            columns='Metric',
            values='Value',
            aggfunc='first'
        ).reset_index()

        # Clean column names
        df_pivot.columns.name = None

        # Separate features and labels
        feature_cols = [col for col in df_pivot.columns
                        if col not in ['defense_active', 'scenario', 'file_source']]

        X = df_pivot[feature_cols]
        y = df_pivot['defense_active']
        scenarios = df_pivot['scenario']

        # Handle missing values
        X = X.fillna(X.median())

        # Store feature names
        self.feature_names = feature_cols

        print(f"Features available: {len(feature_cols)}")
        print(f"Anomaly detection setup: Baseline={len(y[y==0])}, Defense scenarios={len(y[y==1])}")

        return X, y, scenarios, df_pivot

    def train_isolation_forest(self, X, y, scenarios, contamination='auto', random_state=42):
        """
        Train Isolation Forest on baseline data to detect defense mechanisms as anomalies
        """
        # Separate baseline (normal) and defense (anomaly) data
        baseline_mask = (y == 0)
        defense_mask = (y == 1)
        
        X_baseline = X[baseline_mask]
        X_defense = X[defense_mask]
        
        print(f"Training on {len(X_baseline)} baseline samples")
        print(f"Will test on {len(X_defense)} defense samples (expected anomalies)")
        
        # Scale the features - use RobustScaler for better anomaly detection
        self.scaler = RobustScaler()
        X_baseline_scaled = self.scaler.fit_transform(X_baseline)
        X_all_scaled = self.scaler.transform(X)
        
        # Calculate baseline statistics for analysis
        self.baseline_stats = {
            'mean': X_baseline.mean(),
            'std': X_baseline.std(),
            'median': X_baseline.median(),
            'q25': X_baseline.quantile(0.25),
            'q75': X_baseline.quantile(0.75)
        }
        
        # Hyperparameter tuning for Isolation Forest
        print("Starting hyperparameter tuning for Isolation Forest...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_samples': ['auto', 0.5, 0.8, 1.0],
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'max_features': [0.5, 0.8, 1.0],
            'bootstrap': [False, True],
            'random_state': [random_state]
        }
        
        # Custom scoring function for anomaly detection
        def anomaly_score(estimator, X_test, y_test):
            predictions = estimator.predict(X_test)
            # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
            predictions_binary = (predictions == -1).astype(int)
            # Calculate accuracy where anomalies should be defense scenarios
            return accuracy_score(y_test, predictions_binary)
        
        # Create a combined dataset for hyperparameter tuning
        X_combined = np.vstack([X_baseline_scaled, self.scaler.transform(X_defense)])
        y_combined = np.hstack([np.zeros(len(X_baseline_scaled)), np.ones(len(X_defense))])
        
        # Grid search
        iso_forest = IsolationForest(random_state=random_state)
        grid_search = GridSearchCV(
            iso_forest, param_grid, cv=3, 
            scoring=lambda est, X, y: anomaly_score(est, X, y),
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_combined, y_combined)
        
        # Best model
        self.isolation_forest = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train final model on baseline data only
        self.isolation_forest.fit(X_baseline_scaled)
        
        # Predict on all data
        anomaly_scores = self.isolation_forest.decision_function(X_all_scaled)
        predictions = self.isolation_forest.predict(X_all_scaled)
        
        # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
        predictions_binary = (predictions == -1).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions_binary)
        
        # For ROC AUC, use anomaly scores (more negative = more anomalous)
        auc_score = roc_auc_score(y, -anomaly_scores)  # Negative because lower scores = more anomalous
        
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc_score:.4f}")
        
        return {
            'model': self.isolation_forest,
            'X_baseline': X_baseline_scaled,
            'X_all': X_all_scaled,
            'y_true': y,
            'scenarios': scenarios,
            'predictions': predictions_binary,
            'anomaly_scores': anomaly_scores,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'best_params': grid_search.best_params_
        }

    def evaluate_isolation_forest(self, results):
        """
        Comprehensive evaluation for Isolation Forest anomaly detection
        """
        y_true = results['y_true']
        predictions = results['predictions']
        anomaly_scores = results['anomaly_scores']
        scenarios = results['scenarios']

        print("\n" + "=" * 70)
        print("ISOLATION FOREST DEFENSE DETECTION - EVALUATION RESULTS")
        print("=" * 70)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, predictions,
                                    target_names=['Normal (Baseline)', 'Anomaly (Defense)']))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, predictions)
        print(cm)
        print("\nConfusion Matrix Interpretation:")
        print(f"True Negatives (baseline correctly identified as normal): {cm[0, 0]}")
        print(f"False Positives (baseline incorrectly flagged as anomaly): {cm[0, 1]}")
        print(f"False Negatives (defense missed as normal): {cm[1, 0]}")
        print(f"True Positives (defense correctly detected as anomaly): {cm[1, 1]}")

        # Additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)

        print(f"\nKey Metrics:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUC Score: {results['auc_score']:.4f}")
        print(f"Precision: {precision:.4f} (of all anomaly predictions, how many were correct)")
        print(f"Recall: {recall:.4f} (of all actual defense scenarios, how many were detected)")
        print(f"F1-Score: {f1:.4f} (harmonic mean of precision and recall)")

        # Analyze by scenario
        print(f"\nDetection by Scenario:")
        for scenario in scenarios.unique():
            mask = scenarios == scenario
            scenario_pred = predictions[mask]
            scenario_true = y_true[mask]
            scenario_acc = accuracy_score(scenario_true, scenario_pred)
            anomaly_rate = np.mean(scenario_pred)
            print(f"  {scenario}: {scenario_acc:.3f} accuracy, {anomaly_rate:.3f} anomaly rate")

        # Anomaly score distribution analysis
        print(f"\nAnomaly Score Analysis:")
        baseline_scores = anomaly_scores[y_true == 0]
        defense_scores = anomaly_scores[y_true == 1]
        
        print(f"Baseline scores   - Mean: {baseline_scores.mean():.4f}, Std: {baseline_scores.std():.4f}")
        print(f"Defense scores    - Mean: {defense_scores.mean():.4f}, Std: {defense_scores.std():.4f}")
        print(f"Score separation: {abs(defense_scores.mean() - baseline_scores.mean()):.4f}")

        return {
            'classification_report': classification_report(y_true, predictions, output_dict=True),
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'baseline_scores': baseline_scores,
            'defense_scores': defense_scores
        }

    def plot_anomaly_scores_distribution(self, results):
        """
        Plot distribution of anomaly scores for baseline vs defense scenarios
        """
        y_true = results['y_true']
        anomaly_scores = results['anomaly_scores']
        scenarios = results['scenarios']
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Score distribution by defense status
        plt.subplot(1, 3, 1)
        baseline_scores = anomaly_scores[y_true == 0]
        defense_scores = anomaly_scores[y_true == 1]
        
        plt.hist(baseline_scores, bins=50, alpha=0.7, label='Baseline (Normal)', color='blue', density=True)
        plt.hist(defense_scores, bins=50, alpha=0.7, label='Defense (Anomaly)', color='red', density=True)
        plt.axvline(0, color='black', linestyle='--', alpha=0.5, label='Decision Boundary')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Score distribution by scenario
        plt.subplot(1, 3, 2)
        for scenario in scenarios.unique():
            mask = scenarios == scenario
            scenario_scores = anomaly_scores[mask]
            plt.hist(scenario_scores, bins=30, alpha=0.6, label=scenario, density=True)
        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Score Distribution by Scenario')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: ROC Curve
        plt.subplot(1, 3, 3)
        fpr, tpr, _ = roc_curve(y_true, -anomaly_scores)  # Negative because lower = more anomalous
        auc_score = results['auc_score']
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Anomaly Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix heatmap for anomaly detection
        """
        plt.figure(figsize=(8, 6))
        labels = ['Normal\n(Baseline)', 'Anomaly\n(Defense)']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Isolation Forest Anomaly Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def analyze_feature_anomalies(self, X, y, scenarios):
        """
        Analyze which features contribute most to anomaly detection
        """
        print("\n" + "=" * 70)
        print("FEATURE ANOMALY ANALYSIS")
        print("=" * 70)
        
        baseline_mask = (y == 0)
        defense_mask = (y == 1)
        
        X_baseline = X[baseline_mask]
        X_defense = X[defense_mask]
        
        # Calculate statistics for each feature
        anomaly_indicators = []
        
        for feature in self.feature_names:
            if feature in X.columns:
                baseline_values = X_baseline[feature]
                defense_values = X_defense[feature]
                
                # Statistical tests
                stat, p_value = stats.mannwhitneyu(baseline_values, defense_values, alternative='two-sided')
                
                # Effect size (standardized difference)
                pooled_std = np.sqrt(((baseline_values.std() ** 2) + (defense_values.std() ** 2)) / 2)
                effect_size = abs(defense_values.mean() - baseline_values.mean()) / pooled_std if pooled_std > 0 else 0
                
                # Outlier ratio in defense scenarios
                Q1 = baseline_values.quantile(0.25)
                Q3 = baseline_values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_in_defense = ((defense_values < lower_bound) | (defense_values > upper_bound)).sum()
                outlier_ratio = outliers_in_defense / len(defense_values)
                
                anomaly_indicators.append({
                    'feature': feature,
                    'baseline_mean': baseline_values.mean(),
                    'defense_mean': defense_values.mean(),
                    'difference': defense_values.mean() - baseline_values.mean(),
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'outlier_ratio': outlier_ratio
                })
        
        # Convert to DataFrame and sort by effect size
        anomaly_df = pd.DataFrame(anomaly_indicators)
        anomaly_df = anomaly_df.sort_values('effect_size', ascending=False)
        
        print("Top 10 features indicating defense anomalies:")
        print("-" * 50)
        for i, row in anomaly_df.head(10).iterrows():
            print(f"{row['feature']:<25}: Effect size={row['effect_size']:.3f}, "
                  f"Outlier ratio={row['outlier_ratio']:.3f}, p-value={row['p_value']:.3e}")
        
        return anomaly_df

    def plot_feature_anomaly_indicators(self, anomaly_df, top_n=15):
        """
        Plot top features that indicate anomalies
        """
        plt.figure(figsize=(15, 10))
        
        top_features = anomaly_df.head(top_n)
        
        # Plot 1: Effect sizes
        plt.subplot(2, 2, 1)
        sns.barplot(data=top_features, x='effect_size', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Features by Effect Size')
        plt.xlabel('Standardized Effect Size')
        
        # Plot 2: Outlier ratios
        plt.subplot(2, 2, 2)
        sns.barplot(data=top_features, x='outlier_ratio', y='feature', palette='plasma')
        plt.title(f'Top {top_n} Features by Outlier Ratio')
        plt.xlabel('Outlier Ratio in Defense Scenarios')
        
        # Plot 3: Statistical significance
        plt.subplot(2, 2, 3)
        top_features_copy = top_features.copy()
        top_features_copy['log_p_value'] = -np.log10(top_features_copy['p_value'])
        sns.barplot(data=top_features_copy, x='log_p_value', y='feature', palette='coolwarm')
        plt.title(f'Statistical Significance (-log10 p-value)')
        plt.xlabel('-log10(p-value)')
        
        # Plot 4: Mean differences
        plt.subplot(2, 2, 4)
        sns.barplot(data=top_features, x='difference', y='feature', palette='RdBu_r')
        plt.title(f'Mean Difference (Defense - Baseline)')
        plt.xlabel('Mean Difference')
        
        plt.tight_layout()
        plt.show()

    def predict_defense_anomalies(self, new_data, threshold=0.0):
        """
        Predict if new network data contains defense mechanisms (anomalies)
        """
        if self.isolation_forest is None:
            raise ValueError("Model not trained yet!")
        
        # Scale new data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Get anomaly scores and predictions
        anomaly_scores = self.isolation_forest.decision_function(new_data_scaled)
        predictions = self.isolation_forest.predict(new_data_scaled)
        
        # Custom threshold predictions
        custom_predictions = (anomaly_scores < threshold).astype(int)
        
        # Convert to readable labels
        labels = ['Normal (No Defense)', 'Anomaly (Defense Active)']
        pred_labels = [labels[pred] for pred in custom_predictions]
        
        results = []
        for i, (pred, score) in enumerate(zip(pred_labels, anomaly_scores)):
            confidence = 'High' if abs(score) > 0.2 else 'Medium' if abs(score) > 0.1 else 'Low'
            results.append({
                'prediction': pred,
                'anomaly_score': score,
                'confidence': confidence
            })
        
        return results

    def save_isolation_forest_results(self, results, anomaly_df, output_dir="./isolation_forest_results/"):
        """
        Save Isolation Forest results and analysis
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature anomaly indicators
        anomaly_df.to_csv(
            os.path.join(output_dir, 'feature_anomaly_indicators.csv'), index=False
        )
        
        # Save model info
        with open(os.path.join(output_dir, 'isolation_forest_info.txt'), 'w') as f:
            f.write("ISOLATION FOREST DEFENSE DETECTION MODEL\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best parameters: {results['best_params']}\n")
            f.write(f"Test accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Test AUC: {results['auc_score']:.4f}\n")
            f.write(f"Number of features: {len(self.feature_names)}\n\n")
            f.write("Top 10 anomaly indicator features:\n")
            for i, row in anomaly_df.head(10).iterrows():
                f.write(f"{i+1:2d}. {row['feature']:<25}: Effect size={row['effect_size']:.3f}\n")


def main():
    """
    Main execution function for Isolation Forest defense detection
    """
    print("🔍 DEFENSE DETECTION USING ISOLATION FOREST 🔍")
    print("=" * 70)
    print("Goal: Detect defense mechanisms through anomaly detection")
    print("Approach: Train on baseline, detect defense scenarios as anomalies")
    print("=" * 70)

    # Initialize detector
    detector = IsolationForestDefenseDetector("./simulations/features/")

    try:
        # Step 1: Load data
        print("\nStep 1: Loading simulation data...")
        df = detector.load_simulation_data()

        # Step 2: Preprocess data
        print("\nStep 2: Preprocessing data for anomaly detection...")
        X, y, scenarios, df_pivot = detector.preprocess_data(df)

        # Step 3: Train Isolation Forest
        print("\nStep 3: Training Isolation Forest model...")
        results = detector.train_isolation_forest(X, y, scenarios)

        # Step 4: Evaluate model
        print("\nStep 4: Evaluating anomaly detection model...")
        eval_results = detector.evaluate_isolation_forest(results)

        # Step 5: Analyze feature anomalies
        print("\nStep 5: Analyzing feature anomaly indicators...")
        anomaly_df = detector.analyze_feature_anomalies(X, y, scenarios)

        # Step 6: Plot results
        print("\nStep 6: Generating visualizations...")
        detector.plot_anomaly_scores_distribution(results)
        detector.plot_confusion_matrix(eval_results['confusion_matrix'])
        detector.plot_feature_anomaly_indicators(anomaly_df)

        # Step 7: Save results
        print("\nStep 7: Saving Isolation Forest results...")
        detector.save_isolation_forest_results(results, anomaly_df)

        print("\n" + "=" * 70)
        print("ISOLATION FOREST DEFENSE DETECTION ANALYSIS COMPLETE!")
        print("=" * 70)
        print("Key Results:")
        print(f"• Model achieved {results['accuracy']:.1%} accuracy in anomaly detection")
        print(f"• AUC Score: {results['auc_score']:.3f} (closer to 1.0 = better)")
        print(f"• Precision: {eval_results['precision']:.3f} (false positive rate)")
        print(f"• Recall: {eval_results['recall']:.3f} (defense detection rate)")

        print(f"\nTop 3 Anomaly Indicators:")
        for i, row in anomaly_df.head(3).iterrows():
            print(f"  {i+1}. {row['feature']}: Effect size={row['effect_size']:.3f}")

        print(f"\nScore Separation: {abs(eval_results['defense_scores'].mean() - eval_results['baseline_scores'].mean()):.3f}")
        print("\nThis model can now detect defense mechanisms as anomalies in new network data!")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()