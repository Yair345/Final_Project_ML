#!/usr/bin/env python3
"""
Binary Defense Detection - Gradient Boosting Machines
Goal: Detect whether defense mechanisms are active in the network or not (regardless of attacks)
Using XGBoost and LightGBM for comparison with Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Gradient Boosting libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier


class BinaryDefenseGBMDetector:
    def __init__(self, data_path="./simulations/features/"):
        """
        Binary defense detector using Gradient Boosting Machines
        """
        self.data_path = data_path
        self.xgb_model = None
        self.lgb_model = None
        self.sklearn_gbm_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_results = {}

    def load_simulation_data(self):
        """
        Load data and group scenarios with defense into one category
        """
        all_data = []

        # Binary mapping: defense active or not
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
        y = df_pivot['defense_active']

        # Handle missing values
        X = X.fillna(X.median())

        # Store feature names
        self.feature_names = feature_cols

        print(f"Features available: {len(feature_cols)}")
        print(f"Binary classification: {y.value_counts().to_dict()}")

        return X, y, df_pivot

    def train_xgboost_model(self, X_train, X_test, y_train, y_test, random_state=42):
        """
        Train XGBoost model for binary defense detection
        """
        print("\n" + "="*50)
        print("TRAINING XGBOOST MODEL")
        print("="*50)

        # XGBoost hyperparameters
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [1, len(y_train[y_train==0]) / len(y_train[y_train==1])]  # Handle imbalance
        }

        # Grid search for XGBoost
        xgb_base = xgb.XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        xgb_grid = GridSearchCV(
            xgb_base, xgb_params, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )

        print("Starting XGBoost hyperparameter tuning...")
        xgb_grid.fit(X_train, y_train)

        self.xgb_model = xgb_grid.best_estimator_
        
        # Predictions
        y_pred_xgb = self.xgb_model.predict(X_test)
        y_pred_proba_xgb = self.xgb_model.predict_proba(X_test)[:, 1]

        # Evaluate
        xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
        xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)

        print(f"XGBoost Best Parameters: {xgb_grid.best_params_}")
        print(f"XGBoost Test Accuracy: {xgb_accuracy:.4f}")
        print(f"XGBoost Test AUC: {xgb_auc:.4f}")

        # Feature importance
        xgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.model_results['xgboost'] = {
            'model': self.xgb_model,
            'y_pred': y_pred_xgb,
            'y_pred_proba': y_pred_proba_xgb,
            'accuracy': xgb_accuracy,
            'auc_score': xgb_auc,
            'best_params': xgb_grid.best_params_,
            'feature_importance': xgb_importance
        }

        return self.model_results['xgboost']

    def train_lightgbm_model(self, X_train, X_test, y_train, y_test, random_state=42):
        """
        Train LightGBM model for binary defense detection
        """
        print("\n" + "="*50)
        print("TRAINING LIGHTGBM MODEL")
        print("="*50)

        # LightGBM hyperparameters
        lgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'class_weight': ['balanced', None]
        }

        # Grid search for LightGBM
        lgb_base = lgb.LGBMClassifier(
            random_state=random_state,
            verbose=-1
        )
        
        lgb_grid = GridSearchCV(
            lgb_base, lgb_params, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )

        print("Starting LightGBM hyperparameter tuning...")
        lgb_grid.fit(X_train, y_train)

        self.lgb_model = lgb_grid.best_estimator_
        
        # Predictions
        y_pred_lgb = self.lgb_model.predict(X_test)
        y_pred_proba_lgb = self.lgb_model.predict_proba(X_test)[:, 1]

        # Evaluate
        lgb_accuracy = accuracy_score(y_test, y_pred_lgb)
        lgb_auc = roc_auc_score(y_test, y_pred_proba_lgb)

        print(f"LightGBM Best Parameters: {lgb_grid.best_params_}")
        print(f"LightGBM Test Accuracy: {lgb_accuracy:.4f}")
        print(f"LightGBM Test AUC: {lgb_auc:.4f}")

        # Feature importance
        lgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.model_results['lightgbm'] = {
            'model': self.lgb_model,
            'y_pred': y_pred_lgb,
            'y_pred_proba': y_pred_proba_lgb,
            'accuracy': lgb_accuracy,
            'auc_score': lgb_auc,
            'best_params': lgb_grid.best_params_,
            'feature_importance': lgb_importance
        }

        return self.model_results['lightgbm']

    def train_sklearn_gbm_model(self, X_train, X_test, y_train, y_test, random_state=42):
        """
        Train Scikit-learn Gradient Boosting model for comparison
        """
        print("\n" + "="*50)
        print("TRAINING SCIKIT-LEARN GBM MODEL")
        print("="*50)

        # Sklearn GBM hyperparameters
        gbm_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }

        # Grid search for Sklearn GBM
        gbm_base = GradientBoostingClassifier(random_state=random_state)
        
        gbm_grid = GridSearchCV(
            gbm_base, gbm_params, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )

        print("Starting Scikit-learn GBM hyperparameter tuning...")
        gbm_grid.fit(X_train, y_train)

        self.sklearn_gbm_model = gbm_grid.best_estimator_
        
        # Predictions
        y_pred_gbm = self.sklearn_gbm_model.predict(X_test)
        y_pred_proba_gbm = self.sklearn_gbm_model.predict_proba(X_test)[:, 1]

        # Evaluate
        gbm_accuracy = accuracy_score(y_test, y_pred_gbm)
        gbm_auc = roc_auc_score(y_test, y_pred_proba_gbm)

        print(f"Sklearn GBM Best Parameters: {gbm_grid.best_params_}")
        print(f"Sklearn GBM Test Accuracy: {gbm_accuracy:.4f}")
        print(f"Sklearn GBM Test AUC: {gbm_auc:.4f}")

        # Feature importance
        gbm_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.sklearn_gbm_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.model_results['sklearn_gbm'] = {
            'model': self.sklearn_gbm_model,
            'y_pred': y_pred_gbm,
            'y_pred_proba': y_pred_proba_gbm,
            'accuracy': gbm_accuracy,
            'auc_score': gbm_auc,
            'best_params': gbm_grid.best_params_,
            'feature_importance': gbm_importance
        }

        return self.model_results['sklearn_gbm']

    def train_all_gbm_models(self, X, y, test_size=0.3, random_state=42):
        """
        Train all GBM models and compare results
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features (especially important for sklearn GBM)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set size: {len(X_train)} samples")
        print(f"Test set size: {len(X_test)} samples")
        print(f"Training distribution: {dict(pd.Series(y_train).value_counts())}")

        # Train all models
        xgb_results = self.train_xgboost_model(X_train_scaled, X_test_scaled, y_train, y_test, random_state)
        lgb_results = self.train_lightgbm_model(X_train_scaled, X_test_scaled, y_train, y_test, random_state)
        gbm_results = self.train_sklearn_gbm_model(X_train_scaled, X_test_scaled, y_train, y_test, random_state)

        # Store test data for evaluation
        self.y_test = y_test
        self.X_test = X_test_scaled

        return {
            'xgboost': xgb_results,
            'lightgbm': lgb_results,
            'sklearn_gbm': gbm_results,
            'test_data': {'X_test': X_test_scaled, 'y_test': y_test}
        }

    def compare_all_models(self):
        """
        Compare all trained models side by side
        """
        print("\n" + "="*70)
        print("GBM MODELS COMPARISON - BINARY DEFENSE DETECTION")
        print("="*70)

        comparison_data = []
        
        for model_name, results in self.model_results.items():
            y_pred = results['y_pred']
            y_pred_proba = results['y_pred_proba']
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{results['accuracy']:.4f}",
                'AUC Score': f"{results['auc_score']:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1:.4f}"
            })

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Find best model by AUC
        best_model_name = max(self.model_results.keys(), 
                             key=lambda x: self.model_results[x]['auc_score'])
        best_auc = self.model_results[best_model_name]['auc_score']
        
        print(f"\n🏆 BEST MODEL: {best_model_name.upper()} (AUC: {best_auc:.4f})")

        return comparison_df, best_model_name

    def evaluate_best_model(self, best_model_name):
        """
        Detailed evaluation of the best performing model
        """
        print(f"\n" + "="*60)
        print(f"DETAILED EVALUATION - {best_model_name.upper()}")
        print("="*60)

        results = self.model_results[best_model_name]
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']

        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred,
                                    target_names=['No Defense', 'Defense Active']))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print("\nConfusion Matrix Interpretation:")
        print(f"True Negatives (correctly identified no defense): {cm[0, 0]}")
        print(f"False Positives (incorrectly detected defense): {cm[0, 1]}")
        print(f"False Negatives (missed defense): {cm[1, 0]}")
        print(f"True Positives (correctly detected defense): {cm[1, 1]}")

        # Cross-validation score
        best_model = results['model']
        cv_scores = cross_val_score(best_model, self.X_test, self.y_test, cv=5, scoring='roc_auc')
        print(f"\nCross-validation AUC scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return cm

    def plot_models_comparison(self):
        """
        Plot ROC curves for all models
        """
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green']
        
        for i, (model_name, results) in enumerate(self.model_results.items()):
            y_pred_proba = results['y_pred_proba']
            auc_score = results['auc_score']
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'{model_name.upper()} (AUC = {auc_score:.4f})')

        # Random classifier line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison - Defense Detection (All GBM Models)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance_comparison(self, top_n=10):
        """
        Compare feature importance across all models
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, (model_name, results) in enumerate(self.model_results.items()):
            importance_df = results['feature_importance'].head(top_n)
            
            sns.barplot(data=importance_df, x='importance', y='feature', 
                       ax=axes[i], palette='viridis')
            axes[i].set_title(f'{model_name.upper()} - Top {top_n} Features')
            axes[i].set_xlabel('Feature Importance')
            
            if i > 0:
                axes[i].set_ylabel('')

        plt.tight_layout()
        plt.show()

        # Print top features for each model
        print(f"\nTop {top_n} Features Comparison:")
        print("="*50)
        
        for model_name, results in self.model_results.items():
            print(f"\n{model_name.upper()}:")
            for j, row in results['feature_importance'].head(top_n).iterrows():
                print(f"  {j+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")

    def predict_defense_presence(self, new_data, model_name='best', threshold=0.5):
        """
        Predict defense presence using specified model
        """
        if model_name == 'best':
            # Find best model by AUC
            model_name = max(self.model_results.keys(), 
                           key=lambda x: self.model_results[x]['auc_score'])

        if model_name not in self.model_results:
            raise ValueError(f"Model {model_name} not found!")

        model = self.model_results[model_name]['model']
        
        # Scale new data
        new_data_scaled = self.scaler.transform(new_data)

        # Get predictions and probabilities
        predictions = model.predict(new_data_scaled)
        probabilities = model.predict_proba(new_data_scaled)

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
                'confidence': 'High' if prob > 0.8 or prob < 0.2 else 'Medium' if prob > 0.6 or prob < 0.4 else 'Low',
                'model_used': model_name.upper()
            })

        return results

    def save_gbm_results(self, output_dir="./binary_defense_gbm_results/"):
        """
        Save all GBM model results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save comparison results
        comparison_df, best_model_name = self.compare_all_models()
        comparison_df.to_csv(os.path.join(output_dir, 'gbm_models_comparison.csv'), index=False)

        # Save individual model results
        for model_name, results in self.model_results.items():
            # Feature importance
            results['feature_importance'].to_csv(
                os.path.join(output_dir, f'{model_name}_feature_importance.csv'), index=False
            )

            # Model info
            with open(os.path.join(output_dir, f'{model_name}_model_info.txt'), 'w') as f:
                f.write(f"{model_name.upper()} DEFENSE DETECTION MODEL\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Best parameters: {results['best_params']}\n")
                f.write(f"Test accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Test AUC: {results['auc_score']:.4f}\n")
                f.write(f"Number of features: {len(self.feature_names)}\n\n")
                f.write("Top 10 most important features:\n")
                for i, row in results['feature_importance'].head(10).iterrows():
                    f.write(f"{i + 1:2d}. {row['feature']:<25}: {row['importance']:.4f}\n")

        # Save overall summary
        with open(os.path.join(output_dir, 'gbm_summary.txt'), 'w') as f:
            f.write("GRADIENT BOOSTING MACHINES - BINARY DEFENSE DETECTION\n")
            f.write("=" * 60 + "\n\n")
            f.write("Model Performance Summary:\n")
            f.write("-" * 30 + "\n")
            for model_name, results in self.model_results.items():
                f.write(f"{model_name.upper():<15}: Accuracy={results['accuracy']:.4f}, AUC={results['auc_score']:.4f}\n")
            f.write(f"\nBest Model: {best_model_name.upper()}\n")

        print(f"\nAll GBM results saved to: {output_dir}")


def main():
    """
    Main execution function for GBM binary defense detection
    """
    print("🚀 BINARY DEFENSE DETECTION WITH GRADIENT BOOSTING MACHINES 🚀")
    print("=" * 80)
    print("Goal: Detect if defense mechanisms are active using XGBoost, LightGBM & Sklearn GBM")
    print("=" * 80)

    # Initialize detector
    detector = BinaryDefenseGBMDetector("./simulations/features/")

    try:
        # Step 1: Load data
        print("\nStep 1: Loading simulation data...")
        df = detector.load_simulation_data()

        # Step 2: Preprocess data
        print("\nStep 2: Preprocessing data for binary classification...")
        X, y, df_pivot = detector.preprocess_data(df)

        # Step 3: Train all GBM models
        print("\nStep 3: Training all Gradient Boosting models...")
        all_results = detector.train_all_gbm_models(X, y)

        # Step 4: Compare models
        print("\nStep 4: Comparing all GBM models...")
        comparison_df, best_model_name = detector.compare_all_models()

        # Step 5: Evaluate best model
        print("\nStep 5: Detailed evaluation of best model...")
        cm = detector.evaluate_best_model(best_model_name)

        # Step 6: Generate visualizations
        print("\nStep 6: Generating comparison visualizations...")
        detector.plot_models_comparison()
        detector.plot_feature_importance_comparison()

        # Step 7: Save results
        print("\nStep 7: Saving all GBM results...")
        detector.save_gbm_results()

        print("\n" + "=" * 80)
        print("GRADIENT BOOSTING MACHINES ANALYSIS COMPLETE!")
        print("=" * 80)
        print("Key Results:")
        
        best_results = detector.model_results[best_model_name]
        print(f"🏆 Best Model: {best_model_name.upper()}")
        print(f"• Accuracy: {best_results['accuracy']:.1%}")
        print(f"• AUC Score: {best_results['auc_score']:.3f}")
        
        print(f"\nAll Models Performance:")
        for model_name, results in detector.model_results.items():
            print(f"• {model_name.upper()}: {results['accuracy']:.1%} accuracy, {results['auc_score']:.3f} AUC")

        print(f"\nTop 3 Defense Indicators (from {best_model_name.upper()}):")
        for i, row in best_results['feature_importance'].head(3).iterrows():
            print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")

        print("\nThese models can now detect defense presence in new network simulations!")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()