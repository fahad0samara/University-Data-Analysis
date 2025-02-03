import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import KFold

class AdvancedModelEvaluator:
    def __init__(self, X, y, model_type='regressor'):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.scaler = RobustScaler()  # Using RobustScaler to handle outliers better
        self.X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, y, test_size=0.2, random_state=42
        )
        
    def train_multiple_models(self):
        """Train and compare multiple models"""
        if self.model_type == 'regressor':
            models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'XGBoost': xgb.XGBRegressor(random_state=42)
            }
            
            results = {}
            for name, model in models.items():
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                results[name] = {'MSE': mse, 'R2': r2, 'model': model}
            
            # Select best model based on R2 score
            self.best_model = max(results.items(), key=lambda x: x[1]['R2'])
            print("\nModel Comparison:")
            for name, metrics in results.items():
                print(f"{name}:")
                print(f"  MSE: {metrics['MSE']:.2e}")
                print(f"  R2: {metrics['R2']:.3f}")
            print(f"\nBest Model: {self.best_model[0]}")
            
            return self.best_model[1]['model']
        else:
            # For classification, we'll stick with RandomForest for now
            self.best_model = RandomForestClassifier(random_state=42)
            self.best_model.fit(self.X_train, self.y_train)
            return self.best_model
    
    def perform_cross_validation(self, model, n_splits=5):
        """Perform k-fold cross validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        if self.model_type == 'regressor':
            scores_r2 = cross_val_score(model, self.X_scaled, self.y, cv=kf, scoring='r2')
            scores_mse = -cross_val_score(model, self.X_scaled, self.y, cv=kf, scoring='neg_mean_squared_error')
            
            print("\nCross-Validation Results:")
            print(f"R² Score: {scores_r2.mean():.3f} (+/- {scores_r2.std() * 2:.3f})")
            print(f"MSE: {scores_mse.mean():.2e} (+/- {scores_mse.std() * 2:.2e})")
        else:
            scores_acc = cross_val_score(model, self.X_scaled, self.y, cv=kf, scoring='accuracy')
            print("\nCross-Validation Results:")
            print(f"Accuracy: {scores_acc.mean():.3f} (+/- {scores_acc.std() * 2:.3f})")
    
    def analyze_predictions(self, model):
        """Detailed analysis of model predictions"""
        y_pred = model.predict(self.X_test)
        
        if self.model_type == 'regressor':
            # Calculate prediction intervals using bootstrapping
            n_bootstraps = 1000
            predictions = np.zeros((n_bootstraps, len(self.X_test)))
            
            for i in range(n_bootstraps):
                # Bootstrap sample
                indices = np.random.randint(0, len(self.X_train), len(self.X_train))
                sample_X = self.X_train[indices]
                sample_y = self.y_train.iloc[indices] if isinstance(self.y_train, pd.Series) else self.y_train[indices]
                
                # Train and predict
                model.fit(sample_X, sample_y)
                predictions[i,:] = model.predict(self.X_test)
            
            # Calculate confidence intervals
            pred_intervals = np.percentile(predictions, [2.5, 97.5], axis=0)
            
            # Plot actual vs predicted with confidence intervals
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values with 95% Confidence Intervals')
            
            # Add confidence intervals
            plt.fill_between(self.y_test, 
                           pred_intervals[0,:], 
                           pred_intervals[1,:], 
                           alpha=0.2, 
                           color='gray', 
                           label='95% Confidence Interval')
            plt.legend()
            plt.savefig('prediction_analysis.png')
            plt.close()
            
        else:
            # For classification, create a detailed confusion matrix visualization
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Detailed Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig('detailed_confusion_matrix.png')
            plt.close()
            
            # Print detailed classification report
            print("\nDetailed Classification Report:")
            print(classification_report(self.y_test, y_pred))
    
    def analyze_feature_importance(self, model, feature_names):
        """Detailed feature importance analysis"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            # Plot feature importances
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances with Standard Deviation")
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), 
                      [feature_names[i] for i in indices], 
                      rotation=45)
            plt.tight_layout()
            plt.savefig('detailed_feature_importance.png')
            plt.close()
            
            # Print detailed feature importance analysis
            print("\nFeature Importance Analysis:")
            for idx in indices:
                print(f"{feature_names[idx]}: {importance[idx]:.4f}")
            
            # Calculate feature correlations
            feature_correlations = pd.DataFrame(self.X, columns=feature_names).corrwith(
                pd.Series(self.y)
            )
            print("\nFeature Correlations with Target:")
            print(feature_correlations.sort_values(ascending=False))

def load_and_prepare_data():
    """Load and prepare the university dataset with advanced preprocessing"""
    # Load data with proper encoding
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv('NorthAmericaUniversities.csv', encoding=encoding)
            break
        except:
            continue
    
    # Clean and convert numeric columns
    df['Established'] = pd.to_numeric(df['Established'], errors='coerce')
    df['Academic Staff'] = pd.to_numeric(df['Academic Staff'].astype(str).str.replace(',', ''), errors='coerce')
    df['Number of Students'] = pd.to_numeric(df['Number of Students'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Clean Endowment column
    df['Endowment'] = df['Endowment'].astype(str).str.replace('$', '').str.replace('B', '').str.replace(',', '')
    df['Endowment'] = pd.to_numeric(df['Endowment'], errors='coerce') * 1e9
    
    # Calculate derived features
    df['Age'] = 2024 - df['Established']
    df['Student_Staff_Ratio'] = df['Number of Students'] / df['Academic Staff']
    df['Students_per_Age'] = df['Number of Students'] / df['Age']  # New feature
    
    # Handle missing values
    df = df.dropna(subset=['Endowment'])
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())  # Using median instead of mean
    
    # Prepare features
    features = ['Age', 'Academic Staff', 'Number of Students', 
               'Student_Staff_Ratio', 'Students_per_Age']
    
    X = df[features]
    y_regression = df['Endowment']
    
    # Create tier labels
    df['Tier'] = pd.qcut(df['Endowment'], q=3, labels=['Lower', 'Middle', 'Upper'])
    y_classification = df['Tier']
    
    return X, y_regression, y_classification, features

def main():
    print("Loading and preparing data with advanced preprocessing...")
    X, y_regression, y_classification, features = load_and_prepare_data()
    
    # Regression Analysis
    print("\nPerforming Advanced Regression Analysis...")
    reg_evaluator = AdvancedModelEvaluator(X, y_regression, 'regressor')
    best_reg_model = reg_evaluator.train_multiple_models()
    reg_evaluator.perform_cross_validation(best_reg_model)
    reg_evaluator.analyze_predictions(best_reg_model)
    reg_evaluator.analyze_feature_importance(best_reg_model, features)
    
    # Classification Analysis
    print("\nPerforming Advanced Classification Analysis...")
    clf_evaluator = AdvancedModelEvaluator(X, y_classification, 'classifier')
    best_clf_model = clf_evaluator.train_multiple_models()
    clf_evaluator.perform_cross_validation(best_clf_model)
    clf_evaluator.analyze_predictions(best_clf_model)
    clf_evaluator.analyze_feature_importance(best_clf_model, features)
    
    print("\nAdvanced evaluation completed! Check the generated plots for detailed analysis.")

if __name__ == "__main__":
    main()
