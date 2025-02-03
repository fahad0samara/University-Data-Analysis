import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ModelEvaluator:
    def __init__(self, X, y, model, model_type='regressor'):
        self.X = X
        self.y = y
        self.model = model
        self.model_type = model_type
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        
    def evaluate_regression(self):
        """Evaluate regression model performance"""
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        print("\nRegression Metrics:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        # Residual Analysis
        residuals = self.y_test - self.y_pred
        
        # Plot residuals
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.savefig('residual_plot.png')
        plt.close()
        
        # Q-Q plot
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.savefig('qq_plot.png')
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def evaluate_classification(self):
        """Evaluate classification model performance"""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        class_report = classification_report(self.y_test, self.y_pred)
        
        print("\nClassification Metrics:")
        print(f"Accuracy Score: {accuracy:.2f}")
        print("\nClassification Report:")
        print(class_report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # For multi-class ROC curve
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(self.X_test)
            
            plt.figure(figsize=(10, 6))
            for i, class_name in enumerate(self.model.classes_):
                fpr, tpr, _ = roc_curve(
                    (self.y_test == class_name).astype(int), 
                    y_prob[:, i]
                )
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {class_name} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.savefig('roc_curves.png')
            plt.close()
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance"""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def plot_learning_curves(self):
        """Plot learning curves"""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X, self.y,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error' if self.model_type == 'regressor' else 'accuracy'
        )
        
        train_scores_mean = -np.mean(train_scores, axis=1) if self.model_type == 'regressor' else np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1) if self.model_type == 'regressor' else np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.fill_between(train_sizes, 
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, 
                        alpha=0.1, color="r")
        plt.fill_between(train_sizes, 
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, 
                        alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title("Learning Curves")
        plt.legend(loc="best")
        plt.savefig('learning_curves.png')
        plt.close()

def load_and_prepare_data():
    """Load and prepare the university dataset"""
    # Try different encodings
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
    
    # Clean Endowment column (remove '$' and 'B', convert to numeric)
    df['Endowment'] = df['Endowment'].astype(str).str.replace('$', '').str.replace('B', '').str.replace(',', '')
    df['Endowment'] = pd.to_numeric(df['Endowment'], errors='coerce') * 1e9  # Convert billions to actual values
    
    # Calculate derived features
    df['Age'] = 2024 - df['Established']
    df['Student_Staff_Ratio'] = df['Number of Students'] / df['Academic Staff']
    
    # Handle missing values
    df = df.dropna(subset=['Endowment'])
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Prepare features
    features = ['Age', 'Academic Staff', 'Number of Students', 
               'Student_Staff_Ratio']
    
    X = df[features]
    y_regression = df['Endowment']
    
    # Create tier labels for classification
    df['Tier'] = pd.qcut(df['Endowment'], q=3, labels=['Lower', 'Middle', 'Upper'])
    y_classification = df['Tier']
    
    return X, y_regression, y_classification, features

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    X, y_regression, y_classification, features = load_and_prepare_data()
    
    # Evaluate Regression Model
    print("\nEvaluating Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_evaluator = ModelEvaluator(X, y_regression, rf_reg, 'regressor')
    reg_evaluator.train_model()
    reg_metrics = reg_evaluator.evaluate_regression()
    reg_evaluator.plot_feature_importance(features)
    reg_evaluator.plot_learning_curves()
    
    # Evaluate Classification Model
    print("\nEvaluating Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_evaluator = ModelEvaluator(X, y_classification, rf_clf, 'classifier')
    clf_evaluator.train_model()
    clf_metrics = clf_evaluator.evaluate_classification()
    clf_evaluator.plot_feature_importance(features)
    clf_evaluator.plot_learning_curves()
    
    print("\nEvaluation completed! Check the generated plots for detailed analysis.")

if __name__ == "__main__":
    main()
