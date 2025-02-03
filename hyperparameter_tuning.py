import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
def load_data():
    try:
        # Try different encodings
        df = pd.read_csv('NorthAmericaUniversities.csv', encoding='latin-1')
    except:
        try:
            df = pd.read_csv('NorthAmericaUniversities.csv', encoding='utf-8')
        except:
            df = pd.read_csv('NorthAmericaUniversities.csv', encoding='cp1252')
    
    # Calculate derived features
    df['Age'] = 2024 - df['Established']
    df['Student_Staff_Ratio'] = df['Number of Students'] / df['Academic Staff']
    
    # Handle missing values
    df = df.dropna(subset=['Endowment'])
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Clean any potential string issues
    df = df.replace({r'[^\x00-\x7F]+': ''}, regex=True)
    
    return df

def prepare_features(df):
    # Features for prediction
    features = ['Age', 'Academic Staff', 'Number of Students', 
                'Student_Staff_Ratio', 'Minimum Tuition Cost', 
                'Volumes in Library']
    
    X = df[features]
    y_regression = df['Endowment']
    
    # Create tier labels for classification
    endowment_tertiles = np.percentile(df['Endowment'], [33.33, 66.66])
    df['Tier'] = pd.qcut(df['Endowment'], q=3, labels=['Lower', 'Middle', 'Upper'])
    y_classification = df['Tier']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_regression, y_classification, features

def tune_random_forest_regressor(X, y):
    # Define parameter grid for Random Forest Regressor
    param_grid_rf = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    
    rf_reg = RandomForestRegressor(random_state=42)
    
    # Perform Grid Search
    grid_search_rf = GridSearchCV(
        estimator=rf_reg,
        param_grid=param_grid_rf,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search_rf.fit(X, y)
    
    print("\nBest parameters for Random Forest Regressor:")
    print(grid_search_rf.best_params_)
    print("\nBest score (MSE):", -grid_search_rf.best_score_)
    
    return grid_search_rf.best_estimator_

def tune_random_forest_classifier(X, y):
    # Define parameter grid for Random Forest Classifier
    param_grid_rf_clf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    
    rf_clf = RandomForestClassifier(random_state=42)
    
    # Perform Grid Search
    grid_search_rf_clf = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid_rf_clf,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    grid_search_rf_clf.fit(X, y)
    
    print("\nBest parameters for Random Forest Classifier:")
    print(grid_search_rf_clf.best_params_)
    print("\nBest accuracy score:", grid_search_rf_clf.best_score_)
    
    return grid_search_rf_clf.best_estimator_

def plot_learning_curves(estimator, X, y, title):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=train_sizes,
        scoring='neg_mean_squared_error' if isinstance(estimator, RandomForestRegressor) else 'accuracy'
    )
    
    train_mean = np.mean(-train_scores if isinstance(estimator, RandomForestRegressor) else train_scores, axis=1)
    train_std = np.std(-train_scores if isinstance(estimator, RandomForestRegressor) else train_scores, axis=1)
    test_mean = np.mean(-test_scores if isinstance(estimator, RandomForestRegressor) else test_scores, axis=1)
    test_std = np.std(-test_scores if isinstance(estimator, RandomForestRegressor) else test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curves for {title}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'learning_curves_{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    print("Loading and preparing data...")
    df = load_data()
    X, y_regression, y_classification, features = prepare_features(df)
    
    print("\nTuning Random Forest Regressor for Endowment Prediction...")
    best_rf_reg = tune_random_forest_regressor(X, y_regression)
    
    print("\nTuning Random Forest Classifier for Tier Classification...")
    best_rf_clf = tune_random_forest_classifier(X, y_classification)
    
    print("\nGenerating learning curves...")
    plot_learning_curves(best_rf_reg, X, y_regression, "Random Forest Regressor")
    plot_learning_curves(best_rf_clf, X, y_classification, "Random Forest Classifier")
    
    print("\nHyperparameter tuning completed! Check the output for best parameters and learning curves.")

if __name__ == "__main__":
    main()
