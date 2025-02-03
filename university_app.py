import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from scipy import stats

# Set page config
st.set_page_config(
    page_title="University Analysis Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the university data"""
    try:
        df = pd.read_csv('NorthAmericaUniversities.csv', encoding='latin-1')
        
        # Clean numeric columns
        df['Established'] = pd.to_numeric(df['Established'], errors='coerce')
        df['Academic Staff'] = pd.to_numeric(df['Academic Staff'].astype(str).str.replace(',', ''), errors='coerce')
        df['Number of Students'] = pd.to_numeric(df['Number of Students'].astype(str).str.replace(',', ''), errors='coerce')
        
        # Clean Endowment
        df['Endowment'] = df['Endowment'].astype(str).str.replace('$', '').str.replace('B', '').str.replace(',', '')
        df['Endowment'] = pd.to_numeric(df['Endowment'], errors='coerce') * 1e9
        
        # Feature Engineering
        df['Age'] = 2024 - df['Established']
        df['Student_Staff_Ratio'] = df['Number of Students'] / df['Academic Staff']
        df['Students_per_Age'] = df['Number of Students'] / df['Age']
        
        # Handle missing values
        df = df.dropna(subset=['Endowment'])
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_models(df):
    """Train regression and classification models"""
    features = ['Age', 'Academic Staff', 'Number of Students', 'Student_Staff_Ratio', 'Students_per_Age']
    X = df[features]
    y_regression = df['Endowment']
    
    # Create tier labels for classification
    df['Tier'] = pd.qcut(df['Endowment'], q=3, labels=['Lower', 'Middle', 'Upper'])
    label_encoder = LabelEncoder()
    y_classification = label_encoder.fit_transform(df['Tier'])
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_regression, test_size=0.2, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_scaled, y_classification, test_size=0.2, random_state=42
    )
    
    # Train models
    reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
    clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    
    reg_model.fit(X_train_reg, y_train_reg)
    clf_model.fit(X_train_clf, y_train_clf)
    
    return {
        'features': features,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'regression': {
            'model': reg_model,
            'X_test': X_test_reg,
            'y_test': y_test_reg
        },
        'classification': {
            'model': clf_model,
            'X_test': X_test_clf,
            'y_test': y_test_clf
        }
    }

def main():
    st.title("🎓 University Analysis Dashboard")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the data file and try again.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        ["Overview", "Data Explorer", "Endowment Analysis", "University Comparison", "Predictions", "ML Model Training"]
    )
    
    if page == "Overview":
        st.header("Dataset Overview")
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Universities", len(df))
        with col2:
            st.metric("Average Endowment", f"${df['Endowment'].mean()/1e9:.1f}B")
        with col3:
            st.metric("Average Age", f"{df['Age'].mean():.0f} years")
        with col4:
            st.metric("Total Students", f"{df['Number of Students'].sum():,.0f}")
        
        # Distribution of Endowments
        st.subheader("Distribution of University Endowments")
        fig = px.histogram(df, x='Endowment', nbins=30,
                          title='Distribution of University Endowments')
        fig.update_layout(xaxis_title='Endowment (USD)',
                         yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        st.subheader("Feature Correlations")
        numeric_cols = ['Age', 'Academic Staff', 'Number of Students', 
                       'Student_Staff_Ratio', 'Students_per_Age', 'Endowment']
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       x=numeric_cols,
                       y=numeric_cols)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Data Explorer":
        st.header("Data Explorer")
        
        # Data viewer
        st.subheader("Raw Data")
        st.dataframe(df)
        
        # Column selector for visualization
        st.subheader("Custom Visualization")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis", df.columns)
        with col2:
            y_axis = st.selectbox("Select Y-axis", df.columns)
        
        plot_type = st.radio("Select Plot Type", ["Scatter", "Line", "Bar"])
        
        if plot_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, 
                           hover_data=['Name'])
        elif plot_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis)
        else:
            fig = px.bar(df, x=x_axis, y=y_axis)
            
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Endowment Analysis":
        st.header("Endowment Analysis")
        
        # Train models
        models = train_models(df)
        
        # Feature importance
        st.subheader("Feature Importance for Endowment Prediction")
        importance = models['regression']['model'].feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': models['features'],
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature',
                    orientation='h',
                    title='Feature Importance in Predicting Endowment')
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance
        st.subheader("Model Performance")
        y_pred = models['regression']['model'].predict(models['regression']['X_test'])
        r2 = r2_score(models['regression']['y_test'], y_pred)
        rmse = np.sqrt(mean_squared_error(models['regression']['y_test'], y_pred))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"${rmse/1e9:.2f}B")
    
    elif page == "University Comparison":
        st.header("University Comparison")
        
        # University selector
        selected_unis = st.multiselect(
            "Select Universities to Compare",
            df['Name'].tolist(),
            default=df['Name'].head(3).tolist()
        )
        
        if selected_unis:
            comparison_data = df[df['Name'].isin(selected_unis)]
            
            # Radar chart of metrics
            metrics = ['Age', 'Academic Staff', 'Number of Students', 
                      'Student_Staff_Ratio', 'Endowment']
            
            fig = go.Figure()
            for uni in selected_unis:
                uni_data = comparison_data[comparison_data['Name'] == uni]
                values = uni_data[metrics].values[0]
                # Normalize values
                values_norm = (values - df[metrics].min()) / (df[metrics].max() - df[metrics].min())
                values_norm = np.append(values_norm, values_norm[0])  # Complete the circle
                
                fig.add_trace(go.Scatterpolar(
                    r=values_norm,
                    theta=metrics + [metrics[0]],
                    name=uni
                ))
                
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.subheader("Detailed Comparison")
            comparison_df = comparison_data[['Name', 'Established', 'Academic Staff', 
                                          'Number of Students', 'Endowment']]
            st.dataframe(comparison_df)
    
    elif page == "Predictions":
        st.header("University Predictions")
        
        # Input form for predictions
        st.subheader("Predict University Endowment")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("University Age", min_value=0, max_value=500, value=100)
            academic_staff = st.number_input("Academic Staff", min_value=0, value=1000)
        with col2:
            students = st.number_input("Number of Students", min_value=0, value=10000)
            student_staff_ratio = students / academic_staff if academic_staff > 0 else 0
            students_per_age = students / age if age > 0 else 0
        
        if st.button("Predict Endowment"):
            # Prepare input data
            input_data = np.array([[age, academic_staff, students, 
                                  student_staff_ratio, students_per_age]])
            
            # Train models
            models = train_models(df)
            
            # Scale input
            input_scaled = models['scaler'].transform(input_data)
            
            # Make predictions
            endowment_pred = models['regression']['model'].predict(input_scaled)[0]
            tier_pred = models['classification']['model'].predict(input_scaled)[0]
            
            # Display predictions
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Endowment", f"${endowment_pred/1e9:.2f}B")
            with col2:
                st.metric("Predicted Tier", models['label_encoder'].inverse_transform([tier_pred])[0])
            
            # Show prediction context
            st.subheader("Prediction Context")
            percentile = (df['Endowment'] < endowment_pred).mean() * 100
            st.write(f"This endowment would rank in the top {100-percentile:.1f}% of universities.")
    
    elif page == "ML Model Training":
        st.header("Machine Learning Model Training and Evaluation")
        
        # Model Selection
        st.subheader("Select Model Configuration")
        model_type = st.selectbox(
            "Choose Model Type",
            ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "AdaBoost", 
             "SVM", "Linear/Logistic Regression"]
        )
        
        # Advanced Options
        with st.expander("Advanced Model Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("Number of Estimators", 50, 500, 200, 50)
                max_depth = st.slider("Max Depth", 3, 50, 20, 1)
                min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1)
            with col2:
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2, 1)
                if model_type == "SVM":
                    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                    C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
        
        # Cross-validation Options
        with st.expander("Cross-validation Settings"):
            cv_folds = st.slider("Number of CV Folds", 3, 10, 5)
            n_jobs = st.slider("Number of Parallel Jobs", 1, 8, 4)
        
        # Feature Selection
        st.subheader("Feature Selection")
        available_features = ['Age', 'Academic Staff', 'Number of Students', 
                            'Student_Staff_Ratio', 'Students_per_Age']
        selected_features = st.multiselect(
            "Select Features for Training",
            available_features,
            default=available_features
        )
        
        # Train Test Split Configuration
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
        
        if st.button("Train Models"):
            # Prepare data
            X = df[selected_features]
            y_regression = df['Endowment']
            
            # Create and encode tier labels
            df['Tier'] = pd.qcut(df['Endowment'], q=3, labels=['Lower', 'Middle', 'Upper'])
            label_encoder = LabelEncoder()
            y_classification = label_encoder.fit_transform(df['Tier'])
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_scaled, y_regression, test_size=test_size, random_state=42
            )
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                X_scaled, y_classification, test_size=test_size, random_state=42
            )
            
            # Initialize models based on selection
            if model_type == "Random Forest":
                reg_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=n_jobs
                )
                clf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=n_jobs
                )
            elif model_type == "Gradient Boosting":
                reg_model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                clf_model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
            elif model_type == "XGBoost":
                reg_model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_child_weight=min_samples_leaf,
                    random_state=42,
                    n_jobs=n_jobs
                )
                clf_model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_child_weight=min_samples_leaf,
                    random_state=42,
                    n_jobs=n_jobs
                )
            elif model_type == "LightGBM":
                reg_model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_child_samples=min_samples_leaf,
                    random_state=42,
                    n_jobs=n_jobs
                )
                clf_model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_child_samples=min_samples_leaf,
                    random_state=42,
                    n_jobs=n_jobs
                )
            elif model_type == "AdaBoost":
                reg_model = AdaBoostRegressor(
                    n_estimators=n_estimators,
                    random_state=42
                )
                clf_model = AdaBoostClassifier(
                    n_estimators=n_estimators,
                    random_state=42
                )
            elif model_type == "SVM":
                reg_model = SVR(
                    kernel=kernel,
                    C=C
                )
                clf_model = SVC(
                    kernel=kernel,
                    C=C,
                    probability=True
                )
            else:  # Linear/Logistic Regression
                reg_model = LinearRegression(n_jobs=n_jobs)
                clf_model = LogisticRegression(
                    random_state=42,
                    n_jobs=n_jobs
                )
            
            with st.spinner("Training models and performing cross-validation..."):
                # Cross-validation for regression
                cv_scores_reg = cross_val_score(
                    reg_model, X_scaled, y_regression,
                    cv=cv_folds, scoring='r2', n_jobs=n_jobs
                )
                
                # Cross-validation for classification
                cv_scores_clf = cross_val_score(
                    clf_model, X_scaled, y_classification,
                    cv=cv_folds, scoring='accuracy', n_jobs=n_jobs
                )
                
                # Train models
                reg_model.fit(X_train_reg, y_train_reg)
                clf_model.fit(X_train_clf, y_train_clf)
                
                # Make predictions
                y_pred_reg = reg_model.predict(X_test_reg)
                y_pred_clf = clf_model.predict(X_test_clf)
                
                # Transform predictions back to original labels
                y_test_clf_labels = label_encoder.inverse_transform(y_test_clf)
                y_pred_clf_labels = label_encoder.inverse_transform(y_pred_clf)
                
                # Calculate metrics
                mse = mean_squared_error(y_test_reg, y_pred_reg)
                r2 = r2_score(y_test_reg, y_pred_reg)
                clf_report = classification_report(y_test_clf_labels, y_pred_clf_labels, output_dict=True)
                
                # Display results
                st.subheader("Model Performance")
                
                # Cross-validation Results
                st.write("Cross-validation Results:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CV R² Score (mean)", f"{cv_scores_reg.mean():.3f}")
                    st.metric("CV R² Score (std)", f"{cv_scores_reg.std():.3f}")
                with col2:
                    st.metric("CV Accuracy (mean)", f"{cv_scores_clf.mean():.3f}")
                    st.metric("CV Accuracy (std)", f"{cv_scores_clf.std():.3f}")
                
                # Learning Curves
                st.subheader("Learning Curves")
                train_sizes = np.linspace(0.1, 1.0, 10)
                
                # Regression Learning Curve
                train_sizes_reg, train_scores_reg, test_scores_reg = learning_curve(
                    reg_model, X_scaled, y_regression,
                    train_sizes=train_sizes, cv=cv_folds, n_jobs=n_jobs,
                    scoring='r2'
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=train_scores_reg.mean(axis=1),
                    name='Training Score',
                    mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=test_scores_reg.mean(axis=1),
                    name='Cross-validation Score',
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title='Regression Learning Curve',
                    xaxis_title='Training Examples',
                    yaxis_title='R² Score'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification Learning Curve
                train_sizes_clf, train_scores_clf, test_scores_clf = learning_curve(
                    clf_model, X_scaled, y_classification,
                    train_sizes=train_sizes, cv=cv_folds, n_jobs=n_jobs,
                    scoring='accuracy'
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=train_scores_clf.mean(axis=1),
                    name='Training Score',
                    mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=test_scores_clf.mean(axis=1),
                    name='Cross-validation Score',
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title='Classification Learning Curve',
                    xaxis_title='Training Examples',
                    yaxis_title='Accuracy'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                if hasattr(reg_model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importance = reg_model.feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': importance
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(feature_importance, x='Importance', y='Feature',
                                orientation='h',
                                title='Feature Importance in Predicting Endowment')
                    st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve for Classification
                if hasattr(clf_model, 'predict_proba'):
                    st.subheader("ROC Curves")
                    y_score = clf_model.predict_proba(X_test_clf)
                    
                    fig = go.Figure()
                    for i in range(3):  # 3 classes
                        fpr, tpr, _ = roc_curve(y_test_clf == i, y_score[:, i])
                        auc_score = auc(fpr, tpr)
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            name=f'Class {label_encoder.inverse_transform([i])[0]} (AUC = {auc_score:.2f})',
                            mode='lines'
                        ))
                    
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        name='Random',
                        mode='lines',
                        line=dict(dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='ROC Curves',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        yaxis=dict(scaleanchor="x", scaleratio=1),
                        xaxis=dict(constrain='domain')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save models
                if st.button("Save Models"):
                    joblib.dump(reg_model, f'{model_type}_regressor.joblib')
                    joblib.dump(clf_model, f'{model_type}_classifier.joblib')
                    joblib.dump(scaler, 'feature_scaler.joblib')
                    joblib.dump(label_encoder, 'label_encoder.joblib')
                    st.success("Models saved successfully!")

if __name__ == "__main__":
    main()
