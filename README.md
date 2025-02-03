# University Data Analysis and ML Platform

An interactive Streamlit application for analyzing university data and training machine learning models to predict university endowments and classifications.

## Features

### 1. Data Analysis
- Interactive data exploration
- Visualization of key metrics
- Correlation analysis
- University comparisons

### 2. Machine Learning Capabilities
- Multiple model types:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - AdaBoost
  - SVM
  - Linear/Logistic Regression
- Cross-validation
- Feature importance analysis
- Advanced model evaluation metrics
- Learning curves and ROC curves

### 3. Prediction Features
- Endowment prediction
- University tier classification
- Interactive model parameter tuning
- Model performance visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd university-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
python -m streamlit run university_app.py
```

2. Navigate to http://localhost:8501 in your web browser

3. Use the navigation menu to explore different features:
   - Overview: General statistics and visualizations
   - Data Explorer: Interactive data exploration
   - Endowment Analysis: Detailed endowment insights
   - University Comparison: Compare different universities
   - Predictions: Make predictions using trained models
   - ML Model Training: Train and evaluate ML models

## Project Structure

- `university_app.py`: Main Streamlit application
- `university_analysis.py`: Core analysis functions
- `university_eda.py`: Exploratory data analysis functions
- `model_evaluation.py`: Model evaluation utilities
- `hyperparameter_tuning.py`: Hyperparameter optimization
- `detailed_analysis.py`: Detailed analysis functions
- `requirements.txt`: Project dependencies

## Machine Learning Models

The platform includes various ML models for both regression (endowment prediction) and classification (tier prediction):

1. **Regression Models**
   - Predict university endowments
   - Feature importance analysis
   - Cross-validation metrics
   - Learning curves

2. **Classification Models**
   - Classify universities into tiers
   - ROC curves and AUC scores
   - Confusion matrix visualization
   - Detailed classification reports

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
