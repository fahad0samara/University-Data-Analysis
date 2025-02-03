import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Read and clean the data
df = pd.read_csv('NorthAmericaUniversities.csv', encoding='latin1')

# Clean and prepare the data
df['Name'] = df['Name'].str.strip()
df['Country'] = df['Country'].str.lower().str.strip()
df['Minimum Tuition cost'] = df['Minimum Tuition cost'].str.replace('"', '').str.replace('$', '').str.replace(',', '').astype(float)
df['Endowment'] = df['Endowment'].str.replace('$', '').str.replace('B', '').astype(float)
df['Age'] = 2024 - df['Established']
df['Student_Staff_Ratio'] = df['Number of Students'] / df['Academic Staff']

# Print initial data info
print("Initial data shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Handle missing values
df = df.dropna(subset=['Endowment'])  # Remove rows where endowment is missing
df = df.fillna({
    'Academic Staff': df['Academic Staff'].mean(),
    'Number of Students': df['Number of Students'].mean(),
    'Volumes in the library': df['Volumes in the library'].mean(),
    'Student_Staff_Ratio': df['Student_Staff_Ratio'].mean()
})

print("\nData shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Create feature matrix
features = ['Age', 'Academic Staff', 'Number of Students', 'Volumes in the library', 'Student_Staff_Ratio']
X = df[features].copy()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\n1. Predicting University Endowment")
print("=================================")

# Prepare target variable for endowment prediction
y_endowment = df['Endowment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_endowment, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nRandom Forest Regression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
})
print("\nFeature Importance for Endowment Prediction:")
print(feature_importance.sort_values('Importance', ascending=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.sort_values('Importance'))
plt.title('Feature Importance for Predicting University Endowment')
plt.tight_layout()
plt.savefig('endowment_feature_importance.png')
plt.close()

print("\n2. University Clustering Analysis")
print("================================")

# Perform K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_stats = df.groupby('Cluster').agg({
    'Endowment': 'mean',
    'Minimum Tuition cost': 'mean',
    'Number of Students': 'mean',
    'Age': 'mean',
    'Student_Staff_Ratio': 'mean'
}).round(2)

print("\nCluster Characteristics:")
print(cluster_stats)

# Visualize clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Endowment'], df['Number of Students'], 
                     c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Endowment (Billion $)')
plt.ylabel('Number of Students')
plt.title('University Clusters based on Multiple Features')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('university_clusters.png')
plt.close()

print("\n3. Predicting University Tier")
print("============================")

# Create university tiers based on endowment
df['Tier'] = pd.qcut(df['Endowment'], q=3, labels=['Lower', 'Middle', 'Upper'])

# Prepare data for classification
y_tier = df['Tier']
le = LabelEncoder()
y_tier_encoded = le.fit_transform(y_tier)

# Split data for classification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_tier_encoded, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Print classification results
print("\nRandom Forest Classification Results:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Lower', 'Middle', 'Upper']))

# Feature importance for classification
feature_importance_clf = pd.DataFrame({
    'Feature': features,
    'Importance': rf_classifier.feature_importances_
})
print("\nFeature Importance for Tier Classification:")
print(feature_importance_clf.sort_values('Importance', ascending=False))

print("\n4. Making Predictions for New Universities")
print("========================================")

# Function to predict endowment and tier for a new university
def predict_university_metrics(age, staff, students, library_volumes):
    # Prepare input data
    new_data = pd.DataFrame({
        'Age': [age],
        'Academic Staff': [staff],
        'Number of Students': [students],
        'Volumes in the library': [library_volumes],
        'Student_Staff_Ratio': [students/staff]
    })
    
    # Scale the input
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    endowment_pred = rf_model.predict(new_data_scaled)[0]
    tier_pred = le.inverse_transform(rf_classifier.predict(new_data_scaled))[0]
    
    return endowment_pred, tier_pred

# Example predictions
print("\nExample Predictions for New Universities:")
test_cases = [
    (50, 1000, 20000, 1000000),  # Young, medium-sized university
    (150, 2000, 35000, 5000000), # Established, large university
    (200, 3000, 50000, 10000000) # Old, very large university
]

for age, staff, students, volumes in test_cases:
    endowment_pred, tier_pred = predict_university_metrics(age, staff, students, volumes)
    print(f"\nUniversity Profile:")
    print(f"Age: {age} years")
    print(f"Academic Staff: {staff}")
    print(f"Students: {students}")
    print(f"Library Volumes: {volumes}")
    print(f"Predicted Endowment: ${endowment_pred:.2f}B")
    print(f"Predicted Tier: {tier_pred}")

# Save all results to a file
with open('ml_analysis_results.txt', 'w') as f:
    f.write("Machine Learning Analysis Results\n")
    f.write("===============================\n\n")
    f.write("1. Endowment Prediction Model\n")
    f.write(f"R² Score: {r2:.3f}\n")
    f.write("\nFeature Importance:\n")
    f.write(feature_importance.to_string())
    f.write("\n\n2. Cluster Analysis\n")
    f.write("\nCluster Characteristics:\n")
    f.write(cluster_stats.to_string())
    f.write("\n\n3. Classification Results\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Lower', 'Middle', 'Upper']))

print("\nAnalysis complete! Check the following files for results:")
print("1. endowment_feature_importance.png - Feature importance visualization")
print("2. university_clusters.png - Cluster visualization")
print("3. ml_analysis_results.txt - Detailed analysis results")
