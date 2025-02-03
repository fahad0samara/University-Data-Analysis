import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 10

# Read and clean the data
df = pd.read_csv('NorthAmericaUniversities.csv', encoding='latin1')
df['Name'] = df['Name'].str.strip()
df['Country'] = df['Country'].str.lower().str.strip()
df['Minimum Tuition cost'] = df['Minimum Tuition cost'].str.replace('"', '').str.replace('$', '').str.replace(',', '').astype(float)
df['Endowment'] = df['Endowment'].str.replace('$', '').str.replace('B', '').astype(float)
df['Age'] = 2024 - df['Established']
df['Student_Staff_Ratio'] = df['Number of Students'] / df['Academic Staff']

# 1. Time Period Analysis
plt.figure(figsize=(15, 10))
establishment_periods = pd.cut(df['Established'], 
                             bins=[1600, 1700, 1800, 1850, 1900, 1950, 2000],
                             labels=['1600s', '1700s', '1800-1850', '1850-1900', '1900-1950', '1950-2000'])
period_stats = df.groupby(establishment_periods).agg({
    'Endowment': 'mean',
    'Minimum Tuition cost': 'mean',
    'Number of Students': 'mean'
}).round(2)

period_stats.plot(kind='bar', subplots=True, layout=(3,1), figsize=(15, 12))
plt.tight_layout()
plt.savefig('time_period_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Resource Distribution Analysis
plt.figure(figsize=(15, 12))
# Create size categories
df['Size_Category'] = pd.qcut(df['Number of Students'], q=4, 
                             labels=['Small', 'Medium', 'Large', 'Very Large'])

# Plot average resources by size
plt.subplot(2,2,1)
sns.boxplot(data=df, x='Size_Category', y='Endowment')
plt.title('Endowment by University Size')
plt.xticks(rotation=45)

plt.subplot(2,2,2)
sns.boxplot(data=df, x='Size_Category', y='Minimum Tuition cost')
plt.title('Tuition Cost by University Size')
plt.xticks(rotation=45)

plt.subplot(2,2,3)
sns.boxplot(data=df, x='Size_Category', y='Volumes in the library')
plt.title('Library Resources by University Size')
plt.xticks(rotation=45)

plt.subplot(2,2,4)
sns.boxplot(data=df, x='Size_Category', y='Student_Staff_Ratio')
plt.title('Student-Staff Ratio by University Size')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('size_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Geographical and Age Analysis
plt.figure(figsize=(15, 12))

plt.subplot(2,2,1)
sns.violinplot(data=df, x='Country', y='Age')
plt.title('Age Distribution by Country')

plt.subplot(2,2,2)
sns.violinplot(data=df, x='Country', y='Student_Staff_Ratio')
plt.title('Student-Staff Ratio by Country')

plt.subplot(2,2,3)
sns.violinplot(data=df, x='Country', y='Endowment')
plt.title('Endowment Distribution by Country')

plt.subplot(2,2,4)
sns.violinplot(data=df, x='Country', y='Minimum Tuition cost')
plt.title('Tuition Cost Distribution by Country')

plt.tight_layout()
plt.savefig('country_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(12, 10))
numeric_columns = ['Age', 'Academic Staff', 'Number of Students', 
                  'Minimum Tuition cost', 'Volumes in the library', 
                  'Endowment', 'Student_Staff_Ratio']
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of University Metrics')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Print detailed statistical analysis
print("\nDetailed Statistical Analysis")
print("============================")

# 1. Time Period Analysis
print("\n1. University Characteristics by Establishment Period:")
print(period_stats)

# 2. Size Category Analysis
print("\n2. University Metrics by Size Category:")
size_metrics = df.groupby('Size_Category').agg({
    'Number of Students': 'mean',
    'Endowment': 'mean',
    'Minimum Tuition cost': 'mean',
    'Student_Staff_Ratio': 'mean'
}).round(2)
print(size_metrics)

# 3. Country Comparison
print("\n3. Detailed Country Comparison:")
country_metrics = df.groupby('Country').agg({
    'Age': ['mean', 'min', 'max'],
    'Endowment': ['mean', 'min', 'max'],
    'Minimum Tuition cost': ['mean', 'min', 'max'],
    'Student_Staff_Ratio': ['mean', 'min', 'max']
}).round(2)
print(country_metrics)

# 4. Top Universities in Different Categories
print("\n4. Notable Universities:")
print("\nTop 5 by Endowment:")
print(df.nlargest(5, 'Endowment')[['Name', 'Endowment', 'Country']])

print("\nTop 5 by Student Population:")
print(df.nlargest(5, 'Number of Students')[['Name', 'Number of Students', 'Country']])

print("\nTop 5 by Library Resources:")
print(df.nlargest(5, 'Volumes in the library')[['Name', 'Volumes in the library', 'Country']])

# Save detailed statistics to file
with open('detailed_analysis.txt', 'w') as f:
    f.write("Detailed University Analysis\n")
    f.write("===========================\n\n")
    f.write("1. Time Period Analysis:\n")
    f.write(period_stats.to_string())
    f.write("\n\n2. Size Category Analysis:\n")
    f.write(size_metrics.to_string())
    f.write("\n\n3. Country Comparison:\n")
    f.write(country_metrics.to_string())
    f.write("\n\nAnalysis generated by detailed_analysis.py")
