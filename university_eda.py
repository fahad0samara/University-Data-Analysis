import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set the style for all plots
plt.style.use('seaborn')
sns.set_palette("husl")

# Read and clean the data
df = pd.read_csv('NorthAmericaUniversities.csv', encoding='latin1')
df['Name'] = df['Name'].str.strip()
df['Country'] = df['Country'].str.lower().str.strip()
df['Minimum Tuition cost'] = df['Minimum Tuition cost'].str.replace('"', '').str.replace('$', '').str.replace(',', '').astype(float)
df['Endowment'] = df['Endowment'].str.replace('$', '').str.replace('B', '').astype(float)
df['Age'] = 2024 - df['Established']

# Create a figure for the first set of visualizations
plt.figure(figsize=(20, 15))

# 1. Distribution of University Ages
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Distribution of University Ages')
plt.xlabel('Age (years)')
plt.ylabel('Count')

# 2. Tuition vs Endowment
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='Minimum Tuition cost', y='Endowment', alpha=0.6)
plt.title('Tuition Cost vs Endowment')
plt.xlabel('Minimum Tuition Cost ($)')
plt.ylabel('Endowment (Billion $)')

# 3. Student-to-Staff Ratio Analysis
df['Student_Staff_Ratio'] = df['Number of Students'] / df['Academic Staff']
plt.subplot(2, 2, 3)
sns.boxplot(data=df, y='Student_Staff_Ratio')
plt.title('Distribution of Student-to-Staff Ratio')
plt.ylabel('Students per Staff Member')

# 4. Library Resources vs Student Population
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Number of Students', y='Volumes in the library', alpha=0.6)
plt.title('Library Resources vs Student Population')
plt.xlabel('Number of Students')
plt.ylabel('Volumes in Library')

plt.tight_layout()
plt.savefig('eda_analysis_1.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a second figure for additional visualizations
plt.figure(figsize=(20, 15))

# 5. Age vs Endowment
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='Age', y='Endowment', alpha=0.6)
plt.title('University Age vs Endowment')
plt.xlabel('Age (years)')
plt.ylabel('Endowment (Billion $)')

# 6. Tuition Cost Distribution by Country
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Country', y='Minimum Tuition cost')
plt.title('Tuition Cost Distribution by Country')
plt.xlabel('Country')
plt.ylabel('Minimum Tuition Cost ($)')

# 7. Student Population Over Time
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='Established', y='Number of Students', alpha=0.6)
plt.title('Student Population vs University Establishment Year')
plt.xlabel('Year Established')
plt.ylabel('Number of Students')

# 8. Library Resources vs Endowment
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Endowment', y='Volumes in the library', alpha=0.6)
plt.title('Library Resources vs Endowment')
plt.xlabel('Endowment (Billion $)')
plt.ylabel('Volumes in Library')

plt.tight_layout()
plt.savefig('eda_analysis_2.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and print statistical summaries
print("\nExploratory Data Analysis Summary:")
print("\n1. Basic Statistics:")
print(df.describe().round(2))

print("\n2. Correlation Analysis:")
numeric_columns = ['Age', 'Academic Staff', 'Number of Students', 
                  'Minimum Tuition cost', 'Volumes in the library', 'Endowment']
correlation_matrix = df[numeric_columns].corr().round(2)
print("\nCorrelation Matrix:")
print(correlation_matrix)

print("\n3. University Age Distribution:")
print(f"Mean Age: {df['Age'].mean():.1f} years")
print(f"Median Age: {df['Age'].median():.1f} years")
print(f"Age Range: {df['Age'].min():.0f} to {df['Age'].max():.0f} years")

print("\n4. Student-to-Staff Ratio Analysis:")
print(f"Average Student-to-Staff Ratio: {df['Student_Staff_Ratio'].mean():.1f}")
print(f"Median Student-to-Staff Ratio: {df['Student_Staff_Ratio'].median():.1f}")
print(f"Range: {df['Student_Staff_Ratio'].min():.1f} to {df['Student_Staff_Ratio'].max():.1f}")

print("\n5. Country-wise Analysis:")
for country in df['Country'].unique():
    country_data = df[df['Country'] == country]
    print(f"\n{country.upper()} Universities:")
    print(f"Count: {len(country_data)}")
    print(f"Average Tuition: ${country_data['Minimum Tuition cost'].mean():,.2f}")
    print(f"Average Endowment: ${country_data['Endowment'].mean():.2f}B")
    print(f"Average Student Population: {country_data['Number of Students'].mean():,.0f}")

# Additional insights
print("\n6. Key Insights:")
print("- Correlation between resources and size:")
print(f"  * Endowment vs Library Volumes: {df['Endowment'].corr(df['Volumes in the library']):.2f}")
print(f"  * Students vs Library Volumes: {df['Number of Students'].corr(df['Volumes in the library']):.2f}")
print(f"  * Endowment vs Student Population: {df['Endowment'].corr(df['Number of Students']):.2f}")

# Save summary statistics to a file
with open('eda_summary.txt', 'w') as f:
    f.write("Exploratory Data Analysis Summary\n")
    f.write("================================\n\n")
    f.write("1. Basic Statistics:\n")
    f.write(df.describe().round(2).to_string())
    f.write("\n\n2. Correlation Matrix:\n")
    f.write(correlation_matrix.to_string())
    f.write("\n\nFile generated by university_eda.py")
