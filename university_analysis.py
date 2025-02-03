import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Read the dataset with encoding specification
df = pd.read_csv('NorthAmericaUniversities.csv', encoding='latin1')

# Clean the data
df['Name'] = df['Name'].str.strip()
df['Country'] = df['Country'].str.lower().str.strip()
df['Minimum Tuition cost'] = df['Minimum Tuition cost'].str.replace('"', '').str.replace('$', '').str.replace(',', '').astype(float)
df['Endowment'] = df['Endowment'].str.replace('$', '').str.replace('B', '').astype(float)

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from numerical columns
print("\nBefore cleaning:")
print(f"Total number of universities: {len(df)}")

columns_to_clean = ['Minimum Tuition cost', 'Number of Students', 'Academic Staff', 'Volumes in the library', 'Endowment']
for column in columns_to_clean:
    if df[column].notna().any():
        df = remove_outliers(df, column)

print("\nAfter removing outliers:")
print(f"Total number of universities: {len(df)}")

# Create visualizations
plt.style.use('seaborn')
fig = plt.figure(figsize=(15, 12))

# 1. Top 10 Universities by Endowment
plt.subplot(2, 2, 1)
top_10_endowment = df.nlargest(10, 'Endowment')
bars = plt.barh(top_10_endowment['Name'].str[:30], top_10_endowment['Endowment'])
plt.title('Top 10 Universities by Endowment (Billion $)')
plt.xlabel('Endowment (Billion $)')
# Add value labels on the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'${width:.1f}B', 
             ha='left', va='center', fontsize=8)

# 2. Box Plot of Tuition Costs
plt.subplot(2, 2, 2)
sns.boxplot(y=df['Minimum Tuition cost']/1000)
plt.title('Distribution of Tuition Costs\n(After Removing Outliers)')
plt.ylabel('Tuition Cost (Thousand $)')

# 3. Student Population vs. Academic Staff with Regression Line
plt.subplot(2, 2, 3)
valid_data = df.dropna(subset=['Academic Staff', 'Number of Students'])
sns.regplot(data=valid_data, x='Academic Staff', y='Number of Students', 
            scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.title('Student Population vs. Academic Staff\n(After Removing Outliers)')
plt.xlabel('Academic Staff')
plt.ylabel('Number of Students')

# 4. Universities by Country
plt.subplot(2, 2, 4)
country_counts = df['Country'].value_counts()
plt.pie(country_counts, 
        labels=[f'{idx.upper()} ({val})' for idx, val in country_counts.items()],
        autopct='%1.1f%%', 
        colors=['lightblue', 'lightgreen'])
plt.title('Universities by Country\n(After Removing Outliers)')

plt.tight_layout()
plt.savefig('university_analysis_cleaned.png', dpi=300, bbox_inches='tight')

# Print statistical insights
print("\nDataset Insights (After Cleaning):")
print(f"\nTuition Costs:")
print(f"Average: ${df['Minimum Tuition cost'].mean():,.2f}")
print(f"Median: ${df['Minimum Tuition cost'].median():,.2f}")
print(f"Standard Deviation: ${df['Minimum Tuition cost'].std():,.2f}")

print(f"\nEndowment (Billions):")
print(f"Average: ${df['Endowment'].mean():,.2f}B")
print(f"Median: ${df['Endowment'].median():,.2f}B")
print(f"Standard Deviation: ${df['Endowment'].std():,.2f}B")

print(f"\nStudent Population:")
print(f"Average: {df['Number of Students'].mean():,.0f}")
print(f"Median: {df['Number of Students'].median():,.0f}")
print(f"Standard Deviation: {df['Number of Students'].std():,.0f}")

# Calculate correlation between staff and students
correlation = df['Academic Staff'].corr(df['Number of Students'])
print(f"\nCorrelation between Academic Staff and Student Population: {correlation:.2f}")

# Age of universities
df['Age'] = 2024 - df['Established']
print(f"\nUniversity Age:")
print(f"Average Age: {df['Age'].mean():,.0f} years")
print(f"Oldest: {df.loc[df['Established'].idxmin()]['Name']} ({df['Age'].max()} years)")
print(f"Newest: {df.loc[df['Established'].idxmax()]['Name']} ({df['Age'].min()} years)")

# Print summary of removed outliers
print(f"\nOutlier Removal Summary:")
print(f"Number of universities removed: {200 - len(df)}")
print("This cleaning process removed extreme values that were statistically unusual,")
print("resulting in a more representative dataset for analysis.")
