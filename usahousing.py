"""
Housing Price Explorer + Price Suggestion Utility
BSAI-3A - Programming for AI Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
print("Project initialized with required imports of libraries")
try:
    df = pd.read_csv('usa_housing.csv')
    print("="*60)
    print("DATASET LOADED SUCCESSFULLY")
    print("="*60)
except FileNotFoundError:
    print("ERROR: usahousing.csv not found in current directory")
    print("Creating sample data...")
    data = {
        'Price': [221958, 771155, 231932, 465838, 359178],
        'Bedrooms': [1, 2, 1, 3, 4],
        'Bathrooms': [1.9, 2.0, 3.0, 3.3, 3.4],
        'SquareFeet': [4827, 1035, 2769, 2708, 1175],
        'YearBuilt': [1979, 1987, 1982, 1907, 1994],
        'GarageSpaces': [2, 2, 1, 3, 2],
        'LotSize': [1.45, 1.75, 1.46, 1.62, 0.74],
        'ZipCode': [82240, 74315, 79249, 80587, 20756],
        'CrimeRate': [48.6, 92.03, 52.08, 61.65, 15.66],
        'SchoolRating': [5, 9, 3, 1, 4]
    }
    df = pd.DataFrame(data)

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())
print("\n" + "="*60)
df_clean = df.copy()
print("DATA CLEANING")
print("="*60)

df_clean['HouseAge'] = 2024 - df_clean['YearBuilt']
df_clean['PricePerSqFt'] = df_clean['Price'] / df_clean['SquareFeet']
df_clean['SizeCategory'] = pd.cut(df_clean['SquareFeet'], 
                                 bins=[0, 1500, 3000, 4500, 6000], 
                                 labels=['Small', 'Medium', 'Large', 'Extra Large'])

print("Basic cleaning and feature engineering completed")
print("\nBASIC STATISTICS:")
print(df_clean[['Price', 'SquareFeet', 'Bedrooms', 'Bathrooms']].describe().round(2))

numeric_cols = ['Price', 'Bedrooms', 'Bathrooms', 'SquareFeet', 'YearBuilt', 'CrimeRate', 'SchoolRating']
correlation = df_clean[numeric_cols].corr()

print("\nCORRELATION WITH PRICE:")
price_corr = correlation['Price'].sort_values(ascending=False)
for col, value in price_corr.items():
    print(f"  {col}: {value:.3f}")

print("\n" + "="*60)

print("DATA VISUALIZATION DASHBOARD")
print("="*60)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('HOUSING PRICE ANALYSIS DASHBOARD', fontsize=20, fontweight='bold', y=1.02)

# Plot 1: Price Distribution
axes[0, 0].hist(df_clean['Price'], bins=25, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].set_title('Price Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Price vs Square Feet
axes[0, 1].scatter(df_clean['SquareFeet'], df_clean['Price'], alpha=0.6, s=20, color='green')
axes[0, 1].set_title('Price vs Square Feet', fontweight='bold')
axes[0, 1].set_xlabel('Square Feet')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Average Price by Bedrooms
bedroom_avg = df_clean.groupby('Bedrooms')['Price'].mean()
axes[0, 2].bar(bedroom_avg.index, bedroom_avg.values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
axes[0, 2].set_title('Average Price by Bedrooms', fontweight='bold')
axes[0, 2].set_xlabel('Bedrooms')
axes[0, 2].set_ylabel('Avg Price ($)')
axes[0, 2].grid(True, alpha=0.3) 

# Plot 4: Price by School Rating
school_avg = df_clean.groupby('SchoolRating')['Price'].mean()
axes[1, 0].bar(school_avg.index, school_avg.values, color='salmon')
axes[1, 0].set_title('Price by School Rating', fontweight='bold')
axes[1, 0].set_xlabel('School Rating')
axes[1, 0].set_ylabel('Avg Price ($)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Price vs Crime Rate
axes[1, 1].scatter(df_clean['CrimeRate'], df_clean['Price'], alpha=0.6, s=20, color='red')
axes[1, 1].set_title('Price vs Crime Rate', fontweight='bold')
axes[1, 1].set_xlabel('Crime Rate')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Price by Size Category
size_avg = df_clean.groupby('SizeCategory')['Price'].mean()
axes[1, 2].bar(size_avg.index.astype(str), size_avg.values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
axes[1, 2].set_title('Price by Size Category', fontweight='bold')
axes[1, 2].set_xlabel('Size Category')
axes[1, 2].set_ylabel('Avg Price ($)')
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].grid(True, alpha=0.3)

# Plot 7: Correlation Heatmap
sns.heatmap(correlation, ax=axes[2, 0], cmap='coolwarm', annot=True, center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
axes[2, 0].set_title('Correlation Heatmap', fontweight='bold')

# Plot 8: Price per SqFt by Bedrooms
price_per_sqft_stats = df_clean.groupby('Bedrooms')['PricePerSqFt'].mean()
axes[2, 1].bar(price_per_sqft_stats.index, price_per_sqft_stats.values, color='teal')
axes[2, 1].set_title('Price per SqFt by Bedrooms', fontweight='bold')
axes[2, 1].set_xlabel('Bedrooms')
axes[2, 1].set_ylabel('Price per SqFt ($)')
axes[2, 1].grid(True, alpha=0.3)

# Plot 9: Price Trend by Year Built
year_built_avg = df_clean.groupby('YearBuilt')['Price'].mean()
axes[2, 2].plot(year_built_avg.index, year_built_avg.values, marker='o', linestyle='-', color='purple', alpha=0.7)
axes[2, 2].set_title('Price Trend by Year Built', fontweight='bold')
axes[2, 2].set_xlabel('Year Built')
axes[2, 2].set_ylabel('Avg Price ($)')
axes[2, 2].grid(True, alpha=0.3)