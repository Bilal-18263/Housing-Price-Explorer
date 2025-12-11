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