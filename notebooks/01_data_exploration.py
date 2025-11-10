"""
# Phishing Brand Classifier - Data Exploration

This notebook explores the phishing detection dataset and performs
comprehensive data analysis.

To run this as a notebook, use jupytext:
    jupytext --to notebook 01_data_exploration.py
    jupyter notebook 01_data_exploration.ipynb
"""

# %%
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# %%
# ## 1. Dataset Structure

data_dir = Path('../data/raw')

# Get all brand directories
brands = [d.name for d in data_dir.iterdir() if d.is_dir()]
brands.sort()

print("Dataset Structure")
print("="*80)
print(f"Total brands: {len(brands)}\n")

# Count images per brand
brand_counts = {}
for brand in brands:
    brand_dir = data_dir / brand
    image_files = list(brand_dir.glob('*.png')) + list(brand_dir.glob('*.jpg'))
    brand_counts[brand] = len(image_files)
    print(f"{brand:15s}: {brand_counts[brand]:5d} images")

print(f"\nTotal images: {sum(brand_counts.values())}")

# %%
# ## 2. Class Distribution Analysis

# Create DataFrame for analysis
df = pd.DataFrame({
    'Brand': list(brand_counts.keys()),
    'Count': list(brand_counts.values())
})

df = df.sort_values('Count', ascending=False)

# Plot class distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
colors = sns.color_palette("husl", len(df))
bars = ax1.bar(df['Brand'], df['Count'], color=colors, alpha=0.8, edgecolor='black')
ax1.set_xlabel('Brand', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Pie chart
ax2.pie(df['Count'], labels=df['Brand'], autopct='%1.1f%%',
        colors=colors, startangle=90)
ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# ## 3. Class Imbalance Analysis

print("\nClass Imbalance Analysis")
print("="*80)

total = df['Count'].sum()
df['Percentage'] = (df['Count'] / total) * 100
df['Ratio_to_Max'] = df['Count'] / df['Count'].max()
df['Ratio_to_Min'] = df['Count'] / df['Count'].min()

print(df.to_string(index=False))

print(f"\nImbalance Ratio (max/min): {df['Count'].max() / df['Count'].min():.2f}")
print(f"Mean samples per class: {df['Count'].mean():.1f}")
print(f"Std samples per class: {df['Count'].std():.1f}")
print(f"Coefficient of Variation: {(df['Count'].std() / df['Count'].mean()):.2f}")

# %%
# ## 4. Sample Images Visualization

def show_sample_images(brand, num_samples=6):
    """Display sample images from a brand."""
    brand_dir = data_dir / brand
    image_files = list(brand_dir.glob('*.png')) + list(brand_dir.glob('*.jpg'))

    if len(image_files) == 0:
        print(f"No images found for {brand}")
        return

    # Select random samples
    samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, img_path in enumerate(samples):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f'{img_path.stem}\n{img.size[0]}x{img.size[1]}',
                         fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Sample Images: {brand}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Show samples for each brand (or a subset)
for brand in brands[:3]:  # Show first 3 brands
    print(f"\n### {brand.upper()}")
    show_sample_images(brand, num_samples=6)

# %%
# ## 5. Image Size Analysis

print("\nImage Size Analysis")
print("="*80)

widths = []
heights = []
aspects = []
brand_labels = []

for brand in brands:
    brand_dir = data_dir / brand
    image_files = list(brand_dir.glob('*.png')) + list(brand_dir.glob('*.jpg'))

    # Sample up to 50 images per brand to speed up analysis
    sampled_files = np.random.choice(
        image_files,
        min(50, len(image_files)),
        replace=False
    )

    for img_path in sampled_files:
        try:
            img = Image.open(img_path)
            widths.append(img.size[0])
            heights.append(img.size[1])
            aspects.append(img.size[0] / img.size[1])
            brand_labels.append(brand)
        except:
            pass

# Statistics
print(f"\nSample size: {len(widths)} images\n")
print(f"Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}, std={np.std(widths):.1f}")
print(f"Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}, std={np.std(heights):.1f}")
print(f"Aspect: min={min(aspects):.2f}, max={max(aspects):.2f}, mean={np.mean(aspects):.2f}, std={np.std(aspects):.2f}")

# %%
# Visualize image size distributions

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Width distribution
axes[0, 0].hist(widths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 0].set_xlabel('Width (pixels)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Image Width Distribution', fontweight='bold')
axes[0, 0].legend()

# Height distribution
axes[0, 1].hist(heights, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 1].set_xlabel('Height (pixels)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Image Height Distribution', fontweight='bold')
axes[0, 1].legend()

# Aspect ratio distribution
axes[1, 0].hist(aspects, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(np.mean(aspects), color='red', linestyle='--', linewidth=2, label='Mean')
axes[1, 0].set_xlabel('Aspect Ratio (W/H)', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Aspect Ratio Distribution', fontweight='bold')
axes[1, 0].legend()

# Scatter: width vs height
axes[1, 1].scatter(widths, heights, alpha=0.5, s=10)
axes[1, 1].set_xlabel('Width (pixels)', fontweight='bold')
axes[1, 1].set_ylabel('Height (pixels)', fontweight='bold')
axes[1, 1].set_title('Width vs Height', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# ## 6. Recommended Data Strategy

print("\n" + "="*80)
print("RECOMMENDED DATA STRATEGY")
print("="*80)

print("\n1. CLASS IMBALANCE HANDLING:")
print("   - Use weighted loss functions (Focal Loss)")
print("   - Apply class weights based on inverse frequency")
print("   - Consider oversampling minority classes or undersampling majority")

print("\n2. DATA AUGMENTATION:")
print("   - Geometric: HorizontalFlip, ShiftScaleRotate, RandomCrop")
print("   - Color: ColorJitter (subtle to preserve brand colors)")
print("   - Quality: GaussianBlur, GaussNoise, Compression")
print("   - Important: Keep augmentations subtle to preserve brand identity")

print("\n3. TRAIN/VAL/TEST SPLIT:")
train_size = int(total * 0.7)
val_size = int(total * 0.15)
test_size = total - train_size - val_size
print(f"   - Train: 70% ({train_size} samples)")
print(f"   - Val:   15% ({val_size} samples)")
print(f"   - Test:  15% ({test_size} samples)")
print("   - Use stratified splitting to maintain class distribution")

print("\n4. IMAGE PREPROCESSING:")
print("   - Resize to: 224x224 or 384x384 (depending on model)")
print("   - Normalize with ImageNet statistics")
print("   - Convert to RGB if needed")

print("\n5. FALSE POSITIVE REDUCTION:")
print("   - Critical: Minimize 'others' misclassified as brands")
print("   - Strategies:")
print("     * Higher confidence threshold for brand predictions")
print("     * Ensemble methods")
print("     * Calibration techniques")
print("     * Cost-sensitive learning with higher penalty for FP")

# %%
# ## 7. Key Insights and Next Steps

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

insights = f"""
1. DATASET SIZE: {total} total images across {len(brands)} brands

2. CLASS DISTRIBUTION:
   - Most common: {df.iloc[0]['Brand']} ({df.iloc[0]['Count']} samples)
   - Least common: {df.iloc[-1]['Brand']} ({df.iloc[-1]['Count']} samples)
   - Imbalance ratio: {df['Count'].max() / df['Count'].min():.2f}:1

3. IMAGE CHARACTERISTICS:
   - Average size: {np.mean(widths):.0f} x {np.mean(heights):.0f} pixels
   - Aspect ratio: {np.mean(aspects):.2f} (mostly landscape)
   - Size variation: High (requires consistent preprocessing)

4. MODELING RECOMMENDATIONS:
   - Architecture: EfficientNet-B3 or ResNet50 (good balance)
   - Loss: Focal Loss with class weights
   - Metric focus: F1-score, FPR for 'others' class
   - Training: Transfer learning with ImageNet weights

5. SUCCESS CRITERIA:
   - High overall accuracy (>95%)
   - Low FPR for 'others' class (<1%)
   - Fast inference (<100ms per image)
   - Robust to image quality variations
"""

print(insights)

# %%
print("\n✅ Data exploration complete!")
print("Next step: Train the model using scripts/train.py")
