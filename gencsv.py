import pandas as pd
import numpy as np

# Citește CSV-ul original
df = pd.read_csv('train.csv')

print(f"Dataset original: {len(df)} imagini")
print("Distribuția labelurilor:")
print(df['label'].value_counts().sort_index())

# Separă imaginile cu label 4 de restul
label_4_images = df[df['label'] == 4].copy()
other_labels_images = df[df['label'] != 4].copy()

print(f"\nImagini cu label 4: {len(label_4_images)}")
print(f"Imagini cu alte labeluri: {len(other_labels_images)}")

# Selectează imaginile pentru dataset-ul echilibrat
np.random.seed(42)  # Pentru reproducibilitate

# Pentru label 4 → label 1 (2500 imagini)
selected_label_4 = label_4_images.sample(n=2500, random_state=42).copy()
selected_label_4['label'] = 1

# Pentru alte labeluri → label 0 (700 din fiecare label: 0, 2, 3)
selected_others_list = []

for label in [0, 1, 2, 3]:
    label_images = other_labels_images[other_labels_images['label'] == label]
    print(f"Imagini disponibile cu label {label}: {len(label_images)}")
    
    if len(label_images) >= 725:
        selected = label_images.sample(n=725, random_state=42).copy()
    else:
        print(f"ATENȚIE: Doar {len(label_images)} imagini cu label {label}, se iau toate")
        selected = label_images.copy()
    
    selected['label'] = 0
    selected_others_list.append(selected)

selected_others = pd.concat(selected_others_list, ignore_index=True)

# Combină cele două seturi
balanced_df = pd.concat([selected_label_4, selected_others], ignore_index=True)

# Amestecă datele
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDataset echilibrat: {len(balanced_df)} imagini")
print("Distribuția finală:")
print(balanced_df['label'].value_counts().sort_index())
print(f"Label 0: {len(balanced_df[balanced_df['label'] == 0])} imagini")
print(f"Label 1: {len(balanced_df[balanced_df['label'] == 1])} imagini")

# Salvează noul CSV
balanced_df.to_csv('train_balanced.csv', index=False)
print(f"\nNoul dataset a fost salvat ca 'train_balanced.csv'")

# Afișează primele câteva rânduri pentru verificare
print(f"\nPrimele 10 rânduri din noul dataset:")
print(balanced_df.head(10))