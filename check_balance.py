import pandas as pd
import matplotlib.pyplot as plt

def check_balance(csv_path, set_name):
    df = pd.read_csv(csv_path)

    if 'label' not in df.columns:
        print(f"[{set_name}] has no label column.")
        return

    counts = df['label'].value_counts().sort_index()
    total = counts.sum()
    imbalance_ratio = counts.max() / counts.min()

    print(f"\n=== {set_name.upper()} ===")
    print(counts)
    print(f"Total: {total}")
    print(f"Raport dezechilibru (max/min): {imbalance_ratio:.2f}")

    counts.plot(kind='bar', title=f"Distribution - {set_name}")
    plt.xlabel("Clasă")
    plt.ylabel("Număr de imagini")
    plt.grid(True)
    plt.show()

check_balance("train.csv", "train")
check_balance("validation.csv", "validation")
check_balance("test.csv", "test")  
