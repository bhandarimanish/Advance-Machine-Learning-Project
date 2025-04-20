import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# 1. Define Ablation Results
# -------------------------
results = {
    "Loss Function": ["BCE Only", "Dice Only", "BCE + Dice"],
    "Final Loss": [0.8102, 0.9731, 0.6990],
    "Avg MSE": [0.9788, 0.7786, 0.0101],
    "Avg Dice": [0.0153, 0.0251, 0.0020]
}

df = pd.DataFrame(results)

# -------------------------
# 2. Save Table as CSV (optional)
# -------------------------
df.to_csv("ablation_results/ablation_summary.csv", index=False)

# -------------------------
# 3. Save Table as Image
# -------------------------
def save_table_as_image(df, filename):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

save_table_as_image(df, "ablation_results/ablation_table.png")

# -------------------------
# 4. Plot Bar Charts
# -------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# MSE
axs[0].bar(df["Loss Function"], df["Avg MSE"], color=["skyblue", "lightcoral", "lightgreen"])
axs[0].set_title("Average MSE per Loss Function")
axs[0].set_ylabel("MSE")
axs[0].set_ylim(0, 1.05)

# Dice
axs[1].bar(df["Loss Function"], df["Avg Dice"], color=["skyblue", "lightcoral", "lightgreen"])
axs[1].set_title("Average Dice Coefficient per Loss Function")
axs[1].set_ylabel("Dice Coefficient")
axs[1].set_ylim(0, 0.03)

plt.tight_layout()
plt.savefig("ablation_results/ablation_bar_charts.png", dpi=300)
plt.show()
