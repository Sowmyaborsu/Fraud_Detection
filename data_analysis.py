import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Transaction counts
print("Transaction Distribution:")
print(df["Class"].value_counts())

# Create labels for the graph
transaction_counts = df["Class"].value_counts()
transaction_counts.index = ["Normal", "Fraud"]

# Plot graph
plt.figure(figsize=(8, 5))

sns.barplot(
    x=transaction_counts.index,
    y=transaction_counts.values
)

plt.title("Normal vs Fraud Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()