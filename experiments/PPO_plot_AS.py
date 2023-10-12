import csv
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Filepath to your CSV
filename = "results/1e-6_PPO_results.csv"
filename2 = "results/1e-4_PPO_results.csv"

# Initialize lists to store your data
ns1 = []
as1 = []
ns2 = []
as2 = []

# Open and read the CSV file
with open(filename, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Append n_steps and avg_score data
        ns1.append(row["n_steps"])
        as1.append(float(row["avg_score"]))

# Open and read the second CSV file
with open(filename2, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Append n_steps and avg_score data
        ns2.append(row["n_steps"])
        as2.append(float(row["avg_score"]))

# Plotting
plt.figure(figsize=(10, 8))

# Bar width
barWidth = 0.3

# Set position of bar on X axis
r1 = range(len(ns1))
r2 = [x + barWidth for x in r1]

# Create bars
plt.bar(r1, as1, width = barWidth, color = 'blue', edgecolor = 'grey', label='1e-6 Learning Rate')
plt.bar(r2, as2, width = barWidth, color = 'cyan', edgecolor = 'grey', label='1e-4 Learning Rate')

# Adding annotations for each bar
for i, txt in enumerate(as1):
    plt.annotate(f"{txt}", (r1[i], as1[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(as2):
    plt.annotate(f"{txt}", (r2[i], as2[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding title and labels
plt.title("Average Score for each n_steps", fontweight='bold')
plt.xlabel("n_steps", fontweight='bold')
plt.ylabel("Average Score", fontweight='bold')
plt.xticks([r + barWidth/2 for r in range(len(ns1))], ns1)  # Label x-axis indices with your n_steps values

# Adding Legend
plt.legend()

# Displaying the plot
plt.tight_layout()
plt.show()