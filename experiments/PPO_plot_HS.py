import csv
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Filepath to your CSV
filename = "results/1e-6_PPO_results.csv"
filename2 = "results/1e-4_PPO_results.csv"

# Initialize lists to store your data
n_steps1 = []
highest_scores1 = []
n_steps2 = []
highest_scores2 = []

# Open and read the CSV file
with open(filename, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Append n_steps and highest_score data
        n_steps1.append(row["n_steps"])
        highest_scores1.append(float(row["highest_score"]))

# Open and read the second CSV file
with open(filename2, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Append n_steps and highest_score data
        n_steps2.append(row["n_steps"])
        highest_scores2.append(float(row["highest_score"]))

# Plotting
plt.figure(figsize=(10, 8))

# Bar width
barWidth = 0.3

# Set position of bar on X axis
r1 = range(len(n_steps1))
r2 = [x + barWidth for x in r1]

# Create bars
# plt.bar(r1, highest_scores1, width = barWidth, color = 'blue', edgecolor = 'grey', label='1e-6 Learning Rate')
# plt.bar(r2, highest_scores2, width = barWidth, color = 'cyan', edgecolor = 'grey', label='1e-4 Learning Rate')
plt.bar(r1, highest_scores1, width = barWidth, edgecolor = 'grey', label='1e-6 Learning Rate')
plt.bar(r2, highest_scores2, width = barWidth, edgecolor = 'grey', label='1e-4 Learning Rate')

# Adding annotations for each bar
for i, txt in enumerate(highest_scores1):
    plt.annotate(f"{txt}", (r1[i], highest_scores1[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(highest_scores2):
    plt.annotate(f"{txt}", (r2[i], highest_scores2[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding title and labels
plt.title("Highest Score for each time steps", fontweight='bold')
plt.xlabel("Steps", fontweight='bold')
plt.ylabel("Highest Score", fontweight='bold')
plt.xticks([r + barWidth/2 for r in range(len(n_steps1))], n_steps1)  # Label x-axis indices with your n_steps values

# Adding Legend
plt.legend()

# Displaying the plot
plt.tight_layout()
plt.show()