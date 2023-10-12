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
plt.figure(figsize=(10, 6))
line1, = plt.plot(n_steps1, highest_scores1, marker='o', label='1e-6 Learning Rate')
line2, = plt.plot(n_steps2, highest_scores2, marker='o', label='1e-4 Learning Rate')

# Adding annotations for each point
for i, txt in enumerate(highest_scores1):
    plt.annotate(f"{txt}", (n_steps1[i], highest_scores1[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(highest_scores2):
    plt.annotate(f"{txt}", (n_steps2[i], highest_scores2[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding title and labels
plt.title("Highest Score for Each n_steps")
plt.xlabel("n_steps")
plt.ylabel("Highest Score")

# Adding Legend
plt.legend(handles=[line1, line2])

# Displaying the plot
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
