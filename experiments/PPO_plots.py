# import csv
# import matplotlib.pyplot as plt

# # Filepath to your CSV
# filename = "1e-6_PPO_results.csv"

# # Initialize lists to store your data
# n_steps = []
# highest_scores = []

# # Open and read the CSV file
# with open(filename, mode='r', newline='') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         # Append n_steps and highest_score data
#         n_steps.append(row["n_steps"])
#         highest_scores.append(float(row["highest_score"]))

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(n_steps, highest_scores, marker='o')

# # Adding title and labels
# plt.title("Highest Score for Each n_steps")
# plt.xlabel("n_steps")
# plt.ylabel("Highest Score")

# # Displaying the plot
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
import csv
import matplotlib.pyplot as plt

# Filepath to your CSV
filename = "1e-4_PPO_results.csv"

# Initialize lists to store your data
n_steps = []
highest_scores = []
highest_rewards = []
avg_rewards = []
avg_scores = []

# Open and read the CSV file
with open(filename, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Append data
        n_steps.append(row["n_steps"])
        highest_scores.append(float(row["highest_score"]))
        highest_rewards.append(float(row["highest_reward"]))
        avg_rewards.append(float(row["avg_reward"]))
        avg_scores.append(float(row["avg_score"]))

# Creating a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plotting each data in a subplot
axs[0, 0].plot(n_steps, highest_scores, marker='o')
axs[0, 1].plot(n_steps, highest_rewards, marker='o', color='green')
axs[1, 0].plot(n_steps, avg_rewards, marker='o', color='red')
axs[1, 1].plot(n_steps, avg_scores, marker='o', color='purple')

# Adding title and labels
axs[0, 0].set(title="Highest Score for Each n_steps", xlabel="n_steps", ylabel="Highest Score")
axs[0, 1].set(title="Highest Reward for Each n_steps", xlabel="n_steps", ylabel="Highest Reward")
axs[1, 0].set(title="Average Reward for Each n_steps", xlabel="n_steps", ylabel="Average Reward")
axs[1, 1].set(title="Average Score for Each n_steps", xlabel="n_steps", ylabel="Average Score")

# Rotating x-axis labels and enabling grid for better readability
for ax in axs.flat:
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.

# Adjust spacing
plt.tight_layout()
plt.show()
