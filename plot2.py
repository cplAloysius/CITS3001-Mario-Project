import matplotlib.pyplot as plt
import numpy as np
import textwrap

labels = ('x pos', 'total reward', 'score', 'no. of actions', 'time remaining (higher is better)')
agents = {
    'Rule Based': (439.875, 371.625, 25, 280, 349),
    'PPO': (456.125, 413.95, 105, 209.325, 352.25)
}

x = np.arange(len(labels))
bar_width = 0.35  # the width of the bars

fig, ax = plt.subplots()

for i, agent in enumerate(agents.keys()):
    measurements = [agents[agent][j] for j in range(len(labels))]
    ax.bar(x + i * bar_width, measurements, bar_width, label=agent)

ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Rule Based vs PPO in Unseen Levels')
label_wrap = [textwrap.fill(label, 12) for label in labels]
ax.set_xticks(x + bar_width * (len(agents) - 1) / 2)
ax.set_xticklabels(label_wrap) 
ax.legend()

plt.show()
