import matplotlib.pyplot as plt
import numpy as np

labels = ('time remaining', 'no. of actions', 'score', 'total reward')
agents = {
    'Rule Based': (297, 2062, 300, 3018),
    'PPO': (343, 1823, 700, 3431)
}

x = np.arange(len(labels))
bar_width = 0.35  # the width of the bars

fig, ax = plt.subplots()

for i, agent in enumerate(agents.keys()):
    measurements = [agents[agent][j] for j in range(len(labels))]
    ax.bar(x + i * bar_width, measurements, bar_width, label=agent)

ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Rule Based vs PPO in World 1 Stage 1')
ax.set_xticks(x + bar_width * (len(agents) - 1) / 2, labels)
ax.legend()

plt.show()
