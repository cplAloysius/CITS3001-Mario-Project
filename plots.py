import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('1e6_1000000_results.csv')

total_rewards = df['total_rewards']

print(total_rewards[0])
print(total_rewards[1])
