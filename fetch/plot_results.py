import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import pandas as pd


# ###### Touch Block Experiments
#
# csv_files = ['small_buffer/seed87_scores.csv', 'bad_cropping/seed71_scores.csv', 'control/seed48_scores.csv', 'messy_scene/seed52_scores.csv', 'no_extra_penalties/seed84_scores.csv', 'old_reward_function/seed94_scores.csv']
# all_mean_scores = []
# for csv_file in csv_files:
#     mean_scores = []
#     reader = csv.reader(open(csv_file, 'r'))
#     for row in reader:
#         mean_scores.append(row[0])
#     all_mean_scores.append(mean_scores)
#
# df = pd.DataFrame(all_mean_scores)
# df = df.transpose()
# df.columns = ['Small Buffer (8K)', 'Bad Cropping', 'Control', 'Messy Scene + Bad Cropping', 'No Additional Z Penalties', '1/distance Reward Function']
# df['index'] = pd.Series(np.arange(0, 1500))
# df = df.astype(float)
# ax = df.plot(x='index')
# ax.set_xlabel("Epochs")
# ax.set_ylabel("Avg Score at End of Epoch")
# plt.title("Variations on 'Touch Block' Experiment")
# plt.legend(loc='best')
# plt.show()


# csv_file = 'gripper_enabled/prepare_to_grasp/seed7_scores.csv'
csv_file = 'gripper_enabled/push_right/seed40_scores.csv'
reader = csv.reader(open(csv_file, 'r'))
data = []
for row in reader:
    data.append((row[0]))
df = pd.DataFrame(data)
df.columns = ['prepare to grasp']
df['index'] = pd.Series(np.arange(0, 1500))
df = df.astype(float)
ax = df.plot(x='index', y='prepare to grasp')
ax.set_xlabel("Epochs")
ax.set_ylabel("Avg Score at End of Epoch")
plt.title("Preparing to Grasp Block")
plt.show()