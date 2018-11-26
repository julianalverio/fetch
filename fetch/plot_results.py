import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import pandas as pd


csv_files = ['small_buffer/seed87_scores.csv'] # TODO
all_mean_scores = []
for csv_file in csv_files:
    mean_scores = []
    reader = csv.reader(open(csv_file, 'r'))
    for row in reader:
        mean_scores.append(row[0])
    all_mean_scores.append(mean_scores)

df = pd.DataFrame(all_mean_scores[0], index=np.arange(0, 1500), columns=['small buffer']) #TODO
df['index'] = pd.Series(np.arange(0, 1500))
df = df.astype(float)
ax = df.plot(x='index', y='short_buffer')
ax.set_xlabel("Epochs")
ax.set_ylabel("Avg Score at End of Epoch")
plt.show()

