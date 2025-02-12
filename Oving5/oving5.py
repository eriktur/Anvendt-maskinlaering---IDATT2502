#kopiert fra egen jupyter notebook

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./agaricus-lepiota.data', header=None, thousands=',').dropna()

data.columns = ['edibility', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment',
                'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                'spore-print-color', 'population', 'habitat']

data.describe()

data[['habitat', 'edibility']].groupby('habitat').describe().transpose()

data.head()


edibility = data[data['edibility'] == 'p']
non_edibility = data[data['edibility'] == 'e']


fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 6))


axs[0].hist(edibility['habitat'], bins=20, color='red')
axs[0].set_title('Habitat for Giftige Sopper')
axs[0].set_xlabel('Habitat')
axs[0].set_ylabel('Antall Sopper')


axs[1].hist(non_edibility['habitat'], bins=20, color='green')
axs[1].set_title('Habitat for Spiselige Sopper')
axs[1].set_xlabel('Habitat')


fig.suptitle('Fordeling av Habitat mellom Giftige og Spiselige Sopper')


fig, axs = plt.subplots(2, (len(data.columns)//2) + 1, sharey=True, tight_layout=True, figsize=(15,5))
fig.suptitle('fordeling av egenskaper')

for index, col in enumerate(data.columns):
	axs[index // 12][index% 11].title.set_text(col)
	axs[index // 12][index % 11].hist(data[col], bins=5)

fig, axs = plt.subplots(1, len(data['population'].unique()), sharey=True, tight_layout=True, figsize=(15,5))
fig.suptitle('Population / edibility distribution')

for index, population in enumerate(data['population'].unique().tolist()):
	axs[index].title.set_text(population)
	axs[index].hist(data[data['population'] == population]['edibility'], bins=5)

dummies = pd.get_dummies(data, prefix=None,prefix_sep='_')
dummies

plt.spy(dummies, markersize=0.1)
fig = plt.gcf()
fig.set_size_inches(60,220)
plt.plot()
plt.show()

