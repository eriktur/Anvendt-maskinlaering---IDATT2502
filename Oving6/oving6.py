#kopiert fra egen jupyter notebook

import  pandas as pd


data = pd.read_csv('./agaricus-lepiota.csv', header=None)

data.columns = ['edibility', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment',
                'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                'spore-print-color', 'population', 'habitat']

X = pd.get_dummies(data.drop('edibility', axis=1))
y = data['edibility'].apply(lambda x: 1 if x == 'p' else 0)  # 1 for giftig ('p'), 0 for spiselig ('e')

X.head()

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# Splitt dataene i trenings- og testsett
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier()
rfe = RFE(clf, n_features_to_select=10)
rfe.fit(X_train, y_train)

important_features = pd.Series(rfe.support_, index=X.columns)
important_features = important_features[important_features == True].index
print("Viktigste features (Feature Selection):", important_features)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Utfør PCA på dataene
pca = PCA(n_components=10)
pca.fit(X)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8,6))
plt.bar(range(1, 11), explained_variance, alpha=0.7, align='center', label='Explained variance')
plt.xlabel('Hovedkomponenter')
plt.ylabel('Forklart varians')
plt.title('Forklart varians av PCA-komponenter')
plt.show()

pca_components = pd.DataFrame(pca.components_, columns=X.columns)
print("Viktigste features fra PCA (Første komponent):")
print(pca_components.iloc[0].nlargest(10))  # De 10 viktigste features fra første komponent


overlapping_features = set(important_features).intersection(set(pca_components.iloc[0].nlargest(10).index))
print("Overlappende features mellom PCA og Feature Selection:", overlapping_features)
