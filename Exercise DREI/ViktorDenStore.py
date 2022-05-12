import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif

try:
    data = pd.read_pickle("data.pkl")
    labels = pd.read_pickle("labels.pkl")
except:
    file = tarfile.open('TCGA-PANCAN-HiSeq-801x20531.tar.gz')
    file.extractall()

    data = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col=0)
    data.to_pickle("data.pkl")

    labels = pd.read_csv("TCGA-PANCAN-HiSeq-801x20531/labels.csv", index_col=0)
    labels.to_pickle("labels.pkl")



print(f"data is loaded as {type(data)} with shape: {data.shape}")

data["Class"] = labels
target = "Class"
print(data.head())
print(data.Class.value_counts())

X = data.loc[:, data.columns != target]
Y = data.loc[:, data.columns == target]
print(X.shape)
print(Y.shape)
print(labels)
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=8)


#print(X.columns[0])
select_feature = SelectKBest(k=200).fit(x_train, y_train) # dont want entire data for overfitting reasons

selected_features_df = pd.DataFrame({'Feature':list(x_train.columns),
                                     'Scores':select_feature.scores_})
print(selected_features_df.sort_values(by='Scores', ascending=False))

x_train_chi = select_feature.transform(x_train)
print(x_train_chi)




