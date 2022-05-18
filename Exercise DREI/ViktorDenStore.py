import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=8)

select_feature = SelectKBest(k=200).fit(x_train, y_train) # dont want entire data for overfitting reasons

selected_features_df = pd.DataFrame({'Feature':list(x_train.columns),
                                     'Scores':select_feature.scores_})
#print(selected_features_df.sort_values(by='Scores', ascending=False))
#print(selected_features_df)
scores = np.array(selected_features_df.sort_values(by='Scores', ascending=False))
#print(scores[:,1])


r = 200
plo = scores[0:r]
#print(plo)
#plt.hist(scores[0:200, 1],bins=200)
plt.bar(np.linspace(0,r,r),plo[:,1])
#plt.show()

#sns.barplot(x="day", y="total_bill", data=plo[:,1], capsize=.1, ci="sd")
#sns.swarmplot(x="day", y="total_bill", data=plo[:,1], color="0", alpha=.35)


plt.title("Top 200 features")
plt.xlabel("Features sorted by f-score")
plt.ylabel("Score function")

fig, ax =plt.subplots(1,1)
data=[[300],
    [146],
    [141],
    [136],
    [78]
      ]
column_labels=["Class distribution"]
df=pd.DataFrame(data,columns=column_labels)
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values,
        colLabels=df.columns,
        rowLabels=["BRCA","KIRC","LUAD","PRAD","COAD"],
        loc="center")

plt.show()

#plt.plot(selected_features_df.sort_values(by='Scores', ascending=False))
#plt.show()

#def histedges_equalN(x, nbin):
 #   npt = len(x)
 #   return np.interp(np.linspace(0, npt, nbin + 1),
 #                    np.arange(npt),
#                     np.sort(x))

#x = np.random.randn(100)
#n, bins, patches = plt.hist(x, histedges_equalN(x, 10))
#plt.show()


#x_train_chi = select_feature.transform(x_train)
#print(x_train_chi)
