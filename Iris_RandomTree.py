from sklearn.datasets import load_iris
#from sklearn.ensemble.forest import Randomforestclassifier
from sklearn.ensemble.forest import RandomForestClassifier
import pandas as pd
import numpy as np
np.random.seed(0)
iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()
df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)
df.head()
df['is_train']=np.random.uniform(0,1,len(df))<=0.75
df.head()


train,test=df[df['is_train']==True],df[df['is_train']==False]
print('Numbers of train datasets:',len(train))
print('Numbers of test datasets:',len(test))

features=df.columns[:4]
features
y=pd.factorize(train['species'])[0]
y

clf=RandomForestClassifier(n_jobs=2,random_state=0)
clf.fit(train[features],y)
clf.predict(test[features])
clf.predict_proba(test[features])[:5]

preds=iris.target_names[clf.predict(test[features])]
preds[:]
test['species'].head()
pd.crosstab(test['species'],preds,rownames=['Actual Species'],colnames=['Predicted Species'])
