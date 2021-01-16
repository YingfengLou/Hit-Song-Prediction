
# coding: utf-8

# In[16]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore') 
from scipy.io import arff
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


# In[121]:


train_set=pd.read_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/train_set.csv",encoding='unicode_escape')
test_set=pd.read_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/test_set.csv")
print("size of train set=",train_set.shape)
print("size of test set=",test_set.shape)


# In[18]:


#Preprocessing & feature extraction
#delete missing data because only 16 which is a very small number
#the feature：track, artist, uri are not usable features, so delete them.
#outlier
df_energy=train_set["energy"].describe()
IQR=df_energy["75%"]-df_energy["25%"]
if (df_energy["min"]>df_energy["25%"]-1.5*IQR)&(df_energy["max"]<df_energy["75%"]+1.5*IQR):
    print("no outlier for danceability")
else:
    print("process for the outlier")
print(df_energy)


# In[19]:


train_set_numerical=train_set.drop(['track'], axis=1)
train_set_numerical=train_set_numerical.drop(['Unnamed: 0'], axis=1).drop(['artist'], axis=1).drop(['uri'],axis=1)
train_set_numerical


# In[20]:


for column,row in train_set_numerical.iteritems():
    #print(index) # 输出列名
    df_column=train_set[column].describe()
    IQR=df_column["75%"]-df_column["25%"]
    if (df_column["min"]>df_column["25%"]-1.5*IQR)&(df_column["max"]<df_column["75%"]+1.5*IQR):
        print("no outlier for column", column)
    else:
        print("process for the outlier of",column)


# In[21]:


df_energy=train_set["energy"].describe()
IQR_energy=df_energy["75%"]-df_energy["25%"]
train_set_numerical["energy"][train_set_numerical.energy>df_energy["75%"]+1.5*IQR_energy]=df_energy["75%"]+1.5*IQR_energy
train_set_numerical["energy"][train_set_numerical.energy<df_energy["25%"]-1.5*IQR_energy]=df_energy["25%"]-1.5*IQR_energy

df_loudness=train_set["loudness"].describe()
IQR_loudness=df_loudness["75%"]-df_loudness["25%"]
train_set_numerical["loudness"][train_set_numerical.loudness>df_energy["75%"]+1.5*IQR_loudness]=df_loudness["75%"]+1.5*IQR_loudness
train_set_numerical["loudness"][train_set_numerical.loudness<df_energy["25%"]-1.5*IQR_loudness]=df_loudness["25%"]-1.5*IQR_loudness

df_speechiness=train_set["speechiness"].describe()
IQR_speechiness=df_speechiness["75%"]-df_speechiness["25%"]
train_set_numerical["speechiness"][train_set_numerical.speechiness
                                   >df_speechiness["75%"]+1.5*IQR_speechiness]=df_speechiness["75%"]+1.5*IQR_speechiness
train_set_numerical["speechiness"][train_set_numerical.speechiness
                                   <df_speechiness["25%"]-1.5*IQR_speechiness]=df_speechiness["25%"]-1.5*IQR_speechiness

df_acousticness=train_set["acousticness"].describe()
IQR_acousticness=df_acousticness["75%"]-df_acousticness["25%"]
train_set_numerical["acousticness"][train_set_numerical.acousticness
                                    >df_acousticness["75%"]+1.5*IQR_acousticness]=df_acousticness["75%"]+1.5*IQR_acousticness
train_set_numerical["acousticness"][train_set_numerical.acousticness
                                    <df_acousticness["25%"]-1.5*IQR_acousticness]=df_acousticness["25%"]-1.5*IQR_acousticness

df_instrumentalness=train_set["instrumentalness"].describe()
IQR_instrumentalness=df_instrumentalness["75%"]-df_instrumentalness["25%"]
train_set_numerical["instrumentalness"][train_set_numerical.instrumentalness
                              >df_instrumentalness["75%"]+1.5*IQR_instrumentalness]=df_instrumentalness["75%"]+1.5*IQR_instrumentalness
train_set_numerical["instrumentalness"][train_set_numerical.instrumentalness
                              <df_instrumentalness["25%"]-1.5*IQR_instrumentalness]=df_instrumentalness["25%"]-1.5*IQR_instrumentalness

df_liveness=train_set["liveness"].describe()
IQR_liveness=df_liveness["75%"]-df_liveness["25%"]
train_set_numerical["liveness"][train_set_numerical.liveness>df_liveness["75%"]+1.5*IQR_liveness]=df_liveness["75%"]+1.5*IQR_liveness
train_set_numerical["liveness"][train_set_numerical.liveness<df_liveness["25%"]-1.5*IQR_liveness]=df_liveness["25%"]-1.5*IQR_liveness

df_tempo=train_set["tempo"].describe()
IQR_tempo=df_tempo["75%"]-df_tempo["25%"]
train_set_numerical["tempo"][train_set_numerical.tempo>df_tempo["75%"]+1.5*IQR_tempo]=df_tempo["75%"]+1.5*IQR_tempo
train_set_numerical["tempo"][train_set_numerical.tempo<df_tempo["25%"]-1.5*IQR_tempo]=df_tempo["25%"]-1.5*IQR_tempo

df_duration_ms=train_set["duration_ms"].describe()
IQR_duration_ms=df_duration_ms["75%"]-df_duration_ms["25%"]
train_set_numerical["duration_ms"][train_set_numerical.duration_ms
                                   >df_duration_ms["75%"]+1.5*IQR_duration_ms]=df_duration_ms["75%"]+1.5*IQR_duration_ms
train_set_numerical["duration_ms"][train_set_numerical.duration_ms
                                   <df_duration_ms["25%"]-1.5*IQR_duration_ms]=df_duration_ms["25%"]-1.5*IQR_duration_ms

df_time_signature=train_set["time_signature"].describe()
IQR_time_signature=df_time_signature["75%"]-df_time_signature["25%"]
train_set_numerical["time_signature"][train_set_numerical.time_signature
                                      >df_time_signature["75%"]+1.5*IQR_time_signature]=df_time_signature["75%"]+1.5*IQR_time_signature
train_set_numerical["time_signature"][train_set_numerical.time_signature
                                      <df_time_signature["25%"]-1.5*IQR_time_signature]=df_time_signature["25%"]-1.5*IQR_time_signature

df_chorus_hit=train_set["chorus_hit"].describe()
IQR_chorus_hit=df_chorus_hit["75%"]-df_chorus_hit["25%"]
train_set_numerical["chorus_hit"][train_set_numerical.chorus_hit
                                  >df_chorus_hit["75%"]+1.5*IQR_chorus_hit]=df_chorus_hit["75%"]+1.5*IQR_chorus_hit
train_set_numerical["chorus_hit"][train_set_numerical.chorus_hit
                                  <df_chorus_hit["25%"]-1.5*IQR_chorus_hit]=df_chorus_hit["25%"]-1.5*IQR_chorus_hit

df_sections=train_set["sections"].describe()
IQR_sections=df_sections["75%"]-df_sections["25%"]
train_set_numerical["sections"][train_set_numerical.sections>df_sections["75%"]+1.5*IQR_sections]=df_sections["75%"]+1.5*IQR_sections
train_set_numerical["sections"][train_set_numerical.sections<df_sections["25%"]-1.5*IQR_sections]=df_sections["25%"]-1.5*IQR_sections


# In[22]:


for column,row in train_set_numerical.iteritems():
    #print(index) #output the index of column
    df_column=train_set_numerical[column].describe()
    IQR=df_column["75%"]-df_column["25%"]
    if (df_column["min"]>= df_column["25%"]-1.5*IQR)&(df_column["max"]<= df_column["75%"]+1.5*IQR):
        print("no outlier for column", column)
    else:
        print("process for the outlier of",column)
#train_set_numerical


# In[24]:


#stdandarization
#self.mean, self.std = X_train.mean(), X_train.std()
#self.feature_num = len(X_train.columns.tolist())
X_train=train_set_numerical.drop(['target'], axis=1)
std_X_train = (X_train - X_train.mean()) / X_train.std()
#applied the std of X_train to the test set
#std_X_test = (X_test - X_train.mean()) / X_train.std()

#find out the time_signature are almostly the same so drop it.
std_X_train=std_X_train.drop(['time_signature'],axis=1)
std_X_train


# In[10]:


sns.pairplot(train_set_numerical, hue="target", kind="scatter")
plt.show()


# In[44]:


sns.boxplot(data=std_X_train['danceability'])
plt.xlabel('danceability')
plt.show()

sns.boxplot(data=std_X_train['energy'])
plt.xlabel('energy')
plt.show()

sns.boxplot(data=std_X_train['key'])
plt.xlabel('key')
plt.show()

sns.boxplot(data=std_X_train['loudness'])
plt.xlabel('loudness')
plt.show()

sns.boxplot(data=std_X_train['mode'])
plt.xlabel('mode')
plt.show()

sns.boxplot(data=std_X_train['speechiness'])
plt.xlabel('speechiness')
plt.show()

sns.boxplot(data=std_X_train['acousticness'])
plt.xlabel('acousticness')
plt.show()

sns.boxplot(data=std_X_train['instrumentalness'])
plt.xlabel('instrumentalness')
plt.show()

sns.boxplot(data=std_X_train['valence'])
plt.xlabel('valence')
plt.show()

sns.boxplot(data=std_X_train['tempo'])
plt.xlabel('tempo')
plt.show()

sns.boxplot(data=std_X_train['duration_ms'])
plt.xlabel('duration_ms')
plt.show()

sns.boxplot(data=std_X_train['chorus_hit'])
plt.xlabel('chorus_hit')
plt.show()

sns.boxplot(data=std_X_train['sections'])
plt.xlabel('sections')
plt.show()

sns.boxplot(data=std_X_train['liveness'])
plt.xlabel('liveness')
plt.show()


# In[54]:


corrmat = std_X_train.corr()
plt.subplots(figsize=(18, 15))
ax = sns.heatmap(corrmat, vmax=1, annot=True, square=True, vmin=0)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Correlation Heatmap Between Each Feature')
plt.show()


# In[41]:


#split train : validation=8:2
from sklearn.model_selection import train_test_split
x_train=std_X_train
y_train=train_set_numerical['target']
#y_train
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=50)
#logistic regression with l1
lamda=np.logspace(-3,3,30)
train_score=[]
val_score=[]
for l in lamda:
    m=LogisticRegression(penalty="l1",C=l, solver="liblinear")
    model=m.fit(x_train, y_train)
    print("lamda:", l)
    print("train score:", model.score(x_train,y_train))
    #train_score.append(model.score(x_train,y_train))
    print("validation score", model.score(x_val, y_val))
    print("")
    train_score.append(model.score(x_train,y_train))
    val_score.append(model.score(x_val, y_val))
    
    
    #test_score.append(m.score(x_test,y_test))
    
best_l=lamda[(val_score.index(max(val_score)))]
print("best l:",best_l)
#print("train acc",train_score)
print("test acc",max(val_score))


# In[42]:


#logistic regression with l1
lamda=np.logspace(-3,3,30)
train_score=[]
val_score=[]
for l in lamda:
    m=LogisticRegression(penalty="l2",C=l, solver="liblinear")
    model=m.fit(x_train, y_train)
    print("lamda:", l)
    print("train score:", model.score(x_train,y_train))
    #train_score.append(model.score(x_train,y_train))
    print("validation score", model.score(x_val, y_val))
    print("")
    train_score.append(model.score(x_train,y_train))
    val_score.append(model.score(x_val, y_val))
    
    
    #test_score.append(m.score(x_test,y_test))
    
best_l=lamda[(val_score.index(max(val_score)))]
print("best l:",best_l)
#print("train acc",train_score)
print("val acc",max(val_score))


# In[86]:


#decision tree
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

model_tree = tree.DecisionTreeClassifier()

# search the best params

grid_tree= {'min_samples_split': [5, 10, 20, 50, 100,200, 500],
         'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'min_samples_leaf': [1, 2, 4,8,16]}

params_tree = GridSearchCV(model_tree, grid_tree,cv=5)
params_tree.fit(x_train, y_train)

pred_tree = params_tree.predict(x_val)

# get the accuracy score
acc_tree = accuracy_score(pred_tree, y_val)
print("the validation error=", acc_tree)
print(params_tree.best_params_)


# In[91]:


#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# search the best params
#grid_rf = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
grid_rf={'max_depth': [2,5,10, 20, None],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [2, 5, 10],
          'n_estimators': [100, 200, 300, 400, 500]}

clf_rf = GridSearchCV(rf, grid_rf, cv=5)
clf_rf.fit(x_train, y_train)

pred_rf = clf_rf.predict(x_val)
# get the accuracy score
acc_rf = accuracy_score(pred_rf, y_val)
print(acc_rf)
print(clf_rf.best_params_)


# In[92]:


from sklearn.ensemble import AdaBoostClassifier

adaboost=AdaBoostClassifier()
grid_ada={'n_estimators':[100,200,300,400,500],
          'learning_rate':[0.025,0.05, 0.1, 0.15,0.20,0.25,0.30]}

param_ada=GridSearchCV(adaboost, grid_ada, cv=5)
param_ada.fit(x_train, y_train)
pred_ada=param_ada.predict(x_val)
acc_ada=accuracy_score(pred_ada, y_val)
print("validation error of AdaBoost=", acc_ada)
print("best paramaters:", param_ada.best_params_)


# In[125]:


test_set_n=test_set.drop(['Unnamed: 0'],axis=1).drop(['track'],axis=1).drop(['artist'],axis=1).drop(['uri'],axis=1)
#test_set_numerical=
std_X_test= (test_set_n.drop(['target'],axis=1)- X_train.mean()) / X_train.std()
std_x_test=std_X_test.drop(['time_signature'],axis=1)
std_x_test

#test_set_n


# In[132]:


#choose random forest
x_test=std_x_test
y_test=test_set_n['target']
model_best = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200)
model_best.fit(std_X_train, train_set_numerical['target'])

pred_best = model_best.predict(x_test)
# get the accuracy score
acc_best = accuracy_score(pred_best, y_test)
print(acc_best)
#print(clf_rf.best_params_)

