
# coding: utf-8

# In[1]:


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
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# In[3]:


#Load dataset and divide them into train : test =8:2
#df1=pd.read_csv('C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/dataset-of-00s.csv')
#df2=pd.read_csv('C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/dataset-of-10s.csv')
#DATA=pd.concat([df1,df2])
#print(df1.shape)
#print(df2.shape)
#print(DATA.shape)
#DATA.reset_index(drop=True, inplace=True)
#DATA
#DATA=DATA.sample(frac=1.0)
#DATA.reset_index(drop=True, inplace=True)#shuffle the dataset
#DATA
#train_set=DATA.loc[0:9816]
#test_set=DATA.loc[9817:12270]
#test_set.reset_index(drop=True, inplace=True)
#print("split the dataset into the proportion train:test=8:2 ")
#print("size of train set=",train_set.shape)
#print("size of test set=",test_set.shape)
#df=test_set.loc[test_set['target'].isin(['1'])]
#print("the target is 1 in test",df.shape)
#test_set.to_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/test_set.csv")
#train_set.to_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/train_set.csv")
#print("Save train and test dataset successfully!")


# In[2]:


train_set=pd.read_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/train_set.csv",
                      encoding='unicode_escape')
train_set=train_set.drop(["Unnamed: 0"],axis=1)
test_set=pd.read_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/test_set.csv",
                     encoding='unicode_escape').drop(["Unnamed: 0"],axis=1)
print("size of train set=",train_set.shape)
print("size of test set=",test_set.shape)
#train_set


# In[4]:


DATA_miss=pd.read_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/data set with missing data.csv",
                      encoding='unicode_escape')
DATA_miss.shape


# In[5]:


DATA_miss.isnull().sum()


# In[6]:


train_set_miss=pd.read_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/train_set_miss.csv",
                      encoding='unicode_escape')
train_set_miss=train_set_miss.drop(["Unnamed: 0"],axis=1)
test_set_miss=pd.read_csv("C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/test_set_miss.csv",
                     encoding='unicode_escape').drop(["Unnamed: 0"],axis=1)
print("size of train set=",train_set_miss.shape)
print("size of test set=",test_set_miss.shape)
#train_set


# In[71]:


def MissValue(data):
    miss_ratio = (data.isnull().sum() / len(data)) * 100
    miss_ratio = miss_ratio.sort_values(ascending=False)
    AllNull_train_ratio = miss_ratio.drop(miss_ratio[miss_ratio == 0].index)
    missing_train_ratio = pd.DataFrame({'Missing train data ratio': AllNull_train_ratio})
    print(missing_train_ratio)

    f, ax = plt.subplots(figsize=(10, 10))
    plt.xticks(rotation='90')  # ratate direction of words for each feature
    sns.barplot(x=miss_ratio.index, y=miss_ratio)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent', fontsize=15)
    plt.title('Percentage of Missing Data', fontsize=15)
    plt.show()


# In[72]:


MissValue(DATA_miss)


# In[73]:


MissValue(train_set_miss)


# In[15]:


#delete "uri and artist"
train_miss_numerical=train_set_miss.drop(['track'], axis=1)
train_miss_numerical=train_miss_numerical.drop(['artist','uri','Unnamed: 0.1'], axis=1)
train_miss_numerical.shape


# In[16]:


test_miss_numerical=test_set_miss.drop(['track'], axis=1)
test_miss_numerical=test_miss_numerical.drop(['artist','uri','Unnamed: 0.1'], axis=1)
test_miss_numerical.shape


# In[17]:


test_miss_numerical.isnull().sum()


# In[18]:


#for missing value in instrumentalness and acousticness - delete the datapoint
train_miss_numerical=train_miss_numerical.dropna(subset=['instrumentalness',
                                                        'speechiness'], how='any')
print(train_miss_numerical.shape)
test_miss_numerical=test_miss_numerical.dropna(subset=['instrumentalness',
                                                        'speechiness'], how='any')
print(test_miss_numerical.shape)


# In[19]:


from sklearn.impute import KNNImputer
impute_knn=KNNImputer(n_neighbors=2)
impute_knn.fit(train_miss_numerical)
impute_knn_=impute_knn.transform(train_miss_numerical)
df_train=pd.DataFrame(impute_knn_)

df_train.columns = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                   'instrumentalness','liveness','valence','tempo','duration_ms','time_signature','chorus_hit',
                    'sections','target']
df_train.shape


# In[20]:


#impute_knn_test=KNNImputer(n_neighbors=2)
#impute_knn_test.fit(test_miss_numerical)
impute_knn_test=impute_knn.transform(test_miss_numerical)
df_test=pd.DataFrame(impute_knn_test)

df_test.columns = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                   'instrumentalness','liveness','valence','tempo','duration_ms','time_signature','chorus_hit',
                    'sections','target']
df_test.shape


# In[21]:


df_test.isnull().sum()


# In[75]:


#Preprocessing & feature extraction
#delete missing data because only 16 which is a very small number
#the feature：track, artist, uri are not usable features, so delete them.
#outlier
train_set_numerical=train_set.drop(['track'], axis=1)
train_set_numerical=train_set_numerical.drop(['artist','uri'], axis=1)
#train_set_numerical
for column,row in train_set_numerical.iteritems():
    #print(index) # 输出列名
    df_column=train_set[column].describe()
    IQR=df_column["75%"]-df_column["25%"]
    if (df_column["min"]>df_column["25%"]-1.5*IQR)&(df_column["max"]<df_column["75%"]+1.5*IQR):
        print("No outlier for column", column)
    else:
        print("Have outlier of ",column)


# In[82]:


x_train_outlier=train_set_numerical.drop(['target'],axis=1)
y_train_outlier=train_set_numerical['target']
model_log_outlier=LogisticRegression(penalty="l2")
model_log_noutlier=model_log_outlier.fit(x_train_outlier, y_train_outlier)

pred_train_log_outlier=model_log_outlier.predict(x_train_outlier)

acc_train_logistic_outlier = accuracy_score(pred_train_log_outlier, y_train_outlier)


print("the train accuracy =", acc_train_logistic_outlier)
print('REC of training set = ',recall_score(y_train_outlier,pred_train_log_outlier,average='micro'))
print('F1-Score of training set = ',f1_score(y_train_outlier,pred_train_log_outlier,average='micro'))
print("")


# In[22]:


for column,row in df_train.iteritems():
    #print(index) # 输出列名
    df_column=df_train[column].describe()
    IQR=df_column["75%"]-df_column["25%"]
    if (df_column["min"]>df_column["25%"]-1.5*IQR)&(df_column["max"]<df_column["75%"]+1.5*IQR):
        print("No outlier for column", column)
    else:
        print("Have outlier of ",column)


# In[23]:


#deal with the outlier
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


for column,row in train_set_numerical.iteritems():
    #print(index) #output the index of column
    df_column=train_set_numerical[column].describe()
    IQR=df_column["75%"]-df_column["25%"]
    if (df_column["min"]>= df_column["25%"]-1.5*IQR)&(df_column["max"]<= df_column["75%"]+1.5*IQR):
        print("no outlier for column", column)
    else:
        print("process for the outlier of",column)
        
print("")
print("Process all the outlier successfully! ")


# In[24]:


#deal with the outlier
df_danceability=df_train["danceability"].describe()
IQR_danceability=df_danceability["75%"]-df_danceability["25%"]
df_train["danceability"][df_train.danceability>df_danceability["75%"]+1.5*IQR_danceability]=df_danceability["75%"]+1.5*IQR_danceability
df_train["danceability"][df_train.danceability<df_danceability["25%"]-1.5*IQR_danceability]=df_danceability["25%"]-1.5*IQR_danceability

df_energy=df_train["energy"].describe()
IQR_energy=df_energy["75%"]-df_energy["25%"]
df_train["energy"][df_train.energy>df_energy["75%"]+1.5*IQR_energy]=df_energy["75%"]+1.5*IQR_energy
df_train["energy"][df_train.energy<df_energy["25%"]-1.5*IQR_energy]=df_energy["25%"]-1.5*IQR_energy

df_loudness=df_train["loudness"].describe()
IQR_loudness=df_loudness["75%"]-df_loudness["25%"]
df_train["loudness"][df_train.loudness>df_energy["75%"]+1.5*IQR_loudness]=df_loudness["75%"]+1.5*IQR_loudness
df_train["loudness"][df_train.loudness<df_energy["25%"]-1.5*IQR_loudness]=df_loudness["25%"]-1.5*IQR_loudness

df_speechiness=df_train["speechiness"].describe()
IQR_speechiness=df_speechiness["75%"]-df_speechiness["25%"]
df_train["speechiness"][df_train.speechiness
                                   >df_speechiness["75%"]+1.5*IQR_speechiness]=df_speechiness["75%"]+1.5*IQR_speechiness
df_train["speechiness"][df_train.speechiness
                                   <df_speechiness["25%"]-1.5*IQR_speechiness]=df_speechiness["25%"]-1.5*IQR_speechiness

df_acousticness=df_train["acousticness"].describe()
IQR_acousticness=df_acousticness["75%"]-df_acousticness["25%"]
df_train["acousticness"][df_train.acousticness
                                    >df_acousticness["75%"]+1.5*IQR_acousticness]=df_acousticness["75%"]+1.5*IQR_acousticness
df_train["acousticness"][df_train.acousticness
                                    <df_acousticness["25%"]-1.5*IQR_acousticness]=df_acousticness["25%"]-1.5*IQR_acousticness

df_instrumentalness=df_train["instrumentalness"].describe()
IQR_instrumentalness=df_instrumentalness["75%"]-df_instrumentalness["25%"]
df_train["instrumentalness"][df_train.instrumentalness
                              >df_instrumentalness["75%"]+1.5*IQR_instrumentalness]=df_instrumentalness["75%"]+1.5*IQR_instrumentalness
df_train["instrumentalness"][df_train.instrumentalness
                              <df_instrumentalness["25%"]-1.5*IQR_instrumentalness]=df_instrumentalness["25%"]-1.5*IQR_instrumentalness

df_liveness=df_train["liveness"].describe()
IQR_liveness=df_liveness["75%"]-df_liveness["25%"]
df_train["liveness"][df_train.liveness>df_liveness["75%"]+1.5*IQR_liveness]=df_liveness["75%"]+1.5*IQR_liveness
df_train["liveness"][df_train.liveness<df_liveness["25%"]-1.5*IQR_liveness]=df_liveness["25%"]-1.5*IQR_liveness

df_tempo=df_train["tempo"].describe()
IQR_tempo=df_tempo["75%"]-df_tempo["25%"]
df_train["tempo"][df_train.tempo>df_tempo["75%"]+1.5*IQR_tempo]=df_tempo["75%"]+1.5*IQR_tempo
df_train["tempo"][df_train.tempo<df_tempo["25%"]-1.5*IQR_tempo]=df_tempo["25%"]-1.5*IQR_tempo

df_duration_ms=df_train["duration_ms"].describe()
IQR_duration_ms=df_duration_ms["75%"]-df_duration_ms["25%"]
df_train["duration_ms"][df_train.duration_ms
                                   >df_duration_ms["75%"]+1.5*IQR_duration_ms]=df_duration_ms["75%"]+1.5*IQR_duration_ms
df_train["duration_ms"][df_train.duration_ms
                                   <df_duration_ms["25%"]-1.5*IQR_duration_ms]=df_duration_ms["25%"]-1.5*IQR_duration_ms

df_time_signature=df_train["time_signature"].describe()
IQR_time_signature=df_time_signature["75%"]-df_time_signature["25%"]
df_train["time_signature"][df_train.time_signature
                                      >df_time_signature["75%"]+1.5*IQR_time_signature]=df_time_signature["75%"]+1.5*IQR_time_signature
df_train["time_signature"][df_train.time_signature
                                      <df_time_signature["25%"]-1.5*IQR_time_signature]=df_time_signature["25%"]-1.5*IQR_time_signature

df_chorus_hit=df_train["chorus_hit"].describe()
IQR_chorus_hit=df_chorus_hit["75%"]-df_chorus_hit["25%"]
df_train["chorus_hit"][df_train.chorus_hit
                                  >df_chorus_hit["75%"]+1.5*IQR_chorus_hit]=df_chorus_hit["75%"]+1.5*IQR_chorus_hit
df_train["chorus_hit"][df_train.chorus_hit
                                  <df_chorus_hit["25%"]-1.5*IQR_chorus_hit]=df_chorus_hit["25%"]-1.5*IQR_chorus_hit

df_sections=df_train["sections"].describe()
IQR_sections=df_sections["75%"]-df_sections["25%"]
df_train["sections"][df_train.sections>df_sections["75%"]+1.5*IQR_sections]=df_sections["75%"]+1.5*IQR_sections
df_train["sections"][df_train.sections<df_sections["25%"]-1.5*IQR_sections]=df_sections["25%"]-1.5*IQR_sections


for column,row in df_train.iteritems():
    #print(index) #output the index of column
    df_column=df_train[column].describe()
    IQR=df_column["75%"]-df_column["25%"]
    if (df_column["min"]>= df_column["25%"]-1.5*IQR)&(df_column["max"]<= df_column["75%"]+1.5*IQR):
        print("no outlier for column", column)
    else:
        print("process for the outlier of",column)
        
print("")
print("Process all the outlier successfully! ")


# In[38]:


#stdandarization
#self.mean, self.std = X_train.mean(), X_train.std()
#self.feature_num = len(X_train.columns.tolist())
X_train=train_set_numerical.drop(['target'], axis=1)
std_X_train = (X_train - X_train.mean()) / X_train.std()
test_set_n=test_set.drop(['track','artist','uri'],axis=1)
#test_set_numerical=
#applied the std of X_train to the test setb
std_X_test= (test_set_n.drop(['target'],axis=1)- X_train.mean()) / X_train.std()
std_x_test=std_X_test.drop(['time_signature'],axis=1)
std_x_test

#std_X_test = (X_test - X_train.mean()) / X_train.std()

#find out the time_signature are almostly the same so drop it.
std_X_train=std_X_train.drop(['time_signature'],axis=1)
std_X_train.shape


# In[39]:


#stdandarization for missing data
X_train_miss=df_train.drop(['target'], axis=1)
std_X_train_miss = (X_train_miss - X_train_miss.mean()) / X_train_miss.std()
test_set_n_miss=df_test
#applied the std of X_train to the test setb
std_X_test_miss= (test_set_n_miss.drop(['target'],axis=1)- X_train_miss.mean()) / X_train_miss.std()
std_x_test_miss=std_X_test_miss.drop(['time_signature'],axis=1)
std_x_test_miss

#find out the time_signature are almostly the same so drop it.
std_X_train_miss=std_X_train_miss.drop(['time_signature'],axis=1)
std_X_train_miss.shape


# In[83]:


corrmat = X_train.corr()
plt.subplots(figsize=(18, 15))
ax = sns.heatmap(corrmat, vmax=1, annot=True, square=True, vmin=0,cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Correlation Heatmap Between Each Feature')
plt.show()


# In[86]:


corrmat = train_set.corr()
plt.subplots(figsize=(18, 15))
ax = sns.heatmap(corrmat, vmax=1, annot=True, square=True, vmin=0,cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Correlation Heatmap Between Each Feature')
plt.show()


# In[8]:


sns.pairplot(train_set_numerical.drop(['loudness','mode','time_signature','duration_ms'],axis=1), hue="target", kind="scatter")
plt.show()


# In[31]:


sns.pairplot(train_set_numerical.drop(['loudness','mode','time_signature','speechiness',
                                      'key','liveness','valence','tempo','chorus_hit','sections'],axis=1), hue="target", kind="scatter")
plt.show()


# In[ ]:


sns.boxplot(x='diagnosis', y='area_mean', data=df)


# In[36]:


sns.boxplot(x='target', y='danceability',data=train_set_numerical)
plt.xlabel('danceability')
plt.show()

sns.boxplot(x='target', y='energy',data=train_set_numerical)
plt.xlabel('energy')
plt.show()
#sns.boxplot(data=std_X_train['key'])
#plt.xlabel('key')
#plt.show()

#sns.boxplot(data=std_X_train['loudness'])
#plt.xlabel('loudness')
#plt.show()

#sns.boxplot(data=std_X_train['mode'])
#plt.xlabel('mode')
#plt.show()

#sns.boxplot(data=std_X_train['speechiness'])
#plt.xlabel('speechiness')
#plt.show()
sns.boxplot(x='target', y='acousticness',data=train_set_numerical)
plt.xlabel('acousticness')
plt.show()

sns.boxplot(x='target', y='instrumentalness',data=train_set_numerical)
plt.xlabel('instrumentalness')
plt.show()

#sns.boxplot(data=std_X_train['valence'])
#plt.xlabel('valence')
#plt.show()

#sns.boxplot(data=std_X_train['tempo'])
#plt.xlabel('tempo')
#plt.show()

sns.boxplot(x='target', y='duration_ms',data=train_set_numerical)
plt.xlabel('duration_ms')
plt.show()

#sns.boxplot(data=std_X_train['chorus_hit'])
#plt.xlabel('chorus_hit')
#plt.show()

#sns.boxplot(data=std_X_train['sections'])
#plt.xlabel('sections')
#plt.show()

#sns.boxplot(data=std_X_train['liveness'])
#plt.xlabel('liveness')
#plt.show()


# In[40]:


#Use cross-validation
from sklearn.model_selection import train_test_split
x_train=std_X_train
y_train=train_set_numerical['target']
#y_train
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=50)
#logistic regression with l1
x_test=std_x_test
y_test=test_set_n['target']


# In[42]:


x_train_miss=std_X_train_miss
y_train_miss=df_train['target']
#y_train
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=50)
#logistic regression with l1
x_test_miss=std_x_test_miss
y_test_miss=df_test['target']


# In[10]:


corrmat = std_X_train.corr()
plt.subplots(figsize=(18, 15))
ax = sns.heatmap(corrmat, vmax=1, annot=True, square=True, vmin=0,cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Correlation Heatmap Between Each Feature')
plt.show()


# In[45]:


#logistic regression
model_log=LogisticRegression(penalty="none")
model_log_n=model_log.fit(x_train, y_train)
model_log_miss=model_log.fit(x_train_miss, y_train_miss)
pred_train_log=model_log.predict(x_train)
pred_train_log_miss=model_log.predict(x_train_miss)

acc_train_logistic = accuracy_score(pred_train_log, y_train)
acc_train_logistic_miss = accuracy_score(pred_train_log_miss, y_train_miss)

print("the train accuracy =", acc_train_logistic)
print('REC of training set = ',recall_score(y_train,pred_train_log,average='micro'))
print('F1-Score of training set = ',f1_score(y_train,pred_train_log,average='micro'))
print("")
print("the train accuracy =", acc_train_logistic_miss)
print('REC of training set = ',recall_score(y_train_miss,pred_train_log_miss,average='micro'))
print('F1-Score of training set = ',f1_score(y_train_miss,pred_train_log_miss,average='micro'))


# In[91]:


model_logistic_l1 = LogisticRegression(penalty="l1",solver="liblinear")

# search the best params-select model

grid_logistic_l1= {'C': np.logspace(-3,1,10)}

params_logistic = GridSearchCV(model_logistic_l1, grid_logistic_l1,cv=5)
params_logistic.fit(x_train, y_train)

pred_train_logistic = params_logistic.predict(x_train)

# get the accuracy score
acc_train_logistic = accuracy_score(pred_train_logistic, y_train)
#acc_test_logistic = accuracy_score(pred_test_logistic, y_test)

print("the best paramater = ",params_logistic.best_params_)
print("best accuracy in validation = ",params_logistic.best_score_)
print("")
print("the train accuracy =", acc_train_logistic)
print('REC of training set = ',recall_score(y_train,pred_train_logistic,average='micro'))
print('F1-Score of training set = ',f1_score(y_train,pred_train_logistic,average='micro'))
print("")
dict_l1=params_logistic.cv_results_
x_p=dict_l1['params']
y_p=1-dict_l1['mean_test_score']
plt.plot(x_p,y_p)
plt.xticks(rotation=270)
plt.xlabel('C')
plt.ylabel('y: error rate')
plt.title('L1-Logistic Regression (complete dataset)')
plt.grid()
plt.show()


# In[93]:


model_logistic_l1 = LogisticRegression(penalty="l1",solver="liblinear")

# search the best params-select model

grid_logistic_l1= {'C': np.logspace(-3,1,10)}

params_logistic_miss = GridSearchCV(model_logistic_l1, grid_logistic_l1,cv=5)
params_logistic_miss.fit(x_train_miss, y_train_miss)

pred_train_logistic_miss = params_logistic_miss.predict(x_train_miss)

# get the accuracy score
acc_train_logistic_miss = accuracy_score(pred_train_logistic_miss, y_train_miss)
#acc_test_logistic = accuracy_score(pred_test_logistic, y_test)

print("the best paramater = ",params_logistic_miss.best_params_)
print("best accuracy in validation = ",params_logistic_miss.best_score_)
print("")
print("the train accuracy =", acc_train_logistic_miss)
print('REC of training set = ',recall_score(y_train_miss,pred_train_logistic_miss,average='micro'))
print('F1-Score of training set = ',f1_score(y_train_miss,pred_train_logistic_miss,average='micro'))
print("")


# In[98]:


dict_l1=params_logistic.cv_results_
dict_l1_miss=params_logistic_miss.cv_results_
x_p=dict_l1['params']
y_p1=1-dict_l1['mean_test_score']
y_p2=1-dict_l1_miss['mean_test_score']
plt.plot(x,y_p1,"-",label="complete")
plt.plot(x,y_p2,"-",label="miss values")
plt.legend(loc='upper right')
plt.xticks(rotation=270)
plt.xlabel('C')
plt.ylabel('error rate')
plt.title('L1-Logistic Regression (L1)')
plt.grid()
plt.show()


# In[53]:


x_train.shape
y_train.shape


# In[68]:


params_logistic.cv_results_


# In[70]:


dict_l1=params_logistic.cv_results_
x=dict_l1['params']
y=1-dict_l1['mean_test_score']
plt.plot(x,y)
plt.xticks(rotation=270)
plt.xlabel('C')
plt.ylabel('y: error rate')
plt.title('L1-Logistic Regression')
plt.grid()
plt.show()


# In[99]:


#logistic regression with l2

model_logistic_l2 = LogisticRegression(penalty="l2",solver="liblinear")

# search the best params-select model

grid_logistic_l2= {'C': np.logspace(-3,3,10)}

params_logistic_l2 = GridSearchCV(model_logistic_l2, grid_logistic_l2,cv=5)
params_logistic_l2.fit(x_train, y_train)

pred_train_logistic_l2= params_logistic_l2.predict(x_train)
#pred_test_logistic_l2=params_logistic_l2.predict(x_test)

# get the accuracy score
acc_train_logistic_l2 = accuracy_score(pred_train_logistic_l2, y_train)
#acc_test_logistic_l2 = accuracy_score(pred_test_logistic_l2, y_test)

print("the best paramater = ",params_logistic_l2.best_params_)
print("best accuracy in validation = ",params_logistic_l2.best_score_)
print("")
print("the train accuracy =", acc_train_logistic_l2)
print('REC of training set = ',recall_score(y_train,pred_train_logistic_l2,average='micro'))
print('F1-Score of training set = ',f1_score(y_train,pred_train_logistic_l2,average='micro'))
print("")
#print("the test accuracy =", acc_test_logistic_l2)
#print('REC of test set = ',recall_score(y_test,pred_test_logistic_l2,average='micro'))
#print('F1-Score of test set = ',f1_score(y_test,pred_test_logistic_l2,average='micro'))


# In[18]:


dict_l2=params_logistic_l2.cv_results_
x=dict_l2['params']
y=1-dict_l2['mean_test_score']
plt.plot(x,y)
plt.xticks(rotation=270)
plt.xlabel('C')
plt.ylabel('y: error rate')
plt.title('L2-Logistic Regression')
plt.grid()
plt.show()


# In[101]:


#decision tree
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

model_tree = tree.DecisionTreeClassifier()

# search the best params
#'min_samples_split': [5, 10, 20, 50, 100,200, 500],
grid_tree= {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None], 'min_samples_leaf': [1, 2, 4,8,16]}

params_tree = GridSearchCV(model_tree, grid_tree,cv=5)
params_tree.fit(x_train, y_train)

pred_tree = params_tree.predict(x_test)

pred_train_tree= params_tree.predict(x_train)
#pred_test_tree= params_tree.predict(x_test)

# get the accuracy score
acc_train_tree = accuracy_score(pred_train_tree, y_train)
#acc_test_tree = accuracy_score(pred_test_tree, y_test)


# get the accuracy score
#acc_tree = accuracy_score(pred_tree, y_test)
#print("the validation error=", acc_tree)
#print(params_tree.best_params_)

print("the best paramater = ",params_tree.best_params_)
print("best accuracy in validation = ",params_tree.best_score_)
print("")
print("the train accuracy =", acc_train_tree)
print('REC of training set = ',recall_score(y_train,pred_train_tree,average='micro'))
print('F1-Score of training set = ',f1_score(y_train,pred_train_tree,average='micro'))


# In[104]:


model_tree_miss = tree.DecisionTreeClassifier()

# search the best params
#'min_samples_split': [5, 10, 20, 50, 100,200, 500],
grid_tree= {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None], 'min_samples_leaf': [1, 2, 4,8,16]}

params_tree_miss = GridSearchCV(model_tree_miss, grid_tree,cv=5)
params_tree_miss.fit(x_train_miss, y_train_miss)

pred_train_tree_miss= params_tree.predict(x_train_miss)

# get the accuracy score
acc_train_tree_miss = accuracy_score(pred_train_tree_miss, y_train_miss)

print("the best paramater = ",params_tree_miss.best_params_)
print("best accuracy in validation = ",params_tree_miss.best_score_)
print("")
print("the train accuracy =", acc_train_tree_miss)
print('REC of training set = ',recall_score(y_train_miss,pred_train_tree_miss,average='micro'))
print('F1-Score of training set = ',f1_score(y_train_miss,pred_train_tree_miss,average='micro'))


# In[51]:


#the code is modified from https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
def Plotgridsearch(cv_results, grid_param_2, grid_param_1, name_param_1, name_param_2):
    
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

        ax.set_title("GridSearch Scores with Maxdepth & min samples leaf")
        ax.set_xlabel(name_param_1)
        ax.set_ylabel('CV Score', fontsize=16)
        ax.legend(loc="best", fontsize=5)
        ax.grid('on')


# In[52]:


#min_samples_split=[5, 10, 20, 50, 100,200, 500]
max_depth=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
min_samples_leaf=[1, 2, 4,8,16]
Plotgridsearch(params_tree.cv_results_, min_samples_leaf, max_depth, 'min_samples_leaf','max_depth')


# In[106]:


dict_tree=params_tree.cv_results_
array=np.array(dict_tree['params'])
x_tree=[]
for i in array:
    i=dict_tree['params'].index(i)
    x_tree.append(i)
x_tree
y=1-dict_tree['mean_test_score']
#plt.figure(figsize=(12, 5))
plt.plot(x_tree,y)
#plt.xticks(rotation=270)
plt.xlabel('C')
plt.ylabel('y: error rate')
plt.title('Decision Tree')
plt.grid()
plt.show()


# In[107]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier()

# search the best params
#grid_rf = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
grid_rf={'max_depth': [2,5,10, 20, None],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [2, 5, 10],
          'n_estimators': [10,50,100,200, 300, 400, 500]}

clf_rf = RandomizedSearchCV(rf, grid_rf, cv=5)
clf_rf.fit(x_train, y_train)

pred_train_rf= clf_rf.predict(x_train)
#pred_test_rf= clf_rf.predict(x_test)

# get the accuracy score
acc_train_rf = accuracy_score(pred_train_rf, y_train)
#acc_test_rf = accuracy_score(pred_test_rf, y_test)

# get the accuracy score
#acc_tree = accuracy_score(pred_tree, y_test)
#print("the validation error=", acc_tree)
#print(params_tree.best_params_)

print("the best paramater = ",clf_rf.best_params_)
print("best accuracy in validation = ",clf_rf.best_score_)
print("")
print("the train accuracy =", acc_train_rf)
print('REC of training set = ',recall_score(y_train,pred_train_rf,average='micro'))
print('F1-Score of training set = ',f1_score(y_train,pred_train_rf,average='micro'))
print("")


# In[157]:


x=x_train[['danceability','instrumentalness']]
y=y_train

rf_b = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                            min_samples_leaf=1, max_depth=None)
rf_b.fit(x,y)

#h=.02
#x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
#y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
#Z = clf_rf.predict(np.c_[xx.ravel(), yy.ravel()])    

x_min, x_max = x.iloc[:, 0].min() - 1, x.iloc[:, 0].max() + 1
y_min, y_max = x.iloc[:, 1].min() - 1, x.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

Z = rf_b.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, alpha=0.8)
plt.show()


# In[117]:


rf_miss = RandomForestClassifier()

# search the best params
#grid_rf = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
grid_rf_miss={'max_depth': [2,5,10, 20, None],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [2, 5, 10],
          'n_estimators': [10,50,100,200, 300, 400, 500]}

clf_rf_miss= RandomizedSearchCV(rf_miss, grid_rf_miss, cv=5)
clf_rf_miss.fit(x_train_miss, y_train_miss)

pred_train_rf_miss= clf_rf_miss.predict(x_train_miss)

acc_train_rf_miss = accuracy_score(pred_train_rf_miss, y_train_miss)


print("the best paramater = ",clf_rf_miss.best_params_)
print("best accuracy in validation = ",clf_rf_miss.best_score_)
print("")
print("the train accuracy =", acc_train_rf_miss)
print('REC of training set = ',recall_score(y_train_miss,pred_train_rf_miss,average='micro'))
print('F1-Score of training set = ',f1_score(y_train_miss,pred_train_rf_miss,average='micro'))
print("")


# In[108]:


rf_ft = RandomForestClassifier(n_estimators=400, min_samples_split=5, min_samples_leaf=2, max_depth=None)
rf_ft.fit(x_train, y_train)
arr_imp=rf_ft.feature_importances_
df_imp=pd.DataFrame(arr_imp)
df_imp.index=['danceability','energy','key','loudness','mode','speechiness','acousticness',
                   'instrumentalness','liveness','valence','tempo','duration_ms','chorus_hit',
                    'sections']
df_imp.columns = ['weight']
df_imp


# In[109]:


clf_rf.cv_results_


# In[112]:


dict_rf=clf_rf.cv_results_
array=np.array(dict_rf['params'])
array


# In[113]:


x=[]
for i in array:
    i=dict_rf['params'].index(i)
    x.append(i)
x


# In[139]:




y=1-dict_rf['mean_test_score']
#plt.figure(figsize=(12, 5))
plt.plot(x,y)
#plt.xticks(rotation=270)
plt.xlabel('C')
plt.ylabel('y: error rate')
plt.title('Random Forest')
plt.grid()
plt.show()


# In[119]:


from sklearn.ensemble import AdaBoostClassifier

adaboost=AdaBoostClassifier()
grid_ada={'n_estimators':[100,200,300,400,500],
          'learning_rate':[0.025,0.05, 0.1, 0.15,0.20,0.25,0.30]}

param_ada=GridSearchCV(adaboost, grid_ada, cv=5)
param_ada.fit(x_train, y_train)

pred_train_ada= param_ada.predict(x_train)
#pred_test_ada= param_ada.predict(x_test)

acc_train_ada = accuracy_score(pred_train_ada, y_train)
#acc_test_ada = accuracy_score(pred_test_ada, y_test)

# get the accuracy score
#acc_tree = accuracy_score(pred_tree, y_test)
#print("the validation error=", acc_tree)
#print(params_tree.best_params_)

print("the best paramater = ",param_ada.best_params_)
print("best accuracy in validation = ",param_ada.best_score_)
print("")
print("the train accuracy =", acc_train_ada)
print('REC of training set = ',recall_score(y_train,pred_train_ada,average='micro'))
print('F1-Score of training set = ',f1_score(y_train,pred_train_ada,average='micro'))


# In[121]:


dict_ada=param_ada.cv_results_
array_ada=np.array(dict_ada['params'])
array_ada


# In[130]:


x_ada=[]
for i in array_ada:
    i=dict_ada['params'].index(i)
    x_ada.append(i)
#x_ada


# In[138]:


y_ada=1-dict_ada['mean_test_score']
#plt.figure(figsize=(12, 5))
plt.plot(x_ada,y_ada)
#plt.xticks(rotation=270)
plt.xlabel('C')
plt.ylabel('y: error rate')
plt.title('Adaboost')
plt.grid()
plt.show()


# In[129]:


adaboost_miss=AdaBoostClassifier()
grid_miss={'n_estimators':[100,200,300,400,500],
          'learning_rate':[0.025,0.05, 0.1, 0.15,0.20,0.25,0.30]}

param_ada_miss=GridSearchCV(adaboost_miss, grid_miss, cv=5)
param_ada_miss.fit(x_train_miss, y_train_miss)
pred_train_ada_miss= param_ada_miss.predict(x_train_miss)
#pred_test_ada= param_ada.predict(x_test)

acc_train_ada_miss = accuracy_score(pred_train_ada_miss, y_train_miss)
#acc_test_ada = accuracy_score(pred_test_ada, y_test)

# get the accuracy score
#acc_tree = accuracy_score(pred_tree, y_test)
#print("the validation error=", acc_tree)
#print(params_tree.best_params_)

print("the best paramater = ",param_ada_miss.best_params_)
print("best accuracy in validation = ",param_ada_miss.best_score_)
print("")
print("the train accuracy =", acc_train_ada_miss)
print('REC of training set = ',recall_score(y_train_miss,pred_train_ada_miss,average='micro'))
print('F1-Score of training set = ',f1_score(y_train_miss,pred_train_ada_miss,average='micro'))


# In[140]:


df1_semi=pd.read_csv('C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/dataset-of-00s.csv')
df2_semi=pd.read_csv('C:/Users/hp/Desktop/EE660/Final_Project/Myproject/dataset/dataset-of-10s.csv')
DATA_semi=pd.concat([df1_semi,df2_semi])
DATA_numerical_semi=DATA_semi.drop(['track','artist','uri','time_signature'], axis=1)
DATA_numerical_semi


# In[141]:


#the code is modified from https://www.jianshu.com/p/2aad8205738c 
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score,recall_score,f1_score
x_semi = DATA_numerical_semi
labels = x_semi.target
x_train_semi=DATA_numerical_semi.drop(['target'])
reg=np.random.RandomState(42)
random_unlabeled_points = reg.rand(len(x_semi.target))<0.3
#random_unlabeled_points = random_unlabeled_points
y=labels[random_unlabeled_points] #the label before delete the labels
labels[random_unlabeled_points]=-1 # make these labels become unlabeled data
print('Unlabeled Number:',list(labels).count(-1))

semi_model = LabelPropagation()
semi_model.fit(x_train_semi,labels)
y_pred = semi_model.predict(x_train_semi)
y_pred = y_pred[random_unlabeled_points] # predict the unlabeled data
print('accuracy',accuracy_score(y,y_pred))
print('Recall',recall_score(y,y_pred,average='micro'))
print('F1-Score',f1_score(y,y_pred,average='micro'))


# In[159]:


rf_final = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                            min_samples_leaf=1, max_depth=None)

rf_final.fit(x_train, y_train)

pred_train_rf= rf_final.predict(x_train)
pred_test_rf= rf_final.predict(x_test)

# get the accuracy score
acc_train_rf = accuracy_score(pred_train_rf, y_train)
acc_test_rf = accuracy_score(pred_test_rf, y_test)

#print("the train accuracy =", acc_train_rf)
#print('REC of training set = ',recall_score(y_train,pred_train_rf,average='micro'))
#print('F1-Score of training set = ',f1_score(y_train,pred_train_rf,average='micro'))

print("the train accuracy =", acc_test_rf)
print('REC of training set = ',recall_score(y_test,pred_test_rf,average='micro'))
print('F1-Score of training set = ',f1_score(y_test,pred_test_rf,average='micro'))
print("")


# In[163]:


import pickle
with open('model.pickle', 'wb') as file:
    pickle.dump(rf_final, file)


# In[162]:


#baseline
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)
DummyClassifier(strategy='most_frequent')
dummy_clf.predict(x_train)
#array([1, 1, 1, 1])
dummy_clf.predict(x_test)
dummy_clf.score(x_train, y_train)
dummy_clf.score(x_test, y_test)

