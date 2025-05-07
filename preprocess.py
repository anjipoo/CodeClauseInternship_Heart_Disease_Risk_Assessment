#!/usr/bin/env python
# coding: utf-8

# In[97]:


# In[98]:


#importing libraires
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import GridSearchCV
import json


# In[99]:


df=pd.read_csv(r'C:\Users\anjan\OneDrive\Desktop\codeclause\heart\data\heart_disease_uci.csv')


# In[100]:


df.head()


# In[101]:


df.info()


# In[102]:


df.shape


# In[103]:


df.isnull().sum()


# In[104]:


df=df.drop(columns=['id','dataset','ca','thal'])


# In[105]:


df.columns


# In[106]:


numcols=['age','trestbps','chol','thalch','oldpeak']


# In[107]:


for c in numcols:
    df[c].fillna(df[c].median(),inplace=True)


# In[108]:


catcols=['sex','cp','fbs','restecg','exang','slope']


# In[109]:


for c in catcols:
    df[c].fillna(df[c].mode()[0],inplace=True)


# In[110]:


df.isnull().sum()


# In[111]:


for c in numcols:
    q1,q99=df[c].quantile([0.01,0.99])
    df[c]=df[c].clip(q1,q99)


# In[112]:


df['num']=df['num'].apply(lambda x: 1 if x>0 else 0)


# In[113]:


df['num'].value_counts()


# In[114]:


plt.figure(figsize=(7,6))
df['num'].value_counts().plot.pie(autopct='%1.1f%%',labels=['no heart disease','heart disease'])
plt.title('target distribution')
plt.savefig('visualizations/target_dist.png')
plt.show()
plt.close()


# In[115]:


for c in catcols:
    df[c]=df[c].astype('category').cat.codes


# In[116]:


plt.figure(figsize=(15,10))
for i,c in enumerate(numcols,1):
    plt.subplot(2,3,i)
    sns.histplot(df[c],kde=True)
    plt.title(f'distribution of {c}')
    plt.xlabel(c)
    plt.ylabel('freq')
plt.tight_layout()
plt.savefig('visualizations/num_histogram.png')
plt.show()
plt.close()


# In[117]:


plt.figure(figsize=(15,10))
for i,c in enumerate(numcols,1):
    plt.subplot(2,3,i)
    sns.boxplot(x='num',y=c,data=df)
    plt.title(f"{c} by heart disease")
    plt.xlabel('0=no 1=yes')
    plt.ylabel(c)
plt.tight_layout()
plt.savefig('visualizations/box_plots.png')
plt.show()
plt.close()


# In[118]:


scaler=RobustScaler()


# In[119]:


df[numcols]=scaler.fit_transform(df[numcols])


# In[120]:

joblib.dump(scaler,'models/scaler.pkl')


# In[121]:


plt.figure(figsize=(8,6))
corrm=df[numcols].corr()
sns.heatmap(corrm, annot=True)
plt.title('corr matrix')
plt.savefig('visualizations/corr_heatmap.png')
plt.show()
plt.close()


# In[122]:


X=df.drop('num',axis=1)
y=df['num']


# In[123]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[124]:


rf={
    'n_estimators':[100,200,300,500],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}
xgb={
    'n_estimators':[100,200,300,500],
    'max_depth':[3,5,7,9],
    'learning_rate':[0.01,0.05,0.1,0.2],
    'subsample':[0.7,0.8,0.9,1.0]
}
gb={
    'n_estimators':[100,200,300,500],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.05,0.1,0.2],
    'subsample':[0.7,0.8,0.9,1.0]
}


# In[125]:


rf_model=RandomForestClassifier(random_state=42)
rf_grid=GridSearchCV(rf_model,rf,cv=5,scoring='accuracy',n_jobs=-1)
rf_grid.fit(X_train,y_train)
rf_pred=rf_grid.predict(X_test)
rf_acc=accuracy_score(y_test,rf_pred)
rf_prec=precision_score(y_test,rf_pred)
rf_recall=recall_score(y_test,rf_pred)
rf_f1=f1_score(y_test,rf_pred)
print(rf_acc)
print(rf_prec)
print(rf_recall)
print(rf_f1)


# In[126]:


xgb_model=XGBClassifier(random_state=42,eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model,xgb,cv=5,scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train,y_train)
xgb_pred=xgb_grid.predict(X_test)
xgb_acc=accuracy_score(y_test,xgb_pred)
xgb_prec=precision_score(y_test,xgb_pred)
xgb_recall=recall_score(y_test,xgb_pred)
xgb_f1=f1_score(y_test,xgb_pred)
print(xgb_acc)
print(xgb_prec)
print(xgb_recall)
print(xgb_f1)


# In[127]:


gb_model=GradientBoostingClassifier(random_state=42)
gb_grid=GridSearchCV(gb_model,gb,cv=5,scoring='accuracy',n_jobs=-1)
gb_grid.fit(X_train,y_train)
gb_pred=gb_grid.predict(X_test)
gb_acc=accuracy_score(y_test,gb_pred)
gb_prec=precision_score(y_test,gb_pred)
gb_recall=recall_score(y_test,gb_pred)
gb_f1=f1_score(y_test, gb_pred)
print(gb_acc)
print(gb_prec)
print(gb_recall)
print(gb_f1)


# In[128]:


acc={
    'rf':rf_acc,
    'xgb':xgb_acc,
    'gb':gb_acc,
}


# In[129]:


best_model=max(acc,key=acc.get)
print(best_model, acc[best_model])


# In[130]:


if best_model=='rf':
    best_model_instance=rf_grid.best_estimator_
    metrics={'accuracy':rf_acc,'precision':rf_prec,'recall':rf_recall,'f1':rf_f1}
elif best_model=='xgb':
    best_model_instance=xgb_grid.best_estimator_
    metrics={'accuracy':xgb_acc,'precision':xgb_prec,'recall':xgb_recall,'f1':xgb_f1}
else:
    best_model_instance=gb_grid.best_estimator_
    metrics={'accuracy':gb_acc,'precision':gb_prec,'recall':gb_recall,'f1':gb_f1}
joblib.dump(best_model_instance,'models/best_model.pkl')


# In[131]:


with open('models/metrics.json','w') as f:
    json.dump(metrics,f)


# In[132]:


if best_model in ['rf','xgb','gb']:
    plt.figure(figsize=(10,6))
    ft_imp=best_model_instance.feature_importances_
    ft_names=X.columns
    imp_df=pd.DataFrame({'feature':ft_names,'importance':ft_imp})
    imp_df=imp_df.sort_values(by='importance',ascending=False)
    sns.barplot(x='importance',y='feature',data=imp_df)
    plt.title(f'feature importance for {best_model}')
    plt.savefig('visualizations/ft_importance.png')
    plt.show()
    plt.close()


# In[133]:


df.to_csv('data/heart_disease_uci_preprocessed.csv',index=False)

