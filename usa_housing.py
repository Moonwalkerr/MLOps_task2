#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df= pd.read_csv('USA_Housing.csv')


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


df.columns


# In[6]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[7]:


y=df['Price']


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:


model=LinearRegression()
model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


plt.scatter(y_test[0:30],y_pred[0:30])


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[10]:


from sklearn.externals import joblib


# In[ ]:


joblib.dump(model,'USA1.pk1')


# In[11]:


model=joblib.load('USA1.pk1')


# In[12]:


model.predict(X_test)


# In[ ]:





# In[ ]:




