#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
A=pd.read_csv("D:/DS_Recordings/DataSets/Cars93.csv")


# In[2]:


A.head()


# In[5]:


A.isna().sum()


# In[6]:


from miss import replacer
replacer(A)


# In[7]:


A.isna().sum()


# In[9]:


A.corr()["Weight"]


# In[10]:


A[["Fuel.tank.capacity","Wheelbase","Width"]].corr()


# In[21]:



X=A[["Fuel.tank.capacity","Wheelbase","Width"]]
Y=A[["Weight"]]
from sklearn.linear_model import LinearRegression
lm=LinearRegression()

def model_creation(X,Y,mobj):
    
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=32)
    model=mobj.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    from sklearn.metrics import mean_absolute_error
    print(round(mean_absolute_error(ytest,pred),3))


# In[22]:


model_creation(X,Y,lm)


# In[24]:


from sklearn.linear_model import Ridge,Lasso
rr=Ridge(alpha=1)
model_creation(X,Y,rr)


# In[27]:



lr=Lasso(alpha=1)
model_creation(X,Y,lr)


# In[28]:


for i in range(1,100,1):
    rr = Ridge(alpha=i-0.01)
    model_creation(X,Y,rr)


# In[29]:


for i in range(1,100,1):
    rr = Ridge(alpha=i+0.01)
    model_creation(X,Y,rr)


# In[30]:


for i in range(1,100,1):
    rr = Ridge(alpha=i-0.0001)
    model_creation(X,Y,rr)


# In[31]:


for i in range(0,100,1):
    rr = Ridge(alpha=i+0.001)
    model_creation(X,Y,rr)
for i in range(0,100,1):
    rr = Ridge(alpha=i-0.1)
    model_creation(X,Y,rr)


# RIDGE --> PENALTY(ALL) --> {SQUARE(COEF)} --> ERROR-- LASSO --> PENALTY(FEW) --> {ABSOLUTES(COEF)} --> ERROR --> | UNNECESARY COLUMNS WILL GET REMOVE

# #    # GridSerachCV 
# 

# In[32]:


x = 1               
W = []     #list of alpha value >0
for i in range(0,1000,1):
    x = x + 0.001
    W.append(round(x,3))   


# In[33]:


x = 1
W1 = []    #list of alpha value <0
for i in range(0,1000,1):
    x = x - 0.001
    W1.append(round(x,3))
    


# In[34]:


W.extend(W1)


# In[35]:


W


# In[37]:


from sklearn.model_selection import GridSearchCV
tp = {"alpha":W}
rr = Ridge()
cv = GridSearchCV(rr,tp,scoring="neg_mean_squared_error",cv=3)# tp tuning paramenter cv cross validation fold 3
cvmodel = cv.fit(xtrain,ytrain)


# In[38]:


cvmodel.best_params_   #will g


# In[ ]:




