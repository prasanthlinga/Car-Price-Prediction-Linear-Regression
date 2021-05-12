#!/usr/bin/env python
# coding: utf-8

# In[479]:


import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np


# In[480]:


data=pd.read_csv("D:\Kaggle\Linear Regression RFE(Recursive feature eimination)- CAR price prediction\CarPrice_Assignment.csv",delimiter=',')
data.head()  #prints first 5 rows
#print(data.head())
print(data.shape,data.size,data.ndim)
#data.shape() will raise error
#data.to_numpy()


# In[ ]:





# In[481]:


#print(data.describe()) # gives mean, stdard deviation ,min,25percentile,50 percentile,75 percentile and max values of numerical data.
# it excludes the cloumns which contains the string object.

#data.describe(percentiles=[0.2,0.4,0.6,.8])   we can give percnetiles as our wish
include=['float','int','object']
data.describe(percentiles=[0.2,0.4,0.6,0.8],include='all') #returns NA for string objects   top is most common value


# In[482]:


data.describe(include='object')


# In[483]:


data.info() 
#data.info(verbose=False) # verbose is used for whether t print full summary or not. false means it won't print


# # Data Cleaning and Preparation

# In[484]:


#Splitting company name from car name column
company_name=data['CarName'].apply(lambda x:x.split(' ')[0]) #Split method retuens a list of strings after 
#breaking the given string by secified separator, here we used space as a separator
#company_name.dtype   data type is object i.e string
#print(company_name)
#data
#company_name
#company_name.insert(3,"CompanyName",company_name,allow_duplicates=False)    # error  series object has no attribute insert
data.insert(3,"CompanyName",company_name,allow_duplicates=True)
data.drop(labels=['CarName'],axis=1,inplace=True)  #axis=1 means column and 0 means row  ..inpalce True means makechanges in original dataframe
data


# In[485]:


data.head()


# In[486]:


data.CompanyName.unique() #returns unique values of data in CompanyName column


# In[487]:


data.CompanyName.unique().size


# # Fixing invalid values
# 
#  maxda=mazda
#  nissan=Nissan
#  porsche=porcshce
#  toyota=toyouta
#  vokswagen=volkswagen=vw

# In[488]:


data.CompanyName=data.CompanyName.str.lower() # converts all letters in CompanyName column into lowercase letters
data.CompanyName.unique()


# In[489]:


data.CompanyName.replace("maxda","mazda",inplace=True)
#data.CompanyName.str.replace("Nissan","nissan")
data.CompanyName.replace("porcshce","porsche",inplace=True)
data.CompanyName.replace("toyouta","toyota",inplace=True)
data.CompanyName.replace("vokswagen","volkswagen",inplace=True)
data.CompanyName.replace("vw","volkswagen",inplace=True)


# In[490]:


data.CompanyName[150]


# In[491]:


data.CompanyName.unique()


# In[492]:


data.loc[data.duplicated()]  #checking for duplicates by name of columns


# In[493]:


data.columns


# # Visualizing the data
# 

# In[494]:


plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title("car price distribution")
sns.distplot(data.price)
#plt.show()
plt.subplot(1,2,2)  # will come in same row
plt.title("car price spread")
sns.boxplot(y=data.price)
plt.show()


# In[495]:


#data[price].describe(percentiles=[0.25,0.5,0.75,0.85,0.9,1]) # raises error
data.price.describe(percentiles=[0.25,0.5,0.75,0.85,0.9,1])


# # 3.1 Visualizing Categorical Data
# - CompanyName
# - Symboling
# - fueltype
# - enginetype
# - carbody
# - doornumber
# - enginelocation
# - fuelsystem
# - cylindernumber
# - aspiration
# - drivewheel

# In[496]:


plt.figure(figsize=(25,6))
plt.subplot(1,3,1)
plt1=data.CompanyName.value_counts().plot(kind='bar')  #returns count of unique values  in dataframe.CompamnyNmae
plt1.set(xlabel="car company", ylabel="frequency of company")
plt.title("companies histogram")
plt.show()

plt.subplot(1,3,2)
plt2=data.fueltype.value_counts().plot(kind="bar")
plt2.set(xlabel="fuel type",ylabel="frequency")
plt.title("fueltypes histogram")
plt.show()

plt.subplot(1,3,3)
plt3=data.carbody.value_counts().plot(kind="bar")
plt3.set(xlabel="car body",ylabel="frequency") #plt3.title(xlabel="car body",ylabel="frequency") will raise an error
plt.title("car body histogram")
plt.show()


# # Inference for above
# - toyota is most favouered comapny
# - gas is used by most
# - sedan is the most frquent car body

# In[497]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt4=data.symboling.value_counts().plot(kind="bar")
plt4.set(xlabel="symboling", ylabel="no")
plt.title("symboling histogram")
plt.show()

plt.subplot(1,2,2)
#plt4.set(xlabel="symboling",ylabel="price")
#plt.title("symboling vs price")
sns.boxplot(x=data.symboling,y=data.price)
plt.show()


# # Inference for above
# - Symboling with 1 and 0 have more no.of rows
# - symboling with -2 have least no.of rows.
# - the price is more for symboling -1..the price is least for symboling 1
# - symboling with 3 has price range similar to symboling with -2
# 

# In[498]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title("engine type histogram")
plt5=data.enginetype.value_counts().plot(kind="bar")
plt5.set(xlabel="enginetype",ylabel="number")
plt.subplot(1,2,2)
sns.boxplot(data.enginetype,data.price)
plt.show()

plt6=pd.DataFrame(data.groupby(['enginetype'])["price"].mean().sort_values())
(plt6)
plt6.plot.bar(figsize=(8,6))
plt.title("engine type vs price")
plt.show()




# # Inference 
# - ohc enginetype is used by most and dohcv is used by ver few.
# - ohcv engine type has highest price, ohc and ohcf prices are almost similar.

# In[499]:


plt7=pd.DataFrame(data.groupby(["CompanyName"])["price"].mean().sort_values())
plt7
plt.figure(figsize=(20,6))
#plt.subplot(1,3,1)
plt7.plot.bar(figsize=(10,5))
plt.title("company name vs avg price")
plt.show()

plt8=pd.DataFrame(data.groupby(["fueltype"])["price"].mean().sort_values())
plt8
plt8.plot.bar(figsize=(10,5))
plt.title("fuel type vs avg price")
plt.show()


plt9=pd.DataFrame(data.groupby(['carbody'])["price"].mean().sort_values())
plt9
plt9.plot.bar(figsize=(10,5))
plt.title("carbody vs avg price")
plt.show()


# # Inference for above
# - buick and jaguar has high avg price
# - diesel has avg price than gas
# - convertible and hardtop has high avg price 

# In[500]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.title("DoorNumber histogram")
plt9=data.doornumber.value_counts().plot(kind="bar")    #sns.countplot(data.doornumber)  # same result
#sns.displot(plt9)
plt.show()
plt.subplot(1,2,2)
plt.title("doornumber vs price")
sns.boxplot(x=data.doornumber,y=data.price)
plt.show()

plt.subplot(1,2,1)
plt.title("aspiration histogram")
sns.countplot(data.aspiration)       # countplot is similar to value_counts function in pandas 
plt.show()

plt.subplot(1,2,2)
plt.title("aspiratioon vs price ")
sns.boxplot(x=data.aspiration,y=data.price)
plt.show()






# # Inference for above
# - door number variable is not effecting price much 
# - aspriration for turbo has some price range than std

# In[501]:


plt.figure(figsize=(10,6))
sns.countplot(data.enginelocation)
plt.show()
sns.boxplot(x=data.enginelocation,y=data.price)
plt.show()


plt.figure(figsize=(10,6))
sns.countplot(data.cylindernumber)
plt.show()
sns.boxplot(x=data.cylindernumber,y=data.price)
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data.fuelsystem)
plt.show()
sns.boxplot(x=data.fuelsystem,y=data.price)
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data.drivewheel)
plt.show()
sns.boxplot(x=data.drivewheel,y=data.price)
plt.show()


# # Inference for above
# - enginelocation variable rear has high price still used by very few.
# - cylinder with 4 5 6 are common but the price for 8 cylinders cars is high
# - mpfi and 2bbl  fuel system are most common type ...mpfi and idi having highest price range.
# - fwd is preferred drive wheel but price is low for that..the price of rwd is  ery high compafred to fewd and 4wd
# 

# # Visualizing Numerical Data

# In[502]:


# learn how to write code  to put xlabel ylabel by using a function
def plottingfn(i,j):
 #plt.xlabel()
 #plt.ylabel(j)
 plt.scatter(i,j)
 #plt.xlabel()
 plt.show()


plottingfn(data.carlength,data.price)
plottingfn(data.carwidth,data.price)
plottingfn(data.carheight,data.price)
plottingfn(data.curbweight,data.price)


#plt.figure(figsize=(10,6))
#plottingfn(data.carlength,data.price)
#plt.show()

#plt.scatter(x=data.carwidth,y=data.price)
#plt.show()

#plt.scatter(x=data.carheight,y=data.price)
#plt.show()

#plt.scatter(x=data.curbweight,y=data.price)
#plt.show()





# # inference for above  
# - cralength carwidth curbweight show positive correlation with price
# - car height does not have positive correlation with price data points are randomly distributed.
# 

# In[503]:


import matplotlib
print(matplotlib.__version__)
def pp(x,y,z):
    sns.pairplot(data,x_vars=[x,y,z],y_vars="price",height=4,aspect=1,kind="scatter",diag_kind=None)  #use diag_kind=none 
                                                        # otheerwise first plot in below pairplots will not plot
    plt.show()
    
pp("enginesize","boreratio","stroke")
pp("compressionratio","horsepower","peakrpm")
pp("wheelbase","citympg","highwaympg")


# # Inference for above
# - enginesize, boreratio, horsepower, wheel base are in positive correlation with  price
# -  citympg and highwaympg are negative correlation with price
# 

# In[504]:


np.corrcoef(data.carlength,data.carwidth)   #correlation coefficient is used for deriving
                                          #the linear relationship b/w data points.


# # Deriving New features

# In[505]:


# why we are not using for loop even thpugh it is printing for all the records in table..how?
data["Fueleconomy"]=0.55*data["citympg"]+0.45*data["highwaympg"]
data["Fueleconomy"]


# In[506]:


#Binning the car companies based on average price of each company
data["price"]=data["price"].astype('int64')
data['price']  # data.price


# In[507]:


temp=data.copy()
print(temp)
table=(temp.groupby(["CompanyName"]))["price"].mean()
table


# In[508]:


#for i in table:
 #   print(table[i])
""" if  table[i] > 0.0 and table[i] < 20000.0:
        data["carsrange"]='medium'
    elif table[i]>20000:
        data["carsrange"]='Highend'
data"""


# In[509]:


print (table.reset_index())
temp=temp.merge(table.reset_index(),how='left',on='CompanyName')   #left outer join  uses keys from left frame only
# on may be column or index level names to join on..   must be present in both dataframes. 
# suffoxes can be manually given not only _x and _y  we can give whatever names we want.

temp


# In[510]:


bins=[0,10000,20000,40000]     # 0 to 10000 and 10000 to 20000 and 20000 to 40000
cars_bin=["budget","medium","highend"]  #bin labels must be one smaller than no.of bins.
data["Carsrange"]=pd.cut(temp['price_y'],bins,labels=cars_bin) # bin values into discrete intervals
# cut function is useful for converting continuos variable to a categorical variable.
data


# # Bivariate Analysis
# 

# In[511]:


plt.figure(figsize=(10,8))
plt.title("fuel economy vs price")
plt.scatter(data['Fueleconomy'],data.price)

plt.xlabel("fuel economy")
plt.ylabel("price")
plt.show()


# # Inference for above
# Fuel economy and price are negatively correlated.

# In[512]:


plt.figure(figsize=(10,8))

z=pd.DataFrame(data.groupby(["fuelsystem","drivewheel","Carsrange"])["price"].mean().unstack(fill_value=0))
print(z)

z.plot.bar()
plt.title("carsrange vs price")
plt.show()

#data.groupby(["Carsrange"])["price"].mean()


# # Inference for above
# highend cars uses rwd drive wheel and (idi and mpfi) as fuel system

# # List of significant variabes after visula analysis
#  - engine size
#  - boreratio
#  - horsepower
#  - wheelbase
#  - engine type  --  categorical
#  - car body  --  categorical
#  - fuel type  --  categorical
#  - aspiration  --  categorical
#  - #engine location 😊
#  - cylinder number  --  categorical
#  - #fuel system 😊
#  - drive wheel  --  categorical
#  - fuel economy
#  - car range  --  categorical
#  - car length
#  - carwidth
#  - curbweight
#  - #citympg 😊
#  - #highwaympg 😊  the above two are negatively correlated with price and did not consider.
#  
#  
# 

# In[513]:


data_lr=data[['price','fueltype','aspiration','carbody','drivewheel','wheelbase','curbweight','enginetype','cylindernumber',
              'enginesize','boreratio','horsepower','Fueleconomy','carlength','carwidth','Carsrange']]
data_lr


# In[514]:


sns.pairplot(data_lr)   # seaborn will show only numeric values
plt.show()


# # Dummy variables
#  -  for categorical variables, we need to create dummy variables
#  

# In[515]:


def dummies(x,y):
    temp=pd.get_dummies(y[x],drop_first=True)
    y=pd.concat([y,temp],axis=1)
    y.drop([x],axis=1,inplace=True)
    return y
    
    
    
"""temp=pd.get_dummies(data_lr['fueltype'],drop_first=True) #drop_first bool, default False
# to get k-1 dummies out of k categorical levels by removing the first level
temp   #if drop_firts is TRue , first column is dropped.  for k variables, we get k-1 dummy variables
df=pd.concat([data_lr,temp],axis=1) # axis =1 means concatenate along columns, and 0 means concatenate along rows
print(df)
df.drop('fueltype',axis=1,inplace=True) #  if axis is not menetioned i.e default =0(along rows)
#then it will raise error because fueltypeis not found in rows and inplace must be TRue tehn only fueltype column will be removed 
df"""


data_lr=dummies('fueltype',data_lr)
data_lr=dummies('aspiration',data_lr)
data_lr=dummies('carbody',data_lr)
data_lr=dummies('drivewheel',data_lr)
data_lr=dummies('enginetype',data_lr)
data_lr=dummies("cylindernumber",data_lr)
data_lr=dummies("Carsrange",data_lr)

data_lr



# # Train Test split and scaling
# 

# In[516]:


from sklearn.model_selection import train_test_split
np.random.seed(0)   # same output for each time.
df_train,df_test=train_test_split(data_lr,train_size=0.7,test_size=0.3,random_state=100)   #if random_state is not given i.e None 
#then everytime(for each execution) different values for train and test sets 
# if random_state = any integer  then the the order/list of values in test and train sets is fixed for that random state value
# if random_state=0  one order is fixed even if we execute multiple times
#  if random_state=1  different order from above will be produced
# if random state= 2,3,4,......different orders for each integers and the order is fixed for a particular intreger value if we run multiple times
print(df_train,df_test)


# In[517]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
print(scaler)
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','Fueleconomy','carlength','carwidth','price']
df_train[num_vars]=scaler.fit_transform(df_train[num_vars])
df_train


# In[518]:


df_train.describe()  # we can use percentiles and include parameter for all rows and few rows


# In[519]:


# correlation using heatmap
plt.figure(figsize=(20,20))
sns.heatmap(df_train)


# In[520]:


plt.figure(figsize=(20,20))
sns.heatmap(df_train.corr(),annot=True,cmap='cubehelix',fmt='0.2f')  # annot=true fills the box with corelation value and
#fmt=0.2 means upto 2 decimal points 


# In[521]:


# divide the data into x and y labels
y_train=df_train.pop("price")
X_train=df_train
print(y_train)  # if we run the cell again , we will get key error saying price is not found because price is already popped out from df_train when we run it for the first time.
#so second time when we run, there is no price attribute in df_train..... for that we need to run all cells once agin
#x_train


# In[522]:


X_train


# # Model Building
# 

# In[523]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
lm=LinearRegression()
a=lm.fit(X_train,y_train)
print(a)
print(a.coef_) # weight vector of shape(no.of targets, no.of features)=(1,30)   e here is 10 
print(a.intercept_)  #-0.006477981385304865   is bias
rfe = RFE(a, n_features_to_select=10)
print(rfe)
rfe.fit(X_train,y_train)
print(rfe)
print(rfe.support_)  # returns True for all selected variables and fase for not slected variables
rfe.ranking_  # returns rank 1 for all features which are true and 

# I did not understand what fetaures it selected


# In[524]:


a=list(zip(X_train.columns, rfe.support_,rfe.ranking_))  
(a)


# In[525]:


print(X_train.columns[rfe.ranking_])
X_train_rfe=X_train[X_train.columns[rfe.support_]]
X_train_rfe


# # building model using stats model  for the detailed statistics

# In[526]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor    #https://www.statsmodels.org/stable/api.html


# In[527]:


def build_model(X,y):
    X=sm.add_constant(X)
    print(X)
    lm=sm.OLS(y,X)
    a=lm.fit()
    print(a.params)   # constant is bias term and remaining are weights for each feature
    print(a.summary())
    return X
X_train_new=build_model(X_train_rfe,y_train)
print(X_train_new)


# # Inference for above
# - as p value > 0.05, then there is strong veidence for null hypothesis . so we will not consider those variable whose p >0.0.5
# - twelve and fuel economy has p >0.05 , so drop those two independent variables

# In[528]:


X_train_new=X_train_new.drop(columns=["Fueleconomy","twelve"])
X_train_new


# In[529]:


X_train_new=build_model(X_train_new,y_train)


# In[530]:


# VIF detects multicollinerairty in regression analysis
# here multicollinearoty refers to correlation b/w independent variables in data  which affect the regression results
def Detect_Multicollinearity(X):
    vif=pd.DataFrame()
    vif['features']=X.columns
    print(X.columns,vif['features'])
    vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(0,X.shape[1])]  
    vif['VIF']=round(vif['VIF'],3)# upto 3 decimal points
    vif=vif.sort_values(['VIF'],ascending=[False])
    return vif
Detect_Multicollinearity(X_train_new)


# In[531]:


Detect_Multicollinearity(X_train_new)
X_train_new=X_train_new.drop(columns=['curbweight'])  # dropping curb weight because of high vif 
X_train_new


# In[532]:


X_train_new=build_model(X_train_new,y_train)


# In[533]:


Detect_Multicollinearity(X_train_new)


# In[534]:


# Drop sedan becuase of high vif value
X_train_new=X_train_new.drop('sedan', axis=1)
X_train_new


# In[535]:


Detect_Multicollinearity(X_train_new)


# In[536]:


X_train_new=build_model(X_train_new,y_train)


# In[537]:


#drop wagon because of high p value   >>> 0.05
X_train_new=X_train_new.drop(columns=['wagon'])
X_train_new


# In[538]:


Detect_Multicollinearity(X_train_new)


# # Residual analysis of Model

# In[539]:


lm=sm.OLS(y_train,X_train_new).fit()
print(lm)
y_train_price=lm.predict(X_train_new)
print(y_train_price)


# In[540]:


plt.figure()
#print(y_train-y_train_price)
sns.distplot((y_train-y_train_price),bins=20)  #20 equal intervals
plt.xlabel('errors')
plt.show()


# In[541]:


# errors are normally distributed and the assumption of linear modelling seems to be fulfilled


# # Prediction and Evaluation
# 

# In[542]:


print(df_test)


# In[543]:


X_train_new


# In[544]:


#X_train_new=X_train_new.drop(columns=['const'])
#X_train_new

#df_test=scaler.fit_transform(df_test)
#df_test


# In[545]:


#X_test=df_test[X_train_new(columns=['horsepower','carwidth','hatchback','dohcv','highend'])]
#print(X_test)
#X_test_new=scaler.fit_transform(X_test)#(columns=['horsepower','carwidth','hatchback','dohcv','highend']))
#X_test_new

num_vars=['horsepower','carwidth','hatchback','dohcv','highend','price']
df_test[num_vars]=scaler.fit_transform(df_test[num_vars])
df_test


# In[546]:


y_test=df_test.pop('price')
print(y_test)


# In[547]:


#X_test_new=scaler.fit_transform(X_test)
#print(X_test_new)

X_test=df_test
print(X_test)


# In[548]:


X_train_new=X_train_new.drop("const",axis=1)    #  (columns=['const'])
X_train_new


# In[549]:


X_test_new=X_test[X_train_new.columns]
X_test_new


# In[550]:


X_test_new=sm.add_constant(X_test_new)
X_test_new


# In[551]:


y_pred=lm.predict(X_test_new)
print(y_pred)


# In[552]:


print(y_pred,y_test)


# In[553]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)    # r2_score(y_pred,y_test)  will give 0.86 which is false  beacuse
#the syntax of r2 score function contains first argument as true and second argument as predicted value


# In[554]:


plt.figure(figsize=(8,8))
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.scatter(y_pred,y_test)
#plt.plot(y_pred,y_test,)#color='red')


# In[555]:


print(lm.summary())


# In[556]:


# all p values are less than 0.05  

