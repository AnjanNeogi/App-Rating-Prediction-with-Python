#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Upload data set
data=pd.read_csv('googleplaystore.csv')


# In[3]:


data.shape


# In[4]:


#information of data set
data.info


# In[5]:


data.head()


# In[6]:


# to check null values in cell
data.isnull().sum()


# In[7]:


# drop null value
data.dropna(inplace=True)


# In[8]:


data.isnull().sum()


# # Convert size column to nummaric and to KB

# In[9]:


data['Size'].value_counts()


# In[10]:


def change(Size):
    if 'M' in Size:
        x=Size[:-1]
        x=float(x)*1000
        return x
    elif 'k' in Size:
        x=Size[:-1]
        x=float(x)
        return x
    else:
        return None


# In[11]:


data['Size']=data['Size'].map(change) # apply and map are same


# In[12]:


data.head()


# # Reviews is a numeric field that is loaded as a string field. Convert it to numeric (int/float).

# In[13]:


data['Reviews']=data['Reviews'].astype('int')   #astype convert the data type


# In[14]:


data.dtypes


# # Installs field is currently stored as string and has values like 1,000,000+
# # Treat 1,000,000+ as 1,000,000
# 
# # Remove ‘+’, ‘,’ from the field, convert it to integer

# In[15]:


data['Installs']=data['Installs'].str.replace('[+,]','') 
# this will replace both [+and,]in installs column with nothing so we put'' this.
# if we need to replace only + we will not put [] only +


# In[16]:


data.head()


# In[17]:


# Price field is a string and has $ symbol. Remove "$" sign, and convert it to numeric.


# In[18]:


data['Price']=data['Price'].str.replace('$','')


# In[19]:


data.head()


# In[20]:


data.dtypes


# In[21]:


data['Price']=data['Price'].astype('float')


# In[22]:


data.dtypes


# # Sanity checks

# In[23]:


#Average rating should be between 1 and 5 as only these values are allowed on the play store. 
#Drop the rows that have a value outside this range.


# In[24]:


# we need to check if there is any data greater than 5
data[data['Rating']>5]


# In[25]:


# we need to check if there is any data less than 1
data[data['Rating']<1]


# #Reviews should not be more than installs as only those who installed can review the app. 
# #If there are any such records, drop them.

# In[26]:


#to convert installs to int
data['Installs']=data['Installs'].astype('int')


# In[27]:


data.dtypes


# In[28]:


#to check reviews are greater than installs, if yes drop
data.drop(data[data['Installs']<data['Reviews']].index, inplace=True)


# For free apps (type = “Free”), the price should not be >0. Drop any such rows

# In[29]:


data.drop(data[(data['Type']=='Free') & (data['Price']>0)].index, inplace= True)


# # Performing univariate analysis

# In[30]:


# Boxplot for Price
# Are there any outliers? Think about the price of usual apps on Play Store


# In[31]:


sns.set(rc={'figure.figsize':(12,8)})#for graph size


# In[32]:


sns.boxplot(x='Price',data=data)


# In[ ]:





# In[33]:


#Boxplot for Reviews

#Are there any apps with very high number of reviews? Do the values seem right?


# In[34]:


sns.boxplot(data['Reviews'])


# In[35]:


sns.boxplot(data['Rating'])


# In[36]:


sns.boxplot(x="Rating", y="Category", data=data)


# In[37]:


# Histogram for Rating


# In[38]:


plt.hist(data['Rating'])


# In[39]:


# Histogram for Size


# In[40]:


plt.hist(data['Size'])


# # Outlier treatment

# In[41]:


#Price: From the box plot, it seems like there are some apps with very high price.
#A price of $200 for an application on the Play Store is very high and suspicious!

#Check out the records with very high price

#Is 200 indeed a high price?

#Drop these as most seem to be junk apps


# In[42]:


more = data.apply(lambda x : True
            if x['Price'] > 200 else False, axis = 1) 


# In[43]:


more_count = len(more[more == True].index)


# In[44]:


data.shape


# In[45]:


data.drop(data[data['Price']>200].index, inplace=True)


# In[46]:


data.shape


# In[47]:


#Reviews: Very few apps have very high number of reviews. 
#These are all star apps that don’t help with the analysis and, in fact, will skew it. 
#Drop records having more than 2 million reviews.


# In[48]:


data.drop(data[data['Reviews']>2000000].index, inplace=True)


# In[49]:


data.shape #reviews more than 2 million are droped.


# In[50]:


#Installs:  There seems to be some outliers in this field too. 
#Apps having very high number of installs should be dropped from the analysis.

#Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99

#Decide a threshold as cutoff for outlier and drop records having values more than that


# In[51]:


sns.boxplot(x='Installs',data=data) #boxplot for outliers fo installs.


# In[52]:


#a quantile is where a sample is divided into equal-sized, adjacent, subgroups
data.quantile([.1, .25, .5, .70, .90, .95, .99], axis = 0)


# In[53]:


# dropping more than 10000000 Installs value
data.drop(data[data['Installs']>10000000].index, inplace=True)


# In[54]:


data.shape


# # Bivariate analysis

# In[55]:


#Make scatter plot/joinplot for Rating vs. Price


# In[56]:


sns.scatterplot(x='Rating',y='Price',data=data)


# #from above plot we find that rating is more for paid apps.

# In[57]:


# Make scatter plot/joinplot for Rating vs. Size


# In[58]:


sns.scatterplot(x='Rating',y='Size',data=data)


# higher the app size better is rating

# In[59]:


#Make scatter plot/joinplot for Rating vs. Reviews


# In[60]:


sns.scatterplot(x='Rating',y='Reviews',data=data)


# More reviews have higher ratting

# In[61]:


#Make boxplot for Rating vs. Content Rating


# In[62]:


sns.boxplot(x='Rating',y='Content Rating',data=data)


# App for Everyone has more bad raing and app for Adult only 18+ has good ratting. 

# In[63]:


#Make boxplot for Rating vs. Category


# In[64]:


sns.boxplot(x='Rating',y='Category',data=data)


# All most all has good raitting

# # Data preprocessing

# In[65]:


# create a copy of the dataframe to make all the edits. Name it inp1


# In[66]:


inp1=data


# In[67]:


inp1.head()


# In[68]:


inp1.skew()


# In[69]:


reviewskew = np.log1p(inp1['Reviews']) #Apply log transformation to reduce skew
inp1['Reviews'] = reviewskew


# In[70]:


reviewskew.skew()


# In[71]:


installsskew = np.log1p(inp1['Installs'])
inp1['Installs']


# In[72]:


installsskew.skew()


# In[ ]:





# In[73]:


inp1.head()


# In[74]:


#Drop columns App, Last Updated, Current Ver, and Android Ver. These variables are not useful for our task.


# In[75]:


inp1.drop(['App','Last Updated','Current Ver','Android Ver','Type'],axis=1,inplace=True)


# In[76]:


inp1.head()


# In[77]:


#Get dummy columns for Category, Genres, and Content Rating.
#This needs to be done as the models do not understand categorical data, and all data should be numeric.
#Dummy encoding is one way to convert character fields to numeric. 
#Name of dataframe should be inp2.


# In[78]:


inp2 = inp1


# In[79]:


inp2.head()


# In[80]:


# Dummy encoding is one way to convert character fields to numeric for 


# In[81]:


# Dummy encoding Category


# In[82]:


#get unique values in Column "Category"
inp2.Category.unique()


# In[ ]:





# In[83]:


#this is second way to Dummy encoding
inp2.Category = pd.Categorical(inp2.Category)

x = inp2[['Category']]
del inp2['Category']

dummies = pd.get_dummies(x, prefix = 'Category')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[84]:


# Dummy encoding Genres


# In[85]:


#get unique values in Column "Genres"
inp2["Genres"].unique()


# => Since, There are too many categories under Genres. Hence, we will try to reduce some categories which have very few samples under them and put them under one new common category i.e. "Other"

# In[ ]:





# In[86]:


lists = []
for i in inp2.Genres.value_counts().index:
    if inp2.Genres.value_counts()[i]<20:
        lists.append(i)
inp2.Genres = ['Other' if i in lists else i for i in inp2.Genres] 


# In[87]:


inp2["Genres"].unique()


# In[88]:


inp2.Genres = pd.Categorical(inp2['Genres'])
x = inp2[["Genres"]]
del inp2['Genres']
dummies = pd.get_dummies(x, prefix = 'Genres')
inp2 = pd.concat([inp2,dummies], axis=1)


# In[89]:


inp2.head()


# In[90]:


# Dummy encoding "Content Rating"


# In[91]:


#get unique values in Column "Content Rating"
inp2["Content Rating"].unique()


# In[92]:


inp2['Content Rating'] = pd.Categorical(inp2['Content Rating'])

x = inp2[['Content Rating']]
del inp2['Content Rating']

dummies = pd.get_dummies(x, prefix = 'Content Rating')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[93]:


inp2.shape


# # Train test split  and apply 70-30 split. Name the new dataframes df_train and df_test.

# In[94]:


from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse


# In[ ]:





# In[95]:


inp2.isnull().sum()


# In[96]:


inp2.dropna(inplace=True)


# In[97]:


inp2.isnull().sum()


# In[99]:


d1 = inp2
X = d1.drop('Rating',axis=1)
y = d1['Rating']

Xtrain, Xtest, ytrain, ytest = tts(X,y, test_size=0.3, random_state=5)


# # Model building

# In[100]:


reg_all = LR()
reg_all.fit(Xtrain,ytrain)


# In[101]:


R2_train = round(reg_all.score(Xtrain,ytrain),3)
print("The R2 value of the Training Set is : {}".format(R2_train))


# In[102]:


R2_test = round(reg_all.score(Xtest,ytest),3)
print("The R2 value of the Testing Set is : {}".format(R2_test))


# In[ ]:




