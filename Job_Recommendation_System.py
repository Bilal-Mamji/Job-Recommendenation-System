#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from IPython.display import clear_output


# In[2]:


# jobs = pd.read_csv('Combined_Jobs_Final.csv')
jobs = pd.read_csv('modified.csv')
experience = pd.read_csv('Experience.csv')
job_view = pd.read_csv('Job_Views.csv')
position = pd.read_csv('Positions_Of_Interest.csv')
# modified = pd.read_csv('job_data.csv')


# In[3]:


jobs.head() 


# In[4]:


# Job.ID
# Status
# Slug
# Title
# Position
# Company
# City
# Industry
# Job.Description
# Employment.Type
# Education.Required

jobs = jobs[['Job.ID','Title','Position','Company','City','Industry']]


# In[5]:


data=jobs
jobs.head()


# In[6]:


nan_city = data[pd.isnull(data['City'])]
print(nan_city.shape)


# In[7]:


nan_city.groupby(['Company'])['City'].count() 


# In[8]:


data['Company'] = data['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')

data.loc[data.Company == 'CHI Payment Systems', 'City'] = 'Illinois'
data.loc[data.Company == 'Academic Year In America', 'City'] = 'Stamford'
data.loc[data.Company == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'
data.loc[data.Company == 'Driveline Retail', 'City'] = 'Coppell'
data.loc[data.Company == 'Educational Testing Services', 'City'] = 'New Jersey'
data.loc[data.Company == 'Genesis Health System', 'City'] = 'Davennport'
data.loc[data.Company == 'Home Instead Senior Care', 'City'] = 'Nebraska'
data.loc[data.Company == 'St. Francis Hospital', 'City'] = 'New York'
data.loc[data.Company == 'Volvo Group', 'City'] = 'Washington'
data.loc[data.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'


# In[9]:


data.isnull().sum()


# In[10]:


# data['Employment.Type']=data['Employment.Type'].fillna('Full-Time/Part-Time')
data['Company']=data['Company'].fillna(" ")
data['Industry']=data['Industry'].fillna(" ")
# data['Job.Description']=data['Job.Description'].fillna(" ")
# data['Education.Required']=data['Education.Required'].fillna(" ")

data.isnull().sum()


# In[11]:


# Job.ID
# Status
# Slug
# Title
# Position
# Company
# City
# Industry
# Job.Description
# Salary
# Employment.Type
# Education.Required
# data['Status'].apply(lambda x:x.split())
# data['Slug'].apply(lambda x:x.split())
data['Title'].apply(lambda x:x.split())
data['Position'].apply(lambda x:x.split())
data['Company'].apply(lambda x:x.split())
data['City'].apply(lambda x:x.split())
data['Industry'].apply(lambda x:x.split())
# data['Job.Description'].apply(lambda x:x.split())
# data['Employment.Type'].apply(lambda x:x.split())
# data['Education.Required'].apply(lambda x:x.split())


# In[12]:


# data['Status'].apply(lambda x:[i.replace(" ","") for i in x])
# data['Slug'].apply(lambda x:[i.replace(" ","") for i in x])
data['Title'].apply(lambda x:[i.replace(" ","") for i in x])
data['Position'].apply(lambda x:[i.replace(" ","") for i in x])
data['Company'].apply(lambda x:[i.replace(" ","") for i in x])
data['City'].apply(lambda x:[i.replace(" ","") for i in x])
data['Industry'].apply(lambda x:[i.replace(" ","") for i in x])
# data['Job.Description'].apply(lambda x:[i.replace(" ","") for i in x])
# data['Employment.Type'].apply(lambda x:[i.replace(" ","") for i in x])
# data['Education.Required'].apply(lambda x:[i.replace(" ","") for i in x])


# In[13]:


data['tag']=data['Title'].str.replace(" "," ")


# In[14]:


data.head()


# In[15]:


data['tag']=data['Title']+" "+data['Position']+" "+data['Company']+" "+data['City']


# In[16]:


data['tag'][0]


# In[17]:


new_data=data[['Job.ID','tag']]


# In[18]:


new_data
new_data['tag'][0]


# In[19]:


new_data['tag']=new_data['tag'].str.replace('[^a-zA-Z \n\.]'," ")


# In[20]:


new_data['tag'][0]


# In[21]:


new_data['tag']=new_data['tag'].str.replace('[\n]'," ")


# In[22]:


new_data['tag']=new_data['tag'].str.replace("\s+"," ")#\s: to remove extra spaces
new_data['tag'][0]


# In[23]:


new_data['tag']=new_data['tag'].str.lower()
new_data['tag'][0]


# In[24]:


get_ipython().system('pip install nltk')


# In[25]:


import nltk
nltk.download('stopwords')


# In[26]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# stop = stopwords.words('english')


# In[27]:


st =  PorterStemmer()


# In[28]:


def stem(text):
    l=[]
    for i in text.split():
        l.append(st.stem(i))
    return " ".join(l)


# In[29]:


new_data.head()


# In[30]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# #from sklearn.feature_extraction.text import CountVectorizer

# tfidf_vectorizer = TfidfVectorizer()

# tfidf_jobid = tfidf_vectorizer.fit_transform((new_data['tag'])).toarray()


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


cv = TfidfVectorizer()


# In[33]:


vectors = cv.fit_transform(new_data['tag'])


# In[34]:


#cv.get_feature_names()


# ## User Table

# In[35]:


cv1 = pd.read_csv('Experience.csv')
cv2 = pd.read_csv('Job_Views.csv')
cv3 = pd.read_csv('Positions_Of_Interest.csv')


# In[36]:


cv1=cv1[['Applicant.ID','Position.Name','Employer.Name','City','State.Name']]
cv1.head()


# In[37]:


cv2=cv2[['Applicant.ID','Title','Position','City','State.Name']]
cv2.head()


# In[38]:


cv3=cv3[['Applicant.ID','Position.Of.Interest']]
cv3.head()


# In[39]:


orig=pd.concat([cv1,cv2,cv3],sort=False)
#orig = orig.dropna(subset=['Position.Name'])


# In[40]:


orig


# In[41]:


view=pd.read_csv('Job_Views.csv')


# In[42]:


view.head()


# In[43]:


# Job.ID
# Status
# Slug
# Title
# Position
# Company
# City
# Industry
# Job.Description
# Employment.Type
# Education.Required
view=view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]


# In[44]:


view["tag"]=view["Position"].map(str) + "  " + view["Company"] +"  "+ view["City"]


# In[45]:


view['tag']=view['tag'].str.replace('[^a-zA-Z \n\.]',"")
view['tag']=view['tag'].str.lower()


# In[46]:


view = view[['Applicant.ID','tag']]


# In[47]:


view.head()


# In[48]:


exp=pd.read_csv("Experience.csv")


# In[49]:


exp.head()


# In[50]:


exp = exp[['Applicant.ID','Position.Name']]

#cleaning the text
exp['Position.Name'] = exp['Position.Name'].str.replace('[^a-zA-Z \n\.]',"")
exp['Position.Name'] = exp['Position.Name'].str.lower()


# In[51]:


#exp.head()


# In[52]:


exp=exp.sort_values(by='Applicant.ID')
exp=exp.fillna(" ")

    


# In[53]:


exp.head()


# In[54]:


exp = exp.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()
exp.head(10)


# In[55]:


pos = pd.read_csv('Positions_Of_Interest.csv')


# In[56]:


pos.head()


# In[57]:


pos['Position.Of.Interest']=pos['Position.Of.Interest'].str.replace('[^a-zA-z \n\.]',"")
pos['Position.Of.Interest']=pos['Position.Of.Interest'].str.lower()
pos = pos.fillna(" ")


# In[58]:


pos=pos[["Applicant.ID","Position.Of.Interest"]]


# In[59]:


pos.head()


# In[60]:


pos=pos.sort_values(by='Applicant.ID')


# In[61]:


pos=pos.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()


# In[62]:


pos['Position.Of.Interest'][2]


# In[63]:


merge=view.merge(exp, left_on='Applicant.ID', right_on='Applicant.ID',how='outer')


# In[64]:


merge=merge.sort_values(by='Applicant.ID')


# In[65]:


merge=merge.fillna(' ')
merge.head()


# In[66]:


merge=merge.merge(pos,how='outer')
merge=merge.fillna(' ')


# In[67]:


merge=merge.sort_values(by='Applicant.ID')


# In[68]:


merge.head()


# In[69]:


merge['tags']=merge["tag"].map(str)+" "+merge["Position.Name"].map(str)+" "+merge["Position.Of.Interest"].map(str)


# In[70]:


merge.head(5)


# In[71]:


merge=merge[['Applicant.ID','tags']]
merge.head(5)


# In[72]:


merge.head(5)


# In[73]:


merge['tags'] = merge['tags'].str.replace('[^a-zA-Z \n\.]',"")
merge['tags'] = merge['tags'].str.lower()
merge=merge.sort_values(by='Applicant.ID')


# In[74]:


merge.to_csv('vectorization.csv')
merge.head()


# In[75]:


merge['tags'][3]
merge = merge.groupby('Applicant.ID', sort=True)['tags'].apply(' '.join).reset_index()


# In[76]:


merge=merge.reset_index(drop=True)
merge.head()


# # ENTER APPLICANT ID (CV / Resume ID)

# In[77]:


yourinput = int(input("Enter Applicant ID: "))
while yourinput not in orig['Applicant.ID'].values:
    print("Desired Applicant ID Not Found")
    clear_output(wait=True)
    yourinput = int(input("Enter Applicant ID: "))

print("\n\nApplicant ID: ", yourinput)


# In[78]:


#u=2
import numpy as np
index = np.where(merge['Applicant.ID'] == yourinput)[0][0]
user = merge.iloc[[index]]
user


# In[79]:


from sklearn.metrics.pairwise import cosine_similarity


# In[80]:


user_vector = cv.transform(user['tags'])


# In[81]:


output = map(lambda x: cosine_similarity(user_vector, x),vectors)


# In[82]:


output = list(output)


# In[83]:


top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:10]
recommendation = pd.DataFrame(columns = ['ApplicantID', 'JobID'])
count = 0
for i in top:
    recommendation.at[count, 'ApplicantID'] = yourinput
    recommendation.at[count, 'JobID'] = new_data['Job.ID'][i]
    count += 1


# In[84]:


import sys

#print(sys.version)


# In[85]:


recommendation


# In[86]:


nearestjobs = recommendation['JobID']
job_description = pd.DataFrame(columns = ['JobID','tag'])
for i in nearestjobs:
    index = np.where(new_data['Job.ID'] == i)[0][0]    
    job_description.at[count, 'JobID'] = i
    job_description.at[count, 'tag'] = new_data['tag'][index]
    count += 1


# In[87]:


s=job_description.sort_values(by="JobID")


# In[88]:


s


# In[89]:


r=pd.read_csv('Combined_Jobs_Final.csv')


# In[90]:


rec =s['JobID'].values


# In[91]:


rec


# In[92]:


for i in rec:
    print(i)


# In[93]:


applicant=orig[orig['Applicant.ID'] == yourinput]


# In[94]:


file1=cv1[cv1['Applicant.ID'] == yourinput]
file2=cv2[cv2['Applicant.ID'] == yourinput]
file3=cv3[cv3['Applicant.ID'] == yourinput]


# In[95]:


file1


# In[96]:


file2


# In[97]:


file3


# In[ ]:





# In[98]:


final_id = r[r['Job.ID'].isin(rec)]


# In[99]:


final_id


# # THE END

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[100]:


#u=pd.read_csv('merg.csv')


# In[101]:


# u.drop_duplicates(subset='Applicant.ID', keep='first', inplace=True)
#applicant=u[u['Applicant.ID'] == 2]


# In[102]:


#applicant


# In[103]:


#cv1 = pd.read_csv('Experience.csv')
#cv2 = pd.read_csv('Job_Views.csv')
#cv3 = pd.read_csv('Positions_Of_Interest.csv')


# In[104]:


#cv1=cv1[['Applicant.ID','Position.Name','Employer.Name','City','State.Name']]
#cv1.head()


# In[105]:


#cv2=cv2[['Applicant.ID','City','State.Name']]
#cv2.head()


# In[106]:


#cv3=cv3[['Applicant.ID','Position.Of.Interest']]
#cv3.head()


# In[107]:


#orig=pd.concat([cv1,cv2,cv3],sort=False)


# In[108]:


#orig=pd.concat([cv1,cv2,cv3],sort=False)
#orig = orig.dropna(subset=['Position.Name'])


# In[109]:


#orig.isnull().sum()


# In[110]:


#orig.head()


# In[111]:


#applicant=orig[orig['Applicant.ID'] == yourinput]


# In[112]:


#applicant


# In[113]:


#orig2=orig[['Applicant.ID']]
#df_sorted = orig2.sort_values(by='Applicant.ID')
#orig2


# In[ ]:




