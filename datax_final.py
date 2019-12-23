
# coding: utf-8

# In[1]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
from bs4 import BeautifulSoup


# In[2]:



browser = webdriver.Chrome("/Users/shray/Desktop/Drivers/chromedriver")

browser.get('https://www.youtube.com/results?search_query=cybersecurity'+'&sp=EgIQAw%253D%253D')


# In[3]:


def scroller(): 
    #browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    browser.execute_script('''document.documentElement.scrollBy(10000,10000) ''')


# In[6]:


for a in range(0,20):
    scroller()


# In[7]:


html = browser.page_source
soup = BeautifulSoup(html,'html.parser')


# In[8]:


a = soup.findAll('div', attrs={'class':'style-scope ytd-playlist-renderer'})
f= []
for b in a:
    f.append(b.a['href'])


# In[9]:


len(f)


# In[12]:


import sys


# In[13]:


final_list = []
for a in f:
    result = a.find('list=')
    final_list.append(a[result+5:])


# In[14]:


final_list[2]


# In[12]:


import os

for i in range(len(final_list)):
    myCmd = "youtube-dl -j --flat-playlist 'https://www.youtube.com/playlist?list="+str(final_list[i])+"' > /Users/shray/Desktop/data/log"+str(i)+".txt"
    os.system(myCmd)


# In[3]:


import glob, os
os.chdir("/Users/shray/Desktop/data/")
txt_file =[]
for file in glob.glob("*.txt"):
    txt_file.append(file)


# In[4]:


import json
full_list =[]
for file in txt_file:
    F = open("/Users/shray/Desktop/data/"+str(file),"r") 
    lista = F.read().split('\n')
    for element in lista[:-1]: 
        full_list.append(json.loads(element))


# In[5]:


len(txt_file)


# In[6]:


len(full_list)


# In[7]:


import requests
import re
import youtube_dl

def captions_test02(element):
    tempo ={}
    ide = str(element['url'])
    url = 'https://www.youtube.com/watch?v='+str(ide)
    ydl = youtube_dl.YoutubeDL({'writesubtitles': True, 'allsubtitles': True, 'writeautomaticsub': True})
    res = ydl.extract_info(url, download=False)
    if res['requested_subtitles'] and res['requested_subtitles']['en']:
        response = requests.get(res['requested_subtitles']['en']['url'], stream=True)                
        new = re.sub(r'\d{2}\W\d{2}\W\d{2}\W\d{3}\s\W{3}\s\d{2}\W\d{2}\W\d{2}\W\d{3}','',response.text)
        tempo['id'] = ide
        tempo['transcript'] = new
        return tempo


# In[9]:


captions_est02({'_type': 'url', 'url': '5MMoxyK1Y9o', 'ie_key': 'Youtube', 'id': '5MMoxyK1Y9o', 'title': 'Cybersecurity Fundamentals | Understanding Cybersecurity Basics | Cybersecurity Course | Edureka'})


# In[8]:


transcript_list= []


# In[79]:


tempo=[]
for a in transcript_list:
    try:
        tempo.append(a['id'])
    except:
        pass


# In[77]:


len(full_list)


# In[80]:


len(set(tempo))


# In[3]:


import time
from joblib import Parallel, delayed

'''
transcript_list = Parallel(n_jobs=50, prefer="threads")(
    delayed(captions_test02)(item) for item in full_list)


'''

import concurrent.futures
start = time.time()
# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    # Start the load operations and mark each future with its URL
    
    future_to_url = {executor.submit(captions_test02, item): item for item in full_list[13200:13251]}
    for future in concurrent.futures.as_completed(future_to_url):
        start = time.time()
        url = future_to_url[future]
        try:
            end = time.time()
            if(end-start<=20):
                data = future.result()
                transcript_list.append(data)
            else:
                pass
        except Exception as exc:
            pass


# In[93]:


len(transcript_list)


# In[ ]:


transcript_li


# In[11]:


len(transcript_list)


# In[102]:



ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s'})

def metaData(element):
    if(str(element['url'])=='AaImCn4a-bI'):
        result ={}
    else:
        with ydl:
            result = ydl.extract_info(
                'https://www.youtube.com/watch?v='+str(element['url']),
                download=False # We just want to extract the info
            )
    return result


# In[4]:


import time
from joblib import Parallel, delayed
metadata_list= []
'''
transcript_list = Parallel(n_jobs=50, prefer="threads")(
    delayed(captions_test02)(item) for item in full_list)

'''

import concurrent.futures
start = time.time()
# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    # Start the load operations and mark each future with its URL
    
    future_to_url = {executor.submit(metaData, item): item for item in full_list}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
            metadata_list.append(data)
            print(end = time.time())
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))


# In[ ]:


metadata_l


# In[95]:


import json
with open('/Users/shray/Desktop/recommendations.json', 'w') as fp:
    json.dump(recommendations, fp)


# In[95]:


import pandas as pd

keywords = pd.read_csv('/Users/shray/Desktop/CurrentProjects/dataxProj/KeywordsCybersecurity.csv')


# In[96]:


keyword_list= []

for a in keywords['Just in time manufacturing']:
    temp = a.lower()
    keyword_list.append(temp)


# In[51]:


metadata_list[1]['title']


# In[97]:


final_keyword_list = list(set(keyword_list))


# In[62]:





# In[106]:


meta_title_df = pd.DataFrame(columns=final_keyword_list)


# In[107]:


for a in range(len(metadata_list)):
    try:
        metadata_list[a]['updated_title']= metadata_list[a]['title'].lower()
    except:
        metadata_list[a]['updated_title']= ''


# In[108]:


for b in range(len(metadata_list)):
    rower =[]
    for a in range(len(final_keyword_list)):
        counter = metadata_list[b]['updated_title'].count(final_keyword_list[a])
        rower.append(counter)
    meta_title_df.loc[len(meta_title_df)] = rower


# In[109]:


meta_title_df.to_csv('/Users/shray/Desktop/final_title.csv')


# In[68]:


meta_desc_df = pd.DataFrame(columns=final_keyword_list)


# In[ ]:


for a in range(len(metadata_list)):
    try:
        metadata_list[a]['updated_desc']= metadata_list[a]['description'].lower()
    except:
        metadata_list[a]['updated_desc']= ''


# In[ ]:


for b in range(len(metadata_list)):
    rower =[]
    for a in range(len(final_keyword_list)):
        counter = metadata_list[b]['updated_desc'].count(final_keyword_list[a])
        rower.append(counter)
    meta_desc_df.loc[len(meta_desc_df)] = rower


# In[ ]:


meta_desc_df.to_csv('/Users/shray/Desktop/final_desc.csv')


# In[69]:


for a in range(len(metadata_list)):
    try:
        metadata_list[a]['updated_desc']= metadata_list[a]['description'].lower()
    except:
        metadata_list[a]['updated_desc']= ''


# In[70]:


meta_desc_df = pd.DataFrame(columns=final_keyword_list)


# In[71]:


for b in range(len(metadata_list)):
    rower =[]
    for a in range(len(final_keyword_list)):
        counter = metadata_list[b]['updated_desc'].count(final_keyword_list[a])
        rower.append(counter)
    meta_desc_df.loc[len(meta_desc_df)] = rower


# In[ ]:


meta_desc_df.to_csv('/Users/shray/Desktop/description.csv')


# In[ ]:


dfa = pd.read_csv('/Users/shray/desktop/keywordsmkc.csv')


# In[34]:


transcript_list = list(dfa['Key_Words1'])


# In[39]:


import json

with open("/Users/shray/desktop/CurrentProjects/dataxProj/final_transcript.json") as json_file:
    json_data = json.load(json_file)


# In[43]:


transcript_final_list[0]


# In[40]:


# initializing bad_chars_list 
bad_chars = ['WEBVTT','Kind: captions', 'Language: en',
            'align:start position:0%','<c>','</c>','<','>',':',
            '0','1','2','3','4','5','6','7','8','9','.','\n'] 
  
# using replace() to  
# remove bad_chars 
for a in json_data:
    try:
        for i in bad_chars : 
            a['transcript'] = a['transcript'].replace(i,'') 
    except:
        pass


# In[41]:


meta_trans_df


# In[42]:


meta_trans_df = pd.DataFrame(columns=transcript_final_list)


# In[44]:


for b in range(len(json_data)):
    rower =[]
    for a in range(len(transcript_final_list)):
        try:
            counter = json_data[b]['transcript'].count(transcript_final_list[a])
        except:
            counter = 0
        rower.append(counter)
    meta_trans_df.loc[len(meta_trans_df)] = rower


# In[101]:


meta_trans_df.to_csv('/Users/shray/desktop/Final_transcript_graphdb.csv')


# In[45]:


meta_trans_df.head()


# In[29]:


ide_list = []
for a in json_data:
    try:
        ide_list.append(a['id'])
    except:
        ide_list.append('')


# In[30]:


len(ide_list)


# In[61]:


meta_trans_df['Video Id'] = ide_list


# In[16]:


meta_trans_df.to_csv('612transcriptCSV.csv')


# In[37]:


transcript_df.head()


# In[32]:


final_json_data = filter(None, json_data)


# In[36]:


final_json_data


# In[31]:


while("" in json_data) : 
    json_data.remove("") 


# In[34]:


transcript_df= pd.DataFrame.from_dict(final_json_data,index=final_json_data.keys())


# In[58]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
transcript_df['transcript'] = transcript_df['transcript'].fillna('')

# dropping duplicate values 
transcript_df.drop_duplicates(keep='first') 

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(transcript_df['transcript'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[59]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[60]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(transcript_df.index, index=transcript_df['id']).drop_duplicates()


# In[55]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return transcript_df['id'].iloc[movie_indices]


# In[63]:


get_recommendations('I3IXZhz0SZA')


# In[73]:


unqiue_videos = list(indices.keys())


# In[91]:


recommendations = []

for element in unqiue_videos:
    try:
        temp =[]
        temp2=[]
        temporary ={}
        temp= list(get_recommendations(element))
        for element in temp:
            temp2.append('https://www.youtube.com/watch?v='+str(element))
        temporary['video'] = 'https://www.youtube.com/watch?v='+str(element)
        temporary['recommendations']= temp2
        recommendations.append(temporary)
    except:
        pass


# In[93]:


count = 0
for element in recommendations:
    count+=1
    element['id']=count
    


# In[94]:


recommendations[1]


# In[3]:


import pandas as pd
df1 = pd.read_csv('/Users/shray/Desktop/CurrentProjects/dataxProj/CodeFiles/612transcriptCSV.csv')


# In[18]:


series = df1.duplicated()


# In[20]:


count=0
for a in series:
    if(a==True):
        count+=1
count


# In[ ]:


for a in df1:
    for b in df1:
        


# In[26]:


df1.iloc[1][2]*df1.iloc[1][3]


# In[44]:


import json

with open("/Users/shray/desktop/CurrentProjects/dataxProj/final_transcript.json") as json_file:
    json_data = json.load(json_file)


# In[ ]:


import json 

with open("/Users/shray/desktop/CurrentProjects/dataxProj/final_metadata.json") as json_file:
    meta_data = json.load(json_file)

