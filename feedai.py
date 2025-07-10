#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd

import sys
import os
import argparse
import yaml
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

import feedparser
import openai

with open("config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)


# In[2]:


def is_jupyter():
    """
    True: if run in jupyter; 
    False: if run in console
    """
    return hasattr(sys, 'ps1') or 'ipykernel' in sys.modules


# In[3]:


def get_arguments(rss_url, portal, model):
    """
    Argumentumok betöltése CLI-ből vagy Jupyter környezetből.
    """
    default_args = {
        "rss_url": rss_url,
        "portal": portal,
        "model": model
    }

    if is_jupyter():
        class Args:
            def __init__(self, args_dict):
                self.__dict__.update(args_dict)
        return Args(default_args)
    else:
        parser = argparse.ArgumentParser(description = "RSS feldolgozó OpenAI modellel")
        parser.add_argument('--rss_url', type = str, required=True, help = 'Az RSS feed URL-je')
        parser.add_argument('--portal', type = str, required=True, help = 'A portál neve')
        parser.add_argument('--model', type = str, default="o4-mini", help = 'A használt OpenAI modell neve')
        return parser.parse_args()


# In[4]:


def load_feed(rss_url):
    feed = feedparser.parse(rss_url)
     
    link = [
    {
        'title': entry.title,
        'published': entry.published,
        'link': entry.link
    }
    for entry in feed.entries[0:200]
    ]
            
    return pd.DataFrame(link)


# In[24]:


args = get_arguments(rss_url = "https://www.telex.hu/rss/", model = "o4-mini", portal = "telex")

rss_url = args.rss_url
model_name = args.model
portal = args.portal


# In[26]:


print("Get feeds!")
df = load_feed(rss_url)

df['date'] = datetime.now().strftime('%y%m%d')
df['portal'] = portal
df['topic'] = df['link'].str.split('/', expand = True)[3] # Ususaly 4. index is the topic
df['published'] = pd.to_datetime(df['published'], format = '%a, %d %b %Y %H:%M:%S %z')
df['published'] = df['published'].dt.strftime('%Y-%m-%d %H:%M')
print(f"Feed list shape: {df.shape[0]} row!")


# In[27]:


print("Init..")
args = get_arguments(rss_url = "https://www.telex.hu.hu/rss/", model = "o4-mini", portal = "telex")

rss_url = args.rss_url
model_name = args.model
portal = args.portal


# In[28]:


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key = api_key)


# In[30]:


# Már feldolgozott cikkek gyűjője. - Already processed articles
df_links = pd.read_csv('links.csv')
# Már feldolgozott cikkek szűrése - Filter alredy processed article
df_news = df[~df['link'].isin(df_links['link'])]
# Feldolgozásra szánt cikkek kiválasztása - Select article count per run
df_news = df_news.iloc[0:config["max_article"]]

print(f"Process {len(df_news)} article from {portal}! ")


# In[31]:


with open('system_prompt.txt', 'r', encoding = 'utf-8') as file:
    system_prompt = file.read()


# In[ ]:


response_text = ''
new_token_count = 0
completion_token_count = 0

print(f"Process {len(df_news)} articles from {portal}!")
for idx, article in tqdm(df_news.iterrows(), total = len(df_news)-20, desc = "Processing articles"):
    
    
    try:
        # OpenAI válasz lekérése az adott link alapján - OpenAI response 
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": article['link']}
        ]
        
        response = client.chat.completions.create(model = model_name, messages = messages)

        # Tokenhasználat nyomon követése - Track token usage
        new_token_count += response.usage.prompt_tokens
        completion_token_count += response.usage.completion_tokens

        # Markdown szöveg összeállítása - Build markdown-formatted summary block
        response_text += (
            f"Dátum: {article['published']} <br>\n"
            f"Topic: {article['topic']} <br><br>\n\n"
            f"{response.choices[0].message.content}<br>\n\n"
            f"[{article['title']}]({article['link']}) <br>\n\n"
            "<br><hr><hr><br>\n"
        )

    except Exception as e:
        print(f"Hiba: {e}")
        
        continue
print("Done!")


# In[34]:


# Tokenköltség kiszámítása (OpenAI pricing alapján) - Calculate token cost (based on OpenAi)
new_token_cost = new_token_count / 1_000_000 * 1.1
completion_token_cost = completion_token_count / 1_000_000 * 4.4
total_cost = new_token_cost + completion_token_cost

response_text += (
    f"New token cost(o4-mini): [{new_token_count}] – {new_token_cost:.5f}$\n"
    f"Completion token cost(o4-mini): [{completion_token_count}] – {completion_token_cost:.5f}$\n"
    f"**Total(o4-mini): {total_cost:.5f}$**\n"
)

# In[22]:


# Mentés google drive-ra - Save to google drive
if config['google_drive_path']:
    with open(f"{config['google_drive_path']}{portal}{datetime.now().strftime('%y%m%d')}.md", "w", encoding = "utf-8") as file:
        file.write(response_text)


# In[23]:


# output local
with open(f"output/{portal}{datetime.now().strftime('%y%m%d')}.md", "w", encoding = "utf-8") as file:
    file.write(response_text)

