#/Users/user/opt/miniconda3/envs/langchain2024/bin/python nyt_news_api.py
import requests
import json
import pandas as pd
import datetime as dt
import glob

def getnews(key,startdate,enddate,country):
    apikey = key
    country=country
    startdate=startdate
    enddate=enddate
    url_date_us=f'https://api.nytimes.com/svc/archive/v1/2024/11.json?api-key={apikey}&fq=section_name:({country})&fl=pub_date,section_name&begin_date={startdate}&end_date={enddate}&sort=oldest&facet=true&facet_mincount=500'
    
    response = requests.get(url_date_us)
    print(response)
    
    #get the data in json
    data = response.json()
    # Extract the articles
    articles = data['response']['docs']
    
    response_data=[]
    for news in articles:
        news_dict = {
            "pub_date" : news.get("pub_date", ''),
            'headline': news.get('headline', {}).get('main', ''),
            "abstract" : news.get("abstract", ''),
            "lead_paragraph" : news.get("lead_paragraph", ''),
            "snippet" : news.get("snippet", ''), 
            "news_desk" : news.get("news_desk", ''),                          
        }
        
        # Append the article data to the list
        response_data.append(news_dict)
        
    df= pd.DataFrame(response_data)
    print(df.shape)
    return df
        
    

if __name__ == "__main__":
    key ='VjBfKZ9yBpLjrNESMlk9GHOoRm8EspOT'
    startdate = "20241101"
    enddate = "20241130"
    country="U.S."
    newsdf= getnews(key,startdate,enddate,country)
    
    # Convert the 'pub_date' column to datetime format
    newsdf['pub_date'] = pd.to_datetime( newsdf['pub_date'])
    
    # Create a new column with the date in the format "2024-11-13"
    newsdf['date'] = newsdf['pub_date'].dt.strftime('%Y-%m-%d')
    print(newsdf.shape)
    newsdf.to_csv("november_news.csv",index=False)
    
    dates= newsdf['date'].unique().tolist()
    #Save df for each day
    for day in dates:
        temp_df =newsdf[newsdf['date'] == day]
        temp_df.to_csv("news_by_date/"+str(day)+"_news.csv", index=False)
        
    print(glob.glob('/news_by_date/*'))
    print("")
    
    
    
    