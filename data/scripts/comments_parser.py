from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import time
import json

def parse_comments(html_content, username_dict):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    comments = soup.find_all('div', class_='bp_post')
    
    results = []
    
    for comment in comments:
        author_tag = comment.find('a', class_='bp_author')
        author_username = None
        if author_tag and author_tag.get('href'):
            author_username = author_tag['href'].lstrip('/')
        
        text_tag = comment.find('div', class_='bp_text')
        comment_text = text_tag.get_text(strip=True) if text_tag else None
        
        reply_to_username = None
        if text_tag:
            mention = text_tag.find('a', class_='mem_link')
            if mention:
                mention_id = mention.get('mention_id')
                if mention_id:
                    reply_to_username = mention_id
                else:
                    href = mention.get('href')
                    if href:
                        reply_to_username = href.lstrip('/')
        if reply_to_username is not None:
            comma_index = comment_text.find(",")
            comment_text = comment_text[comma_index + 1 + (1 if comma_index != -1 and comma_index + 1 < len(comment_text) and comment_text[comma_index + 1] == " " else 0):]
        
        if author_username not in username_dict.keys():
            username_dict[author_username] = len(username_dict)
        
        if reply_to_username not in username_dict.keys():
            username_dict[reply_to_username] = len(username_dict)

        results.append({
            'author': username_dict[author_username],
            'reply_to': username_dict[reply_to_username],
            'text': comment_text
        })
    
    return results

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

all_comments = []
username_dict = dict()

for offset in tqdm(range(0, 3501, 20)):
    response = requests.get(
        'https://vk.com/topic-36567706_26132044?offset=' + str(offset),
        headers=headers,
        cookies={'remixlang': 'ru'}
    )

    all_comments += parse_comments(response.text, username_dict)
    time.sleep(0.05)

with open('research/comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_comments, f, ensure_ascii=False, indent=2)
