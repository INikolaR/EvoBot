import requests
from bs4 import BeautifulSoup
import json
import re

def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_property_header(tag):
    if tag.name != "p":
        return False

    has_style16 = (
        (tag.get("class") and "style16" in tag.get("class")) or
        tag.find("span", class_="style16")
    )

    if not has_style16:
        return False

    strong = tag.find("strong")
    if not strong or not clean_text(strong.get_text()):
        return False

    em_tags = tag.find_all("em")
    for em in em_tags:
        em_text = clean_text(em.get_text())
        if em_text and len(em_text) > 3 and "?" in em_text:
            return False
    
    return True

def is_question(tag):
    if tag.name != "p":
        return False
    em = tag.find("em")
    if not em:
        return False
    text = clean_text(em.get_text())
    return len(text) > 3 and "?" in text

def parse_evolution_faq_by_properties(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    response = requests.get(url, headers=headers, cookies={"remixlang": "ru"})
    response.encoding = "windows-1251"
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    result = []
    current_property = None
    current_lines = []
    
    def save_property():
        nonlocal current_property, current_lines
        if current_property and current_lines:
            full_block = current_property + "\n" + "\n".join(current_lines)
            result.append(full_block)
        current_property = None
        current_lines = []

    for p in soup.find_all("p"):
        raw_text = p.get_text()
        cleaned = clean_text(raw_text)
        
        if not cleaned or len(cleaned) < 2:
            continue

        if any(skip in cleaned for skip in [
            "Официальный ЧАВО",
            "Вопросы по базовой",
            "Вопросы по дополнению",
            "Здесь собраны часто задаваемые"
        ]):
            continue

        if is_property_header(p):
            save_property()
            current_property = cleaned
            continue

        if not current_property:
            continue

        if is_question(p):
            current_lines.append(cleaned)
        elif p.get("class") and "style2" in p.get("class") and not is_question(p):
            if cleaned and cleaned not in [" ", "&nbsp;", "\xa0"]:
                current_lines.append(cleaned)

    save_property()
    
    return result

faq_data = parse_evolution_faq_by_properties("https://rightgames.ru/evolution_faq.html")

with open("research/faq.json", "w", encoding="utf-8") as f:
    json.dump(faq_data, f, ensure_ascii=False, indent=2)
