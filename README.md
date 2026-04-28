# FeedAI  
## RSS alapú AI hírösszefoglaló pipeline

A FeedAI egy automatizált adatfeldolgozó pipeline, amely RSS feed-ekből gyűjt híreket, majd az OpenAI API segítségével semleges, tömör és zajmentesített összefoglalót készít.

A rendszer célja egy objektív, jól olvasható hírfolyam előállítása, amely kiszűri a clickbait és érzelmi torzításokat.

---

## Architektúra áttekintés

RSS Feed → DataFrame → Szűrés (links.csv) → OpenAI feldolgozás → Markdown output

---

## Technológia

- Python  
- OpenAI API  
- pandas  
- feedparser  
- tqdm  
- python-dotenv  
- YAML konfiguráció  

---

## Fő működés

1. RSS feed betöltése  
2. Cikkek kinyerése (cím, dátum, link)  
3. Duplikáció szűrés (links.csv)  
4. OpenAI modell meghívása  
5. Semleges összefoglaló generálása  
6. Markdown mentés  

---

## Feature-ök

- RSS aggregáció több forrásból  
- Automatikus cikkfeldolgozás  
- Duplikáció kezelés  
- AI alapú újrafogalmazás  
- Semleges, tárgyilagos stílus  
- Markdown export  
- Token használat és költség követés  
- Opcionális Google Drive mentés  

---

## Input források (példák)

- https://telex.hu/rss  
- https://g7.hu/feed/  
- https://www.portfolio.hu/rss/all.xml  

---

## Output formátum

Dátum: 2026-04-28  
Topic: gazdasag  

- Rövid, semleges összefoglaló  
- Kulcspontok bullet formában  

[Eredeti cikk](link)

---

## Konfiguráció

A config.yaml fájl szabályozza:
```
max_article: 20  
google_drive_path: ""
```

---

## Projekt státusz

Prototípus.

Fókusz:
- LLM-alapú tartalomfeldolgozás  
- RSS aggregáció  
- információs zaj csökkentés  

---

# Használati útmutató

## 1. Telepítés
```
pip install -r requirements.txt
```

---

## 2. Környezeti változók

Hozz létre egy `.env` fájlt:
```
OPENAI_API_KEY=your_api_key_here
```

Fontos: idézőjelek nélkül.

---

## 3. Konfiguráció

A config.yaml fájlban állítható:

- max_article → feldolgozott cikkek száma  
- google_drive_path → mentési útvonal (opcionális)  

---

## 4. Futtatás
```
python feedai.py --rss_url "https://www.telex.hu/rss" --portal "telex" --model "o4-mini"
```

---

## 5. Működés

A script:

- betölti az RSS feed-et  
- kiszűri a már feldolgozott linkeket  
- minden új cikket feldolgoz  
- AI-val összefoglalót készít  
- .md fájlba menti  

---

## 6. Output

output/{portal}{YYMMDD}.md

---

## 7. Költség követés

A rendszer számolja:

- prompt tokenek  
- completion tokenek  
- becsült költség  

---

## 8. Tipikus use case-ek

- napi hír digest  
- több portál aggregálása  
- zajmentes hírfogyasztás  
- AI alapú monitoring  

---

## 9. Megjegyzés

A rendszer a cikk linkjét adja át az LLM-nek, nem a teljes tartalmat.  
Az összefoglaló minősége modellfüggő.
