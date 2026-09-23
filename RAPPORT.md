# Kundsegmentering med RFM och K-Means

Projektuppgift del 2  
Amir Hamza Jafari, Andreas Malmgren och Robin Hellgren  
GitHub: https://github.com/amirhamza247/ai-customer-segmentation

## 1. Bakgrund

Vi ville göra något som ett riktigt företag skulle kunna ha nytta av, och valde därför kundsegmentering. Tanken är att ett företag ska kunna dela in sina kunder i grupper, till exempel kunder som handlar ofta och mycket, kunder som bara handlat en gång och kunder som inte kommit tillbaka på länge. Då kan man rikta kampanjer till rätt grupp.

Vi använder datasetet Online Retail II från Kaggle. Det innehåller transaktioner från en brittisk webbutik mellan december 2009 och december 2011, drygt en miljon rader. Varje rad är en produkt på en faktura, med fakturanummer, antal, datum, pris och kund-ID. Vi tyckte att datan räckte eftersom det är just de kolumnerna som behövs för en RFM-analys:

- Recency: hur många dagar sedan kunden handlade senast
- Frequency: hur många olika fakturor kunden har
- Monetary: hur mycket kunden har handlat för totalt (antal gånger pris)

För själva segmenteringen använder vi K-Means. Datan har inga färdiga etiketter som säger vilken typ av kund någon är, så det blir oövervakad inlärning. K-Means är också ganska lätt att förklara, vilket var viktigt för oss.

Vi bestämde tidigt att vi skulle göra en proof of concept och fokusera på att hela flödet fungerar, från att man laddar upp en fil till att man ser segmenten. Därför stödjer appen bara ett CSV-format och vi har inte gjort någon churn- eller LTV-modell.

## 2. Hur appen fungerar

Användaren laddar upp CSV-filen i Streamlit och sedan händer följande:

1. Filen valideras och rensas. Vi tar bort rader som saknar kund-ID, makulerade fakturor (de som börjar på C), rader med ogiltigt datum och rader där antal eller pris saknas eller är noll eller negativt. Appen visar hur många rader som togs bort och varför. Dubbletter behåller vi, eftersom samma rad kan vara ett riktigt köp två gånger.
2. RFM räknas ut per kund. Som referensdatum använder vi dagen efter sista köpet i datan, annars skulle resultatet ändras beroende på vilken dag man kör appen.
3. Frequency och Monetary är väldigt sneda, några få kunder har handlat för enorma summor. Därför gör vi log1p på dem och sedan StandardScaler på alla tre, så att Monetary inte tar över bara för att siffrorna är större.
4. Appen testar K från 2 till 8 och räknar silhouette score för varje. Den föreslår det K som fick högst poäng, men användaren väljer själv 2, 3 eller 4.
5. K-Means tränas med det valda K. För varje kluster visas antal kunder och medelvärden för R, F och M i vanliga enheter. Varje kluster får också en enkel beskrivning som säger om det ligger över eller under snittet, till exempel "More recent / Higher frequency / Higher spending".
6. Kunderna visas i två spridningsdiagram (Recency mot Monetary), ett med vanlig och ett med logaritmisk skala.
7. Om man har en API-nyckel kan man be en AI-modell via Groq förklara klustren i text. Vi skickar bara sammanfattningstabellen, inga kund-ID:n. Procentandelarna räknar vi ut själva i Python innan, eftersom språkmodeller ofta räknar fel.

## 3. Resultat

Datasetet hade 1 067 371 rader och efter rensningen var 805 549 kvar. Det mesta som försvann var rader utan kund-ID (243 007 st). Sedan var det 18 744 makuleringar och 71 rader med pris 0. Kvar blev 5 878 kunder.

Datan är väldigt sned. Mediankunden har handlat 3 gånger för ungefär 900 pund, men den största kunden har handlat för över 600 000 pund. Det var därför vi behövde log-transformeringen.

Silhouette score för olika K:

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Silhouette | 0,419 | 0,401 | 0,361 | 0,367 | 0,349 | 0,336 | 0,317 |

K = 2 fick högst poäng men K = 3 var nästan lika bra, och tre grupper säger mer än två. Så här blev det med K = 3:

| Kluster | Beskrivning | Antal kunder | Recency (dagar) | Frequency | Monetary (£) | Andel av intäkterna |
|---|---|---|---|---|---|---|
| 1 | Nyligen, ofta, mycket | 1 689 (29 %) | 58 | 15,9 | 8 691 | 82,7 % |
| 2 | Nyligen, sällan, lite | 2 352 (40 %) | 93 | 2,9 | 861 | 11,4 % |
| 0 | Länge sedan, sällan, lite | 1 837 (31 %) | 474 | 1,8 | 566 | 5,9 % |

Det vi tycker är mest intressant är att ungefär 29 % av kunderna står för 83 % av intäkterna. Kluster 0 har inte handlat på över ett år och har antagligen slutat handla. Kluster 2 är fortfarande aktiva men köper sällan, så det är kanske där företaget har mest att vinna på att försöka få dem att handla mer. Väljer man K = 4 delas de bästa kunderna upp mer, och då står 952 kunder (16 %) för 71 % av intäkterna.

En silhouette score på runt 0,4 betyder att klustren går att skilja åt men att de överlappar en del. Det är ganska normalt för kunddata. Beskrivningarna säger bara om ett kluster ligger över eller under snittet och inte hur mycket, så vi har valt att inte ge klustren namn som "VIP" eller "lojala kunder" eftersom det kan låta mer säkert än det är.

## 4. Teknik

- Python 3.14 med uv för att hantera paket (alla i gruppen får samma versioner via uv.lock)
- pandas och NumPy för att läsa in och rensa datan och räkna ut RFM
- scikit-learn för StandardScaler, KMeans och silhouette_score
- Streamlit för gränssnittet
- Groq för AI-förklaringen (API-nyckeln ligger i en fil som inte laddas upp till GitHub)
- Ruff och Black för att hålla koden snygg
- Git och GitHub för versionshantering

Koden ligger i tre filer. segmentation.py har alla funktioner för rensning, RFM, skalning och klustring. main.py är Streamlit-appen. ai_analysis.py sköter anropet till AI-modellen. Vi ville hålla isär logiken och gränssnittet så att det är lättare att förstå vad som händer var.

### Databas

Uppgiften säger att datan ska lagras i en databas, och det gjorde vi i vår första version. Där hade vi ett skript som läste in CSV-filen till SQLite, och sedan räknades RFM ut med SQL-frågor. Den tränade modellen och segmenten sparades också i en databas som Streamlit-appen läste från.

När vi sedan ändrade appen så att man laddar upp sin egen fil kom vi fram till att databasen mest var i vägen för oss:

- Varje uppladdning analyseras för sig, så det fanns inget som behövde sparas mellan körningarna. Databasen blev bara en kopia av CSV-filen.
- Man var tvungen att köra fyra skript i rätt ordning innan appen fungerade, och det var lätt att datan i databasen inte stämde med koden längre.
- Databasfilerna låg i Git och de går inte att läsa i en pull request.

Därför görs allt i minnet med pandas nu. Om vi senare vill spara resultat, till exempel för att jämföra segment över tid, är det då vi skulle lägga tillbaka SQLite.

### AI-verktyg

Vi har använt AI-kodagenter (Claude Code och Codex) för att skriva en stor del av koden. För att den inte skulle bli för stor och krånglig skrev vi en fil med regler för agenten (först CLAUDE.md, sedan AGENTS.md). Där står bland annat att koden ska vara så enkel som möjligt, att inga nya paket får läggas till utan anledning och att agenten ska förklara det den gör. Vi har försökt se till att vi förstår all kod vi lämnar in.

## 5. Utvärdering av vårt arbete

Det som gick bra var att vi ganska snabbt fick ett flöde som fungerade hela vägen, vilket var målet med PoC:n. Vi delade upp arbetet i små steg (rensa, RFM, skala, välja K, klustra, beskriva, visualisera) och fick varje del att fungera innan vi gick vidare. Vi hade också standups som vi skrev ner i MEETINGS.md.

Vi har lärt oss mycket om hur man förbereder data för klustring, till exempel varför man behöver log-transformera och skala, och att silhouette score bara är en fingervisning. Vi har också lärt oss att klustrens nummer inte betyder något, man måste titta på siffrorna för att förstå vad varje kluster är. En annan stor sak var att jobba med AI-agenter. De skriver kod väldigt snabbt men utan regler blir det lätt mer kod än det behövs. Vi lärde oss att granska diffen innan varje commit och fråga agenten när vi inte förstod något. Ibland satt vi tillsammans i Live Share och följde när en av oss jobbade med agenten.

Med Git och GitHub jobbade vi med en branch per person och uppgift och slog ihop dem med pull requests. Första dagen testade vi hela PR-flödet med några testfiler så att alla visste hur det gick till. Det som fungerade sämre var att två olika versioner av appen byggdes på olika brancher samtidigt, en med SQLite-skript och en med uppladdning. Det gjorde att vi fick börja om lite och lägga tid på att slå ihop. Vi skrev också commit-meddelanden både på svenska och engelska.

Om vi gjorde om projektet skulle vi:

- bestämma arkitekturen tillsammans innan vi började koda, så att vi inte bygger två versioner parallellt
- komma överens om hur vi jobbar med brancher, commits och code review från början (code review-delen i MEETINGS.md använde vi nästan inte)
- skriva några enkla tester för rensningen och RFM tidigt
- försöka lägga upp appen på Streamlit Community Cloud så att den har en länk
