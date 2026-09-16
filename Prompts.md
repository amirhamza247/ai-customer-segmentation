# Prompts för kundsegmenteringsprojektet

Kopiera en prompt i taget till din AI-assistent, i ordning.

---

## Så funkar det (enkel förklaring)

1. **Rådata:** varje rad är en vara på ett kvitto.
2. **En rad per kund:** vi räknar ihop kvittona till fyra siffror per kund:
   - **R**ecency: hur många dagar sedan senaste köpet
   - **F**requency: hur många köp
   - **M**onetary: hur mycket pengar totalt
   - **LTV**: vad kunden förväntas vara värd de kommande 3 åren
     (hur mycket kunden handlar per år, gånger 3)
3. **RFM-kod:** varje kund får betyget 1–5 på R, F och M, till exempel
   `545`. Koden är till för människor och är lätt att läsa.
4. **Klustring:** K-means grupperar kunder som liknar varandra på R, F, M
   och LTV.
5. **Namn:** varje grupp får ett namn, till exempel "Champions" eller "At risk".
6. **App:** Streamlit visar grupperna och placerar nya kunder i rätt grupp.

**Varför klustrar vi på R, F och M var för sig och inte på RFM-koden?**
Koden `511` (ny kund) och `155` (förlorad storkund) har samma summa, 11.
Modellen måste se de tre siffrorna var för sig för att hålla isär dem.
RFM-koden blir därför en kolumn som hjälper oss att läsa resultatet.

---

## Fakta om datasetet

Fil: `online_retail_II.csv` (91 MB, 1 067 371 rader, 2009-12-01 till 2011-12-09)

Kolumner: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`

| Egenskap                         | Värde                  |
| -------------------------------- | ---------------------- |
| Saknade Customer ID              | 243 007 rader (22,8 %) |
| Annullerade ordrar (Invoice C…)  | 19 494 rader           |
| Quantity <= 0                    | 22 950 rader           |
| Price <= 0                       | 6 207 rader            |
| Kunder efter rensning            | 5 878                  |
| Monetary: median / max           | 899 / 608 822          |
| Monetary: skevhet                | 25,3                   |
| Andel United Kingdom             | 90 %                   |

Snapshot-datum: `2011-12-10` (dagen efter sista transaktionen).

Skevheten 25,3 betyder att några få grossister handlar enormt mycket.
Utan log-transform drar de till sig ett eget kluster.

---

## Steg 0: projektuppsättning

```
Sätt upp ett Python-projekt med uv. Beroenden: pandas, numpy, scikit-learn,
matplotlib, seaborn, sqlalchemy, streamlit, joblib.

Skapa .gitignore (.venv, __pycache__, *.db, *.pkl, data/*.csv) och en
README som förklarar hur man sätter upp miljön och återskapar databasen.
Ge mig kommandona jag ska köra.
```

---

## Steg 1a: ladda in till SQLite

```
Skriv 01_ladda_data.py som laddar online_retail_II.csv till tabellen
transactions i data/kunder.db (sqlalchemy, to_sql, chunksize=10000).

- Döp om kolumnerna till: invoice_no, stock_code, description, quantity,
  invoice_date, unit_price, customer_id, country
- customer_id som Int64, invoice_date som text "YYYY-MM-DD HH:MM:SS"
  (annars returnerar julianday() tyst NULL)
- Skapa index på customer_id och invoice_date
- Rensa ingenting, rådatan ska spegla filen
- Skriv ut antal rader
```

---

## Steg 1b: EDA

```
Skriv en notebook med EDA på tabellen transactions i data/kunder.db:

1. Antal rader, kunder, ordrar och datumspann
2. Hur många rader varje rensningsfilter tar bort, var för sig och
   kumulativt: saknat customer_id, quantity <= 0, unit_price <= 0,
   invoice_no som börjar med C
3. Fördelning av antal ordrar och totalbelopp per kund (med och utan log)
4. Topp 10 länder

Skriv en mening under varje figur om vad den betyder för modellen.
```

---

## Steg 1c: en rad per kund

```
Skriv queries.py med en SQL-fråga som gör om tabellen transactions i
SQLite till en rad per kund. Använd julianday() för datum och en namngiven
parameter :snapshot.

Rensning: customer_id IS NOT NULL, quantity > 0, unit_price > 0,
invoice_no NOT LIKE 'C%'.

Kolumner:
- customer_id
- recency: dagar från senaste köp till :snapshot (heltal)
- frequency: antal unika invoice_no
- monetary: summan av quantity * unit_price
- rfm: text som "545". Betyg 1–5 med NTILE(5) för R, F och M.
  Senaste köp ger R=5, flest köp ger F=5, mest pengar ger M=5.
- ltv: monetary / år som kund * 3.
  År som kund = dagar från första köp till :snapshot, minst 90, delat med 365.
  Golvet på 90 dagar hindrar att en kund som köpte igår får 365 gånger
  sitt köp som årsvärde.

Lägg aggregeringen i en CTE och NTILE i den yttre SELECT-satsen.

Skriv också load_customers(snapshot) som kör frågan med pandas.read_sql.
Den ska ge ungefär 5 878 rader.
```

---

## Steg 2: prepare

```
Skriv prepare.py:

FEATURES = ["recency", "frequency", "monetary", "ltv"]
LOG_FEATURES = ["frequency", "monetary", "ltv"]

prepare(df):
1. Kasta ValueError om någon kolumn i FEATURES saknas
2. np.log1p på LOG_FEATURES (monetary har skevhet 25,3)
3. Returnera df[FEATURES] (låser ordningen, scikit-learn går på position)

rfm är inte med i FEATURES. Den är till för att läsa resultatet.
StandardScaler ska ligga utanför prepare, eftersom den lär sig från datan
och måste sparas.
```

---

## Steg 3a: välj antal kluster

```
Skriv 02_valj_k.py. Data: load_customers -> prepare -> StandardScaler.

För k = 2..10: räkna inertia (elbow), silhouette och Davies-Bouldin.
KMeans med random_state=42, n_init=10.

Rita elbow och silhouette, och skriv ut en tabell med alla tre måtten.
Föreslå ett k med motivering, men låt oss välja.
```

---

## Steg 3b: träna och namnge

```
Skriv 03_trana.py:

1. load_customers -> prepare -> StandardScaler.fit_transform
2. KMeans med vårt k, random_state=42, n_init=10
3. Relative importance på OTRANSFORMERADE värden:
   df.groupby("cluster")[FEATURES].mean() / df[FEATURES].mean() - 1
   Rita som heatmap och spara som png.
4. Ge varje kluster ett namn utifrån värdena, till exempel Champions, Loyal,
   New customers, At risk eller Lost. Visa också den vanligaste rfm-koden
   per kluster som stöd.
5. Spara scaler, kmeans, FEATURES, namnen och snapshot i EN dict i
   models/modell.pkl med joblib.
6. Skriv tabellen segments till SQLite: customer_id, recency, frequency,
   monetary, ltv, rfm, cluster, segment_name.

Skriv ut antal kunder per kluster. Varna om något kluster har färre än 50.
```

---

## Steg 4a: Streamlit

```
Bygg app.py som läser tabellen segments från data/kunder.db med
@st.cache_data. Appen får aldrig anropa fit.

Sidor via st.sidebar.radio:

1. Översikt: antal kunder, total omsättning, stapeldiagram med kunder
   och omsättning per segment
2. Segmentprofil: välj segment och visa relative importance som liggande
   staplar, medelvärden mot populationen och en mening om de två största
   avvikelserna
3. Kundlista: filtrera på segment, visa tabellen, ladda ner som CSV
4. Kundsökning: skriv in customer_id och visa segment, rfm och ltv mot
   snittet

Visa ingen annan persondata än customer_id.
```

---

## Steg 4b: ny kund

```
Lägg till sidan "Ny kund" i app.py.

Fyra number_input: recency, frequency, monetary, ltv (default = medianen).
Ladda models/modell.pkl, kör prepare -> scaler.transform -> kmeans.predict.
Använd aldrig fit här, det ger tyst fel segment.
Visa segmentnamnet och kundens värden bredvid segmentets snitt.
```

---

## Steg 5: rapportunderlag

```
Skriv ett utkast (stödord, inte färdig text) till metodavsnittet utifrån
koden:

1. Dataset och rensning (rader per filter)
2. Variablerna R, F, M och LTV, och varför rfm-koden inte är med i modellen
3. Log-transform (skevhet 25,3) och StandardScaler
4. Val av k (elbow, silhouette, Davies-Bouldin)
5. Tolkning med relative importance
6. Begränsningar: LTV antar att kunden stannar 3 år och tar inte hänsyn
   till churn, ingen marginaldata eller demografi, 90 % Storbritannien
```

---

## Kom ihåg

- `prepare()` används både i träning och i appen. Det är samma funktion
  från samma fil.
- `fit` körs bara i 03_trana.py. Allt annat laddar modell.pkl.
- Kan ni inte förklara en rad kod, be om en enklare version.
