# CLAUDE.md

Instruktioner till AI-agenten (och till oss). Läs hela innan du skriver kod.

## Projektet

Kundsegmentering på Online Retail II (Kaggle): RFM + LTV per kund, K-means, en Streamlit-app.
Skolprojekt i en grupp om tre (Amir, Andreas, Robin) som är nya på agentisk kodning.

Kursens krav, det här är det vi bedöms på:

1. Data i en databas (SQLite, `data/kunder.db`)
2. ML-modellering i Python (scikit-learn)
3. Frontend i Streamlit (bonus: publicerad med en länk)
4. Publikt GitHub-repo med tydlig README, ca 3 sidors rapport (pdf)

**Vi måste kunna förklara varje rad som lämnas in.** Det slår allt annat nedan.

## Flödet

```
CSV ──> 01_ladda_data ──> transactions (SQLite)
                              │
                queries.load_customers()   en rad per kund: recency, frequency, monetary, rfm, ltv
                              │
                prepare.prepare()          log1p + fast kolumnordning
                              │
        02_valj_k (välj k) ──> 03_trana ──> models/modell.pkl + data/app.db
                                                      │
                                                   app.py (Streamlit, läser, tränar aldrig)
```

Kör allt från repots rot:

```bash
uv sync
uv run python 01_ladda_data.py                   # bygger data/kunder.db från CSV:n
uv run python 02_valj_k.py                       # skriver rapport/valj_k.png
uv run python 03_trana.py                        # skriver models/modell.pkl + data/app.db
uv run python 04_visualisera.py                  # skriver rapport/kluster.png
uv run streamlit run app.py
uv run python queries.py                         # självtester (samma för prepare.py)
uv run ruff check .
```

## Var saker hör hemma

Allt ligger platt i roten. Inga undermappar för kod.

| Fil                        | Ansvar                                                                  |
| -------------------------- | ----------------------------------------------------------------------- |
| `queries.py`               | Databaserna, `SNAPSHOT` och all SQL. Ingen SQL någon annanstans.         |
| `prepare.py`               | Enda transformeringen före modellen. Delas av träning och app.          |
| `01_ladda_data.py` … `04_*` | Pipeline-steg, körs i nummerordning, ett ansvar per fil.                |
| `app.py`                   | Allt användaren ser och klickar på.                                     |
| `rapport/`                 | PNG-figurer och underlag till rapporten.                                |
| `eda_robin.ipynb`          | Utforskning av rådatan. Ingår inte i flödet.                            |
| `Prompts.md`               | Plan: stegen vi bygger i ordning.                                       |
| `MEETINGS.md`              | Standups och beslut.                                                    |

Hittar du ingen rad som passar: fråga innan du skapar en ny fil.

## Regler för koden

Målet: någon som öppnar repot för första gången förstår det på tio minuter.

- **Minsta möjliga kod.** Skriv inte det som inte behövs just nu. Inga klasser, config-filer,
  wrappers eller hjälpfunktioner "för senare".
- **Färre filer.** Utöka en befintlig fil hellre än att skapa en ny. Inga `utils.py`, `helpers.py`.
- **Inga nya beroenden** utan att fråga. Streamlit har redan diagram (`st.bar_chart`, Altair).
- **Rakt uppifrån och ned.** Pipeline-skript är vanliga skript, inte funktioner som anropar funktioner.
- **Kommentarer förklarar varför, inte vad.** `# log1p: monetary har skevhet 25,3` är bra,
  `# loggar kolumnen` är brus.
- **Svenska** i kommentarer, utskrifter och appens text. Kolumner och delade funktioner på engelska
  (`recency`, `load_customers`), som i befintlig kod.
- **Fel ska synas.** Ingen `try/except` som sväljer fel. Hellre en krasch med tydligt meddelande.
- **Hårdkoda inte samma värde på flera ställen.** `SNAPSHOT` och `FEATURES` har ett hem.
- **En självkontroll räcker.** Logik som kan gå sönder får `assert` under `if __name__ == "__main__":`,
  som i `queries.py` och `prepare.py`. Inget testramverk.
- **Notebooks är för utforskning.** Kod som behövs i flödet ligger i `.py`-filer.

## ML-regler (lätt att göra fel tyst)

- `fit` körs **bara** i `02_valj_k.py` (utforskning) och `03_trana.py`. App och visualisering
  laddar `models/modell.pkl` och kör `transform`/`predict`.
- Samma `prepare()` i träning och app, alltid följt av den **sparade** scalern.
- `random_state=42` överallt så att alla tre får samma kluster.
- Klusternamn sätts utifrån klustrens värden, aldrig utifrån klusternummer.
- Ingen persondata utöver `customer_id` i appen.

## Git

- En branch per sak, PR till `main` och `dev`, någon annan i gruppen läser innan merge.
- Små commits med tydligt meddelande: `Steg 4a: app.py med översiktssida`.
- Checka in koden som skapar filer, inte filerna (`*.db`, `*.csv`, `*.pkl` ligger i `.gitignore`).
  Undantag: `data/app.db` och `models/modell.pkl`, som den publicerade appen behöver. Committa dem bara
  efter medveten omträning, och bara en person åt gången (binärfiler går inte att merga).
- Agenten pushar aldrig och mergar aldrig utan att bli ombedd.

## Så jobbar du med oss (agenten)

- **Ifrågasätt.** Om ett steg i `Prompts.md` eller en idé från oss är onödigt komplicerad eller
  metodologiskt fel, säg det kort med ett bättre förslag innan du bygger.
- **Ett steg i taget.** Gör det som efterfrågas, inte nästa steg också.
- **Förklara kort** efteråt: vilka filer som ändrades och hur vi kör dem. Peka ut rader som är
  svåra att förstå.
- **Städa efter dig.** Döda filer, oanvända importer och gamla experiment tas bort, inte kommenteras ut.
- **Kör det du skrivit** innan du säger att det fungerar.
