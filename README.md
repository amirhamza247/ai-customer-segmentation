# AI Customer Segmentation

Kundsegmentering (RFM + klustring) på datasetet Online Retail II.

## Sätta upp miljön

Kräver [uv](https://docs.astral.sh/uv/). uv installerar själv rätt Python-version
(3.14, se `.python-version`).

```bash
# 1. Installera uv (en gång)
brew install uv            # macOS
# eller: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Klona och installera beroenden
git clone <repo-url>
cd ai-customer-segmentation
uv sync
```

`uv sync` skapar `.venv/` och installerar exakt de versioner som står i `uv.lock`,
så alla tre har samma miljö. Kör skript med `uv run`, t.ex. `uv run streamlit run app.py`.

Lägga till ett nytt beroende: `uv add <paket>` och checka in både
`pyproject.toml` och `uv.lock`.

## Data: varför den inte finns i repot

Följande checkas **inte** in (se `.gitignore`):

| Fil                          | Varför                                                                                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/online_retail_II.csv`  | 91 MB. GitHub varnar vid 50 MB och stoppar filer över 100 MB. Datan ändras inte, så det finns ingen historik att versionshantera.              |
| `data/kunder.db`             | Genereras från CSV-filen. Binärfil som går att återskapa, så varje ändring skulle bara ge en ny stor blob i git-historiken och ge merge-konflikter som inte går att lösa. |
| `*.pkl`, `*.joblib`          | Tränade modeller. Också genererade och binära.                                                                                                |

Grundregeln: koden som skapar filerna checkas in, inte filerna.

## Återskapa datan lokalt

```bash
# 1. Ladda ner CSV-filen från Kaggle (kräver inloggning):
#    https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
#    Originalkälla: https://archive.ics.uci.edu/dataset/502/online+retail+ii

# 2. Lägg den i data/
mkdir -p data
mv ~/Downloads/online_retail_II.csv data/

# 3. Bygg databasen
uv run python databearbetning/01_ladda_data.py
```

Kontrollera att filen är rätt: 1 067 371 rader och headern
`Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country`.
