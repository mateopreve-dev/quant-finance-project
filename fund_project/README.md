# Fund Project: Momentum in the Electrification of the Economy

Mateo Preve, Geo Cauzza, Alvaro Grohmann Borsio

- `FundProject_Report.docx`: report
- `FundProject_Slides.pptx`: presentation (speaker notes included)
- `code/`: strategy and evaluation code
- `data/raw/`: all raw inputs (S&P 500 membership and sectors, Yahoo prices and shares, Ken French factors, AQR BAB and QMJ factors)
- `output/`: every table and figure used in the report and slides

## Reproduce

```
pip install -r requirements.txt
cd code
python run.py
```

`python run.py` reads `data/raw/` and rewrites `output/`. `python run.py design` runs the 2016-2022 design period only, without opening the holdout. To download the raw data again: `python -c "import data; data.download_all()"`.
