# Does the market get inflation right? Focus vs. IPCA, 2001–2025

Code for the Salada de Dados article **"Does the Market Get Inflation Right? 25 Years of Focus Against the IPCA"**
([English](https://saladadedados.com/en/blog/focus-inflation-forecasts) ·
[Português](https://saladadedados.com/pt/blog/focus-inflation-forecasts)).

It compares the Central Bank of Brazil's Focus survey (the market's median forecast for the IPCA, Brazil's
consumer price index) with the inflation that actually happened, and with two simple guesses:
"it will stay the same" (the last 12 months) and "it will hit the target" (the CMN target).
No API key, no sign-up: everything comes from the Central Bank's public APIs.

## What's inside

| File | What it does |
|---|---|
| `fetch.py` | Downloads the Focus annual and 12-months-ahead IPCA expectations (Olinda API), the 12-month IPCA (SGS 13522) and the CMN target (SGS 13521) to `data/` |
| `analyze.py` | Builds the year-by-year table and every number in the article, and writes them to `results/` |
| `charts.py` | Draws the article's two charts as SVG, in Portuguese and English |
| `results/` | The outputs used in the article (data downloaded on 2026-11-24) |

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fetch.py
python analyze.py
python charts.py out/
```

`fetch.py` always downloads the latest data, so the numbers will change once the 2026 IPCA is out
(change `LAST` in `analyze.py` to include it).

## Choices that matter

- **The forecast is the last Focus published in December of the year before** (median, `baseCalculo` 0,
  that is, answers from the last 30 days). Using a forecast from later in the year itself makes the
  market look 16 times better than it is, because by then most of the year's inflation is already known.
- **"It will stay the same" uses the 12-month IPCA up to November**, the latest figure released by the end
  of December. December's IPCA only comes out in January, after the forecast.
- **Two API traps:** the Olinda API answers 400 if spaces in `$filter` are encoded as `+` (the default in
  `requests` with `params=`), so the query string is built with `%20`; and it silently caps each response
  at `$top` rows, so `fetch.py` pages with `$skip` until a page comes back short. The SGS API sometimes
  answers 304 with an empty body unless `dataInicial` and `dataFinal` are given.
