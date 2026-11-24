"""Download the data used in the article, straight from the Central Bank of Brazil (BCB).

- Focus survey: market expectations for IPCA, per calendar year and 12 months ahead (Olinda API).
- IPCA accumulated over 12 months (SGS series 13522).
- CMN inflation target (SGS series 13521).

Files go to data/. Run again to refresh; nothing is edited by hand.
"""
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
OLINDA = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata"
# Without dataInicial/dataFinal the SGS API sometimes answers 304 with an empty body.
SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json&dataInicial=01/01/1999&dataFinal=31/12/2030"


def olinda(resource: str, filter_: str) -> pd.DataFrame:
    """Fetch every row of an Olinda resource, page by page (the API caps a single response)."""
    rows, skip, page = [], 0, 10000
    while True:
        # Olinda rejects "+" for spaces in $filter, so the query string is built with %20.
        query = f"$filter={quote(filter_)}&$format=json&$top={page}&$skip={skip}&$orderby=Data"
        r = requests.get(f"{OLINDA}/{resource}?{query}", timeout=120)
        r.raise_for_status()
        batch = r.json()["value"]
        rows += batch
        if len(batch) < page:
            return pd.DataFrame(rows)
        skip += page


def sgs(code: int) -> pd.DataFrame:
    # Retry a few times: the API is occasionally slow or returns an empty body.
    for attempt in range(5):
        r = requests.get(SGS.format(code), headers={"Cache-Control": "no-cache"}, timeout=60)
        if r.ok and r.text.strip():
            break
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"SGS series {code}: empty response (status {r.status_code})")
    df = pd.DataFrame(r.json())
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float)
    return df


def main() -> None:
    DATA.mkdir(exist_ok=True)
    annual = olinda("ExpectativasMercadoAnuais", "Indicador eq 'IPCA'")
    annual.to_csv(DATA / "focus_ipca_annual.csv", index=False)
    ahead = olinda("ExpectativasMercadoInflacao12Meses", "Indicador eq 'IPCA'")
    ahead.to_csv(DATA / "focus_ipca_12m.csv", index=False)
    sgs(13522).to_csv(DATA / "ipca_12m.csv", index=False)
    sgs(13521).to_csv(DATA / "cmn_target.csv", index=False)
    for f in sorted(DATA.glob("*.csv")):
        print(f.name, len(pd.read_csv(f)))


if __name__ == "__main__":
    main()
