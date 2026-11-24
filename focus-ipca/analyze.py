"""Does the market get inflation right? Focus survey forecasts against the IPCA that came true.

Reads data/ (run fetch.py first) and writes results/.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA, RESULTS = HERE / "data", HERE / "results"
FIRST, LAST = 2001, 2025  # years with a forecast made in December of the year before and a full IPCA


def load():
    annual = pd.read_csv(DATA / "focus_ipca_annual.csv", parse_dates=["Data"])
    ahead = pd.read_csv(DATA / "focus_ipca_12m.csv", parse_dates=["Data"])
    ipca = pd.read_csv(DATA / "ipca_12m.csv", parse_dates=["data"]).set_index("data")["valor"]
    target = pd.read_csv(DATA / "cmn_target.csv", parse_dates=["data"])
    target = target.set_index(target["data"].dt.year)["valor"]
    return annual, ahead, ipca, target


def last_in_month(df: pd.DataFrame, year: int, month: int) -> pd.Series:
    """The last survey published in a given month (the Focus is daily; the report is weekly)."""
    rows = df[(df["Data"].dt.year == year) & (df["Data"].dt.month == month)]
    return rows.loc[rows["Data"].idxmax()]


def metrics(err: pd.Series) -> dict:
    return {
        "bias": err.mean(),
        "mae": err.abs().mean(),
        "within_1pp": (err.abs() <= 1).mean(),
        "worst": err.loc[err.abs().idxmax()],
        "worst_year": err.abs().idxmax(),
    }


def annual_table(annual, ipca, target, stat="Mediana", base=0, first=FIRST) -> pd.DataFrame:
    a = annual[annual["baseCalculo"] == base]
    rows = []
    for y in range(first, LAST + 1):
        ref = a[a["DataReferencia"] == y]
        dec_before = last_in_month(ref, y - 1, 12)
        late = ref.loc[ref["Data"][ref["Data"].dt.year == y].idxmax()]  # last survey of the year itself
        rows.append({
            "year": y,
            "realized": ipca[pd.Timestamp(y, 12, 1)],
            "focus": dec_before[stat],
            "focus_date": dec_before["Data"].date(),
            "respondents": dec_before["numeroRespondentes"],
            "last12": ipca[pd.Timestamp(y - 1, 11, 1)],  # November IPCA is out by mid-December
            "target": target[y],
            "focus_late": late[stat],
            "focus_late_date": late["Data"].date(),
        })
    return pd.DataFrame(rows).set_index("year")


def by_horizon(annual, ipca, base=0, stat="Mediana") -> pd.DataFrame:
    """MAE of the Focus annual forecast by how many months before the end of the year it was made."""
    a = annual[annual["baseCalculo"] == base]
    out = []
    for h in range(0, 24):
        errs = []
        for y in range(FIRST + 1, LAST + 1):
            when = pd.Timestamp(y, 12, 1) - pd.DateOffset(months=h)
            ref = a[a["DataReferencia"] == y]
            try:
                f = last_in_month(ref, when.year, when.month)[stat]
            except ValueError:
                continue
            errs.append(ipca[pd.Timestamp(y, 12, 1)] - f)
        e = pd.Series(errs)
        out.append({"months_before_end": h, "n": len(e), "mae": e.abs().mean(), "bias": e.mean()})
    return pd.DataFrame(out)


def ahead_12m(ahead, ipca, shift_months: int, suav="S", base=0, stat="Mediana") -> pd.Series:
    """Focus 12 months ahead, at the end of each month, against the 12-month IPCA `shift_months` later."""
    m = ahead[(ahead["Suavizada"] == suav) & (ahead["baseCalculo"] == base)].copy()
    m["month"] = m["Data"].dt.to_period("M").dt.to_timestamp()
    m = m.sort_values("Data").groupby("month").last()[stat]
    realized = ipca.reindex(m.index + pd.DateOffset(months=shift_months))
    err = pd.Series(realized.values - m.values, index=m.index).dropna()
    return err


def main():
    RESULTS.mkdir(exist_ok=True)
    annual, ahead, ipca, target = load()

    t = annual_table(annual, ipca, target)
    t.to_csv(RESULTS / "annual_forecasts.csv")
    print(t.round(2).to_string(), "\n")

    summary = {}
    for name in ["focus", "last12", "target", "focus_late"]:
        summary[name] = metrics(t["realized"] - t[name])
    summary = pd.DataFrame(summary).T
    summary.to_csv(RESULTS / "annual_summary.csv")
    print(f"Annual IPCA {FIRST}-{LAST} (error = realized - forecast, p.p.)")
    print(summary.to_string(), "\n")

    # Since 2004 only, leaving out the 2002-2003 crisis
    s04 = {n: metrics(t.loc[2004:, "realized"] - t.loc[2004:, n]) for n in ["focus", "last12", "target"]}
    print("2004-2025\n", pd.DataFrame(s04).T.to_string(), "\n")

    # Variants: mean vs median, 30-day vs 4-day respondents. Base 1 exists only since 2014,
    # so every variant is compared on the same years (2015-2025).
    variants = {}
    for stat in ["Mediana", "Media"]:
        for base in [0, 1]:
            tv = annual_table(annual, ipca, target, stat=stat, base=base, first=2015)
            variants[f"{stat}/base{base}"] = metrics(tv["realized"] - tv["focus"])
    variants = pd.DataFrame(variants).T
    variants.to_csv(RESULTS / "variants_2015_2025.csv")
    print("Variants 2015-2025\n", variants.to_string(), "\n")

    ef, en = (t["realized"] - t["focus"]).abs(), (t["realized"] - t["last12"]).abs()
    et = (t["realized"] - t["target"]).abs()
    print("Focus closer than last 12 months:", (ef < en).sum(), "of", len(t))
    print("Focus closer than target:", (ef < et).sum(), "of", len(t))
    print("Realized above Focus:", (t["realized"] > t["focus"]).sum(), "of", len(t), "\n")

    h = by_horizon(annual, ipca)
    h.to_csv(RESULTS / "by_horizon.csv", index=False)
    print(h.round(2).to_string(index=False), "\n")

    rows = []
    for shift in [11, 12, 13]:
        for suav in ["S", "N"]:
            e = ahead_12m(ahead, ipca, shift, suav=suav)
            rows.append({"shift": shift, "suavizada": suav, "n": len(e), "mae": e.abs().mean(), "bias": e.mean()})
    print("12 months ahead\n", pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
