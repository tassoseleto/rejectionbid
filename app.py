import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="GAM Bid Analyzer (Daily MVP)", layout="wide")

st.title("GAM Bid Analyzer — MVP diário")
st.write("Faça upload do CSV do GAM (com Date, Bidder, Bids, Average bid CPM, Bid rejection reason, Bid range).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

def to_float_bids(x):
    # "56,296" -> 56296
    return pd.to_numeric(str(x).replace(",", ""), errors="coerce")

def wavg_cpm(g: pd.DataFrame) -> float:
    g2 = g.dropna(subset=["Average bid CPM (R$)", "Bids"]).copy()
    if g2.empty:
        return np.nan
    denom = g2["Bids"].sum()
    if denom == 0:
        return np.nan
    return (g2["Average bid CPM (R$)"] * g2["Bids"]).sum() / denom

def normalize_share(series: pd.Series) -> pd.Series:
    s = series.fillna(0)
    tot = s.sum()
    return s / tot if tot else s

def top_changes(today: pd.Series, prev: pd.Series, topn=10) -> pd.DataFrame:
    idx = today.index.union(prev.index)
    t = today.reindex(idx, fill_value=0)
    p = prev.reindex(idx, fill_value=0)
    delta_pp = (t - p) * 100
    out = pd.DataFrame({
        "item": idx,
        "share_today_%": (t * 100).round(2),
        "share_prev_%": (p * 100).round(2),
        "delta_pp": delta_pp.round(2),
    }).sort_values("delta_pp", ascending=False)
    return out.head(topn), out.tail(topn).sort_values("delta_pp")

def build_daily_tables(df: pd.DataFrame):
    daily = df.groupby("Date").apply(lambda g: pd.Series({
        "Total Bids": g["Bids"].sum(),
        "Avg Bid CPM (R$) (wavg)": wavg_cpm(g),
        "Unique Bidders": g["Bidder"].nunique(),
        "Unique Reasons": g["Bid rejection reason"].nunique(),
        "Unique Ranges": g["Bid range"].nunique(),
    })).reset_index()

    return daily.sort_values("Date")

if uploaded:
    df = pd.read_csv(uploaded)

    # Limpa possíveis linhas "Total"
    if "Date" in df.columns:
        df = df[df["Date"].astype(str).str.lower().ne("total")].copy()

    required = {"Date","Bidder","Bids","Average bid CPM (R$)","Bid rejection reason","Bid range"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Colunas faltando no CSV: {missing}")
        st.stop()

    # Parse
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    df["Bids"] = df["Bids"].apply(to_float_bids)
    df["Average bid CPM (R$)"] = pd.to_numeric(df["Average bid CPM (R$)"], errors="coerce")

    df = df.dropna(subset=["Date","Bids"]).copy()

    st.subheader("Resumo diário")
    daily = build_daily_tables(df)
    st.dataframe(daily, use_container_width=True)

    st.divider()
    st.subheader("Análise por dia (com mudanças vs dia anterior)")

    dates = sorted(df["Date"].unique())
    if not dates:
        st.warning("Nenhuma data válida encontrada.")
        st.stop()

    selected = st.selectbox("Escolha o dia para ver detalhes", dates, format_func=lambda d: d.strftime("%Y-%m-%d"))
    topn = st.slider("Top N itens", 5, 30, 10)

    day_df = df[df["Date"] == selected].copy()
    prev_date = None
    for i, d in enumerate(dates):
        if d == selected and i > 0:
            prev_date = dates[i-1]
            break

    # Cards principais do dia
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bids", f"{int(day_df['Bids'].sum()):,}".replace(",", "."))
    col2.metric("Avg Bid CPM (wavg)", f"R$ {wavg_cpm(day_df):.3f}")
    col3.metric("Bidders únicos", int(day_df["Bidder"].nunique()))
    col4.metric("Reasons únicas", int(day_df["Bid rejection reason"].nunique()))

    # Distribuições do dia
    bidder_share = normalize_share(day_df.groupby("Bidder")["Bids"].sum()).sort_values(ascending=False)
    reason_share = normalize_share(day_df.groupby("Bid rejection reason")["Bids"].sum()).sort_values(ascending=False)
    range_share  = normalize_share(day_df.groupby("Bid range")["Bids"].sum()).sort_values(ascending=False)

    left, right = st.columns(2)

    with left:
        st.markdown("### Share de bidders (Top)")
        st.dataframe((bidder_share.head(topn)*100).round(2).rename("share_%"), use_container_width=True)

        st.markdown("### Share de Bid rejection reason")
        st.dataframe((reason_share*100).round(2).rename("share_%"), use_container_width=True)

    with right:
        st.markdown("### Distribuição de bid range (Top)")
        st.dataframe((range_share.head(topn)*100).round(2).rename("share_%"), use_container_width=True)

        # CPM por reason (ponderado)
        st.markdown("### CPM (wavg) por reason")
        reason_cpm = day_df.groupby("Bid rejection reason").apply(wavg_cpm).sort_values(ascending=False)
        st.dataframe(reason_cpm.rename("cpm_wavg").round(3), use_container_width=True)

    # Mudanças vs dia anterior
    if prev_date is not None:
        st.divider()
        st.markdown(f"## O que mudou vs {prev_date.strftime('%Y-%m-%d')}")

        prev_df = df[df["Date"] == prev_date].copy()
        prev_bidder_share = normalize_share(prev_df.groupby("Bidder")["Bids"].sum())
        prev_reason_share = normalize_share(prev_df.groupby("Bid rejection reason")["Bids"].sum())
        prev_range_share  = normalize_share(prev_df.groupby("Bid range")["Bids"].sum())

        # Top mudanças bidders
        inc_b, dec_b = top_changes(bidder_share, prev_bidder_share, topn=topn)
        st.markdown("### Bidders — maiores altas (p.p.)")
        st.dataframe(inc_b, use_container_width=True)
        st.markdown("### Bidders — maiores quedas (p.p.)")
        st.dataframe(dec_b, use_container_width=True)

        # Top mudanças reasons
        inc_r, dec_r = top_changes(reason_share, prev_reason_share, topn=topn)
        st.markdown("### Reasons — maiores altas (p.p.)")
        st.dataframe(inc_r, use_container_width=True)
        st.markdown("### Reasons — maiores quedas (p.p.)")
        st.dataframe(dec_r, use_container_width=True)

        # Top mudanças ranges
        inc_rg, dec_rg = top_changes(range_share, prev_range_share, topn=topn)
        st.markdown("### Bid range — maiores altas (p.p.)")
        st.dataframe(inc_rg, use_container_width=True)
        st.markdown("### Bid range — maiores quedas (p.p.)")
        st.dataframe(dec_rg, use_container_width=True)

        # Delta CPM e bids (dia)
        prev_total = prev_df["Bids"].sum()
        prev_cpm = wavg_cpm(prev_df)
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Δ Total Bids", f"{int(day_df['Bids'].sum()-prev_total):,}".replace(",", "."), delta=f"{(day_df['Bids'].sum()/prev_total-1)*100:.1f}%")
        c2.metric("Δ Avg Bid CPM (wavg)", f"R$ {wavg_cpm(day_df)-prev_cpm:.3f}", delta=f"{(wavg_cpm(day_df)/prev_cpm-1)*100:.1f}%")

    else:
        st.info("Este é o primeiro dia do dataset — não há dia anterior para comparar.")
