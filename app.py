# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="GAM Bid Analyzer (Daily MVP)", layout="wide")
st.title("GAM Bid Analyzer — MVP diário")
st.write("Faça upload do CSV do GAM (com Date, Bidder, Bids, Average bid CPM, Bid rejection reason, Bid range).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# ----------------------------
# Helpers
# ----------------------------
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

def top_changes(today: pd.Series, prev: pd.Series, topn=10):
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

    inc = out.head(topn)
    dec = out.tail(topn).sort_values("delta_pp")
    return inc, dec

def build_daily_tables(df: pd.DataFrame):
    daily = df.groupby("Date").apply(lambda g: pd.Series({
        "Total Bids": g["Bids"].sum(),
        "Avg Bid CPM (R$) (wavg)": wavg_cpm(g),
        "Unique Bidders": g["Bidder"].nunique(),
        "Unique Reasons": g["Bid rejection reason"].nunique(),
        "Unique Ranges": g["Bid range"].nunique(),
    })).reset_index()
    return daily.sort_values("Date")

def fmt_int(n):
    return f"{int(n):,}".replace(",", ".")

def fmt_pct(x):
    if pd.isna(x):
        return "n/a"
    return f"{x*100:.1f}%"

def daily_insights(df_all: pd.DataFrame, day: pd.Timestamp, topn: int = 5):
    day_df = df_all[df_all["Date"] == day].copy()
    if day_df.empty:
        return None

    dates = sorted(df_all["Date"].unique())
    prev_day = None
    for i, d in enumerate(dates):
        if d == day and i > 0:
            prev_day = dates[i - 1]
            break

    # KPIs do dia
    kpis = {
        "total_bids": float(day_df["Bids"].sum()),
        "avg_cpm": float(wavg_cpm(day_df)),
        "unique_bidders": int(day_df["Bidder"].nunique()),
        "unique_reasons": int(day_df["Bid rejection reason"].nunique()),
        "unique_ranges": int(day_df["Bid range"].nunique()),
    }

    bidder_share = normalize_share(day_df.groupby("Bidder")["Bids"].sum()).sort_values(ascending=False)
    reason_share = normalize_share(day_df.groupby("Bid rejection reason")["Bids"].sum()).sort_values(ascending=False)
    range_share  = normalize_share(day_df.groupby("Bid range")["Bids"].sum()).sort_values(ascending=False)

    result = {
        "day": day,
        "prev_day": prev_day,
        "kpis": kpis,
        "bidder_share": bidder_share,
        "reason_share": reason_share,
        "range_share": range_share
    }

    if prev_day is None:
        result["deltas"] = None
        result["highlights"] = ["Primeiro dia do dataset (sem comparação)."]
        return result

    prev_df = df_all[df_all["Date"] == prev_day].copy()
    prev_kpis = {
        "total_bids": float(prev_df["Bids"].sum()),
        "avg_cpm": float(wavg_cpm(prev_df)),
        "unique_bidders": int(prev_df["Bidder"].nunique()),
        "unique_reasons": int(prev_df["Bid rejection reason"].nunique()),
        "unique_ranges": int(prev_df["Bid range"].nunique()),
    }

    prev_bidder_share = normalize_share(prev_df.groupby("Bidder")["Bids"].sum())
    prev_reason_share = normalize_share(prev_df.groupby("Bid rejection reason")["Bids"].sum())
    prev_range_share  = normalize_share(prev_df.groupby("Bid range")["Bids"].sum())

    deltas = {
        "total_bids_abs": kpis["total_bids"] - prev_kpis["total_bids"],
        "total_bids_pct": (kpis["total_bids"] / prev_kpis["total_bids"] - 1) if prev_kpis["total_bids"] else np.nan,
        "avg_cpm_abs": kpis["avg_cpm"] - prev_kpis["avg_cpm"],
        "avg_cpm_pct": (kpis["avg_cpm"] / prev_kpis["avg_cpm"] - 1) if prev_kpis["avg_cpm"] else np.nan,
        "unique_bidders_abs": kpis["unique_bidders"] - prev_kpis["unique_bidders"],
        "unique_reasons_abs": kpis["unique_reasons"] - prev_kpis["unique_reasons"],
        "unique_ranges_abs": kpis["unique_ranges"] - prev_kpis["unique_ranges"],
    }

    inc_b, dec_b = top_changes(bidder_share, prev_bidder_share, topn=topn)
    inc_r, dec_r = top_changes(reason_share, prev_reason_share, topn=topn)
    inc_rg, dec_rg = top_changes(range_share, prev_range_share, topn=topn)

    highlights = []

    # KPI highlights
    if deltas["avg_cpm_abs"] > 0:
        highlights.append(f"✅ CPM médio do bid subiu {deltas['avg_cpm_abs']:.3f} ({fmt_pct(deltas['avg_cpm_pct'])}).")
    else:
        highlights.append(f"⚠️ CPM médio do bid caiu {abs(deltas['avg_cpm_abs']):.3f} ({fmt_pct(deltas['avg_cpm_pct'])}).")

    if deltas["total_bids_abs"] > 0:
        highlights.append(f"✅ Volume de bids subiu {fmt_int(deltas['total_bids_abs'])} ({fmt_pct(deltas['total_bids_pct'])}).")
    else:
        highlights.append(f"⚠️ Volume de bids caiu {fmt_int(abs(deltas['total_bids_abs']))} ({fmt_pct(deltas['total_bids_pct'])}).")

    if deltas["unique_bidders_abs"] > 0:
        highlights.append(f"✅ Bidders únicos aumentaram em {deltas['unique_bidders_abs']}.")
    elif deltas["unique_bidders_abs"] < 0:
        highlights.append(f"⚠️ Bidders únicos caíram em {abs(deltas['unique_bidders_abs'])}.")
    else:
        highlights.append("• Bidders únicos estáveis.")

    # Reason story
    top_reason_up = inc_r.iloc[0] if len(inc_r) else None
    top_reason_down = dec_r.iloc[0] if len(dec_r) else None
    if top_reason_up is not None and top_reason_down is not None:
        highlights.append(
            f"📌 Reasons: maior alta foi **{top_reason_up['item']}** ({top_reason_up['delta_pp']:+.2f} p.p.) "
            f"e maior queda foi **{top_reason_down['item']}** ({top_reason_down['delta_pp']:+.2f} p.p.)."
        )

    # Bidder story
    top_bidder_up = inc_b.iloc[0] if len(inc_b) else None
    top_bidder_down = dec_b.iloc[0] if len(dec_b) else None
    if top_bidder_up is not None and top_bidder_down is not None:
        highlights.append(
            f"📌 Bidders: maior alta foi **{top_bidder_up['item']}** ({top_bidder_up['delta_pp']:+.2f} p.p.) "
            f"e maior queda foi **{top_bidder_down['item']}** ({top_bidder_down['delta_pp']:+.2f} p.p.)."
        )

    result["deltas"] = deltas
    result["tables"] = {
        "bidder_up": inc_b,
        "bidder_down": dec_b,
        "reason_up": inc_r,
        "reason_down": dec_r,
        "range_up": inc_rg,
        "range_down": dec_rg,
    }
    result["highlights"] = highlights
    return result

# ----------------------------
# Main
# ----------------------------
if uploaded:
    df = pd.read_csv(uploaded)

    # Remove linha total se existir
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

    st.subheader("Resumo diário (tabela)")
    daily = build_daily_tables(df)
    st.dataframe(daily, use_container_width=True)

    st.divider()

    # Controls
    dates = sorted(df["Date"].unique())
    topn = st.slider("Top N itens", 5, 30, 10)
    mode = st.radio("Modo de visualização", ["Resumo diário (todos os dias)", "Um dia (detalhado)"], horizontal=True)

    if mode == "Resumo diário (todos os dias)":
        st.write("Diagnóstico por dia: o que melhorou, o que piorou e o que mais mudou vs dia anterior.")
        for d in dates:
            info = daily_insights(df, d, topn=topn)
            if info is None:
                continue

            title = f"{d.strftime('%Y-%m-%d')} — Resumo do dia"
            if info["prev_day"] is not None:
                title += f" (vs {info['prev_day'].strftime('%Y-%m-%d')})"

            with st.expander(title, expanded=False):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total bids", fmt_int(info["kpis"]["total_bids"]))
                c2.metric("Avg bid CPM (wavg)", f"R$ {info['kpis']['avg_cpm']:.3f}")
                c3.metric("Bidders únicos", info["kpis"]["unique_bidders"])
                c4.metric("Reasons únicas", info["kpis"]["unique_reasons"])
                c5.metric("Ranges únicos", info["kpis"]["unique_ranges"])

                for line in info["highlights"]:
                    st.write(line)

                if info["deltas"] is not None:
                    st.markdown("#### Maiores mudanças (p.p.)")
                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown("**Bidders — altas**")
                        st.dataframe(info["tables"]["bidder_up"], use_container_width=True)
                        st.markdown("**Reasons — altas**")
                        st.dataframe(info["tables"]["reason_up"], use_container_width=True)
                    with colB:
                        st.markdown("**Bidders — quedas**")
                        st.dataframe(info["tables"]["bidder_down"], use_container_width=True)
                        st.markdown("**Reasons — quedas**")
                        st.dataframe(info["tables"]["reason_down"], use_container_width=True)

                    st.markdown("**Bid ranges — mudanças**")
                    colC, colD = st.columns(2)
                    with colC:
                        st.markdown("Altas")
                        st.dataframe(info["tables"]["range_up"], use_container_width=True)
                    with colD:
                        st.markdown("Quedas")
                        st.dataframe(info["tables"]["range_down"], use_container_width=True)

    else:
        selected = st.selectbox("Escolha o dia para ver detalhes", dates, format_func=lambda d: d.strftime("%Y-%m-%d"))
        day_df = df[df["Date"] == selected].copy()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Bids", fmt_int(day_df["Bids"].sum()))
        col2.metric("Avg Bid CPM (wavg)", f"R$ {wavg_cpm(day_df):.3f}")
        col3.metric("Bidders únicos", int(day_df["Bidder"].nunique()))
        col4.metric("Reasons únicas", int(day_df["Bid rejection reason"].nunique()))

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

            st.markdown("### CPM (wavg) por reason")
            reason_cpm = day_df.groupby("Bid rejection reason").apply(wavg_cpm).sort_values(ascending=False)
            st.dataframe(reason_cpm.rename("cpm_wavg").round(3), use_container_width=True)
