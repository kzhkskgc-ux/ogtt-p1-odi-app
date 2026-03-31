import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from model_core import BG_COLS, IRI_COLS, REQUIRED_COLS, run_batch, run_single_subject

st.set_page_config(page_title="OGTT P1 / oDI App", layout="wide")

st.title("OGTTからP1（glucose effectiveness）・oDI・糖吸収曲線を算出するアプリ")
st.caption("神戸大学臨床糖尿病グループ・研究用プロトタイプ")

with st.expander("このアプリで算出する項目"):
    st.markdown(
        """
- **P1 (p1_GE)**: 添付いただいた旧版OGTTモデルから推定した glucose effectiveness
- **SI**: p3 / p2
- **Insulinogenic index**: (IRI30 - IRI0) / (BG30 - BG0)
- **Matsuda index**: 10000 / sqrt[(FPG×FIRI)×(mean PG×mean IRI)]
- **oDI**: Insulinogenic index × Matsuda index
- **糖吸収曲線**: モデル内の Ra(t)

入力単位は **血糖 mg/dL、インスリン μU/mL** を想定しています。
        """
    )

mode = st.sidebar.radio("モード", ["1例入力", "CSV/Excel一括入力"])

st.sidebar.markdown("---")
st.sidebar.markdown("**必要列名（一括入力）**")
st.sidebar.code("\n".join(REQUIRED_COLS))


def show_metrics(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P1 (p1_GE)", f"{metrics['p1_GE']:.5f}")
    c2.metric("SI", f"{metrics['SI']:.5f}")
    c3.metric("Matsuda index", f"{metrics['Matsuda_index']:.3f}")
    c4.metric("oDI", f"{metrics['oDI']:.3f}" if pd.notna(metrics['oDI']) else "NaN")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Insulinogenic index", f"{metrics['Insulinogenic_index']:.4f}" if pd.notna(metrics['Insulinogenic_index']) else "NaN")
    d2.metric("Ka", f"{metrics['Ka']:.5f}")
    d3.metric("GI0", f"{metrics['GI0']:.5f}")
    d4.metric("GI half-life (min)", f"{metrics['GI_half_life_min']:.2f}")

    with st.expander("推定パラメーター詳細"):
        show_df = pd.DataFrame([metrics]).T.reset_index()
        show_df.columns = ["Item", "Value"]
        st.dataframe(show_df, use_container_width=True)


def plot_single(summary):
    fit_df = summary.fit_table
    dense_df = summary.dense_table

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(dense_df["Time_min"], dense_df["G_pred"], label="Model-predicted glucose")
    ax1.scatter(fit_df["Time_min"], fit_df["G_obs"], label="Observed glucose")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_title("Observed vs model-predicted glucose")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(dense_df["Time_min"], dense_df["Ra_pred"])
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Ra (model unit)")
    ax2.set_title("糖吸収曲線 / Glucose absorption curve")
    st.pyplot(fig2)


if mode == "1例入力":
    st.subheader("1例ごとの手入力")
    left, right = st.columns([1, 1])

    with left:
        subject_id = st.text_input("ID", value="Case_001")
        st.markdown("**血糖 (mg/dL)**")
        bg0 = st.number_input(BG_COLS[0], value=90.0, step=1.0)
        bg30 = st.number_input(BG_COLS[1], value=160.0, step=1.0)
        bg60 = st.number_input(BG_COLS[2], value=180.0, step=1.0)
        bg90 = st.number_input(BG_COLS[3], value=150.0, step=1.0)
        bg120 = st.number_input(BG_COLS[4], value=120.0, step=1.0)

    with right:
        st.markdown("**インスリン (μU/mL)**")
        iri0 = st.number_input(IRI_COLS[0], value=5.0, step=0.1)
        iri30 = st.number_input(IRI_COLS[1], value=40.0, step=0.1)
        iri60 = st.number_input(IRI_COLS[2], value=55.0, step=0.1)
        iri90 = st.number_input(IRI_COLS[3], value=35.0, step=0.1)
        iri120 = st.number_input(IRI_COLS[4], value=20.0, step=0.1)

    if st.button("算出する", type="primary"):
        try:
            summary = run_single_subject(
                subject_id=subject_id,
                glucose_5pt=[bg0, bg30, bg60, bg90, bg120],
                insulin_5pt=[iri0, iri30, iri60, iri90, iri120],
            )
            show_metrics(summary.metrics)
            plot_single(summary)

            with st.expander("時点別データ"):
                st.dataframe(summary.fit_table, use_container_width=True)
            with st.expander("高密度シミュレーションデータ"):
                st.dataframe(summary.dense_table, use_container_width=True)

            st.download_button(
                "1例の時点別CSVをダウンロード",
                data=summary.fit_table.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{subject_id}_fit_table.csv",
                mime="text/csv",
            )
            st.download_button(
                "1例の高密度シミュレーションCSVをダウンロード",
                data=summary.dense_table.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{subject_id}_dense_table.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"計算に失敗しました: {e}")

else:
    st.subheader("CSV / Excel 一括入力")
    uploaded = st.file_uploader("CSV または Excel ファイルをアップロードしてください", type=["csv", "xlsx"])

    sample_df = pd.DataFrame(
        columns=REQUIRED_COLS,
        data=[["Case_001", 90, 160, 180, 150, 120, 5, 40, 55, 35, 20]],
    )
    st.download_button(
        "入力テンプレートCSVをダウンロード",
        data=sample_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ogtt_template.csv",
        mime="text/csv",
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.markdown("**読み込みデータ（先頭）**")
            st.dataframe(df.head(), use_container_width=True)

            result_df, files = run_batch(df)
            st.success("計算が完了しました。")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "結果Excelをダウンロード",
                data=files["batch_results.xlsx"],
                file_name="batch_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.download_button(
                "結果CSVをダウンロード",
                data=files["batch_results.csv"],
                file_name="batch_results.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"処理に失敗しました: {e}")
