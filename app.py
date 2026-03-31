import io
import time
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from model_core import BG_COLS, IRI_COLS, REQUIRED_COLS, run_single_subject


def setup_japanese_font():
    candidate_fonts = [
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "IPAexGothic",
        "Noto Sans CJK JP",
        "TakaoGothic",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidate_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_japanese_font()

st.set_page_config(page_title="OGTT P1 / oDI App", layout="wide")

st.title("OGTTからP1（glucose effectiveness）・oDI・糖吸収曲線を算出するアプリ")
st.caption("神戸大学臨床糖尿病グループ・研究用プロトタイプ")

WEIGHT_COL = "BW"
REQUIRED_COLS_WITH_BW = ["ID", WEIGHT_COL] + [c for c in REQUIRED_COLS if c != "ID"]

with st.expander("このアプリで算出する項目"):
    st.markdown(
        """
- **P1 (p1_GE)**: 旧版OGTTモデルから推定した glucose effectiveness
- **SI**: p3 / p2
- **Insulinogenic index**: (IRI30 - IRI0) / (BG30 - BG0)
- **Matsuda index**: 10000 / sqrt[(FPG×FIRI)×(mean PG×mean IRI)]
- **oDI**: Insulinogenic index × Matsuda index
- **糖吸収曲線（比較用）**: **Ra_pred_per_kg = Ra_pred / BW**

入力単位は **血糖 mg/dL、インスリン μU/mL、体重 kg** を想定しています。  
この版では、**体重変化の影響をならすために、群内比較にも体重補正後の糖吸収曲線 `Ra_pred_per_kg` を使う**設計です。  
表示単位は **mg/dL/min/kg** としています。  
参考のため、元の **Ra_pred (mg/dL/min)** も出力データ内には残しています。
        """
    )

mode = st.sidebar.radio("モード", ["1例入力", "CSV/Excel一括入力"])

st.sidebar.markdown("---")
st.sidebar.markdown("**必要列名（一括入力）**")
st.sidebar.code("\n".join(REQUIRED_COLS_WITH_BW))



def dataframe_to_excel_bytes(df_dict: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            safe_sheet_name = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()



def add_bw_normalized_columns(fit_df: pd.DataFrame, dense_df: pd.DataFrame, bw_kg: float):
    if bw_kg is None or pd.isna(bw_kg) or float(bw_kg) <= 0:
        raise ValueError("BW は 0 より大きい値を入力してください。")

    fit_df = fit_df.copy()
    dense_df = dense_df.copy()

    fit_df["BW_kg"] = float(bw_kg)
    dense_df["BW_kg"] = float(bw_kg)

    fit_df["Ra_pred_per_kg"] = pd.to_numeric(fit_df["Ra_pred"], errors="coerce") / float(bw_kg)
    dense_df["Ra_pred_per_kg"] = pd.to_numeric(dense_df["Ra_pred"], errors="coerce") / float(bw_kg)

    return fit_df, dense_df



def prepare_single_output(subject_id: str, bw_kg: float, summary):
    fit_df, dense_df = add_bw_normalized_columns(summary.fit_table, summary.dense_table, bw_kg)
    metrics = dict(summary.metrics)
    metrics["ID"] = subject_id
    metrics["BW_kg"] = float(bw_kg)

    ra_fit = pd.to_numeric(fit_df["Ra_pred_per_kg"], errors="coerce")
    ra_dense = pd.to_numeric(dense_df["Ra_pred_per_kg"], errors="coerce")
    time_dense = pd.to_numeric(dense_df["Time_min"], errors="coerce")

    metrics["Ra_peak_per_kg"] = float(ra_dense.max()) if ra_dense.notna().any() else np.nan
    metrics["Ra_AUC_per_kg"] = float(np.trapz(ra_dense.values, time_dense.values)) if ra_dense.notna().sum() >= 2 else np.nan

    # バッチ結果でも使いやすいよう、観測点の Ra_pred_per_kg も保持
    for _, rr in fit_df.iterrows():
        t = int(rr["Time_min"])
        metrics[f"Ra_perkg_{t}"] = rr["Ra_pred_per_kg"]

    metrics_df = pd.DataFrame([metrics])
    return metrics, metrics_df, fit_df, dense_df



def show_metrics(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P1 (p1_GE)", f"{metrics['p1_GE']:.5f}")
    c2.metric("SI", f"{metrics['SI']:.5f}")
    c3.metric("Matsuda index", f"{metrics['Matsuda_index']:.3f}")
    c4.metric("oDI", f"{metrics['oDI']:.3f}" if pd.notna(metrics['oDI']) else "NaN")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "Insulinogenic index",
        f"{metrics['Insulinogenic_index']:.4f}" if pd.notna(metrics['Insulinogenic_index']) else "NaN"
    )
    d2.metric("Ka", f"{metrics['Ka']:.5f}")
    d3.metric("GI0", f"{metrics['GI0']:.5f}")
    d4.metric("GI half-life (min)", f"{metrics['GI_half_life_min']:.2f}")

    e1, e2, e3 = st.columns(3)
    e1.metric("BW (kg)", f"{metrics['BW_kg']:.1f}")
    e2.metric(
        "Peak Ra/BW",
        f"{metrics['Ra_peak_per_kg']:.4f}" if pd.notna(metrics['Ra_peak_per_kg']) else "NaN"
    )
    e3.metric(
        "AUC Ra/BW",
        f"{metrics['Ra_AUC_per_kg']:.4f}" if pd.notna(metrics['Ra_AUC_per_kg']) else "NaN"
    )

    with st.expander("推定パラメーター詳細"):
        show_df = pd.DataFrame([metrics]).T.reset_index()
        show_df.columns = ["Item", "Value"]
        st.dataframe(show_df, use_container_width=True)



def plot_single(fit_df, dense_df):
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(dense_df["Time_min"], dense_df["G_pred"], label="Model-predicted glucose")
    ax1.scatter(fit_df["Time_min"], fit_df["G_obs"], label="Observed glucose")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_title("Observed vs model-predicted glucose")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(dense_df["Time_min"], dense_df["Ra_pred_per_kg"], label="Ra_pred / BW")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Ra_pred_per_kg (mg/dL/min/kg)")
    ax2.set_title("糖吸収曲線（体重補正後） / Weight-normalized glucose absorption curve")
    ax2.legend()
    st.pyplot(fig2)

    with st.expander("参考: 体重補正前の糖吸収曲線"):
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        ax3.plot(dense_df["Time_min"], dense_df["Ra_pred"], label="Ra_pred")
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("Ra_pred (mg/dL/min)")
        ax3.set_title("Raw glucose absorption curve")
        ax3.legend()
        st.pyplot(fig3)



def read_uploaded_table(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)



def run_batch_with_progress(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLS_WITH_BW if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

    progress = st.progress(0, text="0 / 0")
    status = st.empty()
    elapsed = st.empty()
    preview = st.empty()

    start_time = time.time()
    n = len(df)

    results = []
    error_rows = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            subject_id = str(row["ID"])
            bw_kg = float(row[WEIGHT_COL])
            status.info(f"計算中: {subject_id} ({i}/{n})")

            try:
                summary = run_single_subject(
                    subject_id=subject_id,
                    glucose_5pt=[row[BG_COLS[0]], row[BG_COLS[1]], row[BG_COLS[2]], row[BG_COLS[3]], row[BG_COLS[4]]],
                    insulin_5pt=[row[IRI_COLS[0]], row[IRI_COLS[1]], row[IRI_COLS[2]], row[IRI_COLS[3]], row[IRI_COLS[4]]],
                )

                metrics, metrics_df, fit_df, dense_df = prepare_single_output(subject_id, bw_kg, summary)
                results.append(metrics)

                one_case_excel = dataframe_to_excel_bytes(
                    {
                        "metrics": metrics_df,
                        "fit_table": fit_df,
                        "dense_table": dense_df,
                    }
                )
                zf.writestr(f"{subject_id}_result.xlsx", one_case_excel)

            except Exception as e:
                error_rows.append({"ID": subject_id, "error": str(e)})

            frac = i / n if n > 0 else 1.0
            progress.progress(frac, text=f"{i} / {n} 例 完了")
            elapsed_sec = time.time() - start_time
            elapsed.caption(f"経過時間: {elapsed_sec:.1f} 秒")

            if results:
                preview.dataframe(pd.DataFrame(results).tail(10), use_container_width=True)

    result_df = pd.DataFrame(results)
    error_df = pd.DataFrame(error_rows)

    final_zip = io.BytesIO()
    with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf_out:
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zf_in:
            for name in zf_in.namelist():
                zf_out.writestr(name, zf_in.read(name))

        zf_out.writestr(
            "batch_results.xlsx",
            dataframe_to_excel_bytes({"batch_results": result_df})
        )
        zf_out.writestr(
            "batch_results.csv",
            result_df.to_csv(index=False).encode("utf-8-sig")
        )

        if not error_df.empty:
            zf_out.writestr(
                "batch_errors.xlsx",
                dataframe_to_excel_bytes({"errors": error_df})
            )
            zf_out.writestr(
                "batch_errors.csv",
                error_df.to_csv(index=False).encode("utf-8-sig")
            )

    final_zip.seek(0)
    progress.progress(1.0, text=f"{n} / {n} 例 完了")
    status.success("一括計算が完了しました。")

    return result_df, error_df, final_zip.getvalue()


if mode == "1例入力":
    st.subheader("1例ごとの手入力")
    left, right = st.columns([1, 1])

    with left:
        subject_id = st.text_input("ID", value="Case_001")
        bw_kg = st.number_input("BW (kg)", value=60.0, step=0.1, min_value=0.1)
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
            with st.spinner("計算中です..."):
                summary = run_single_subject(
                    subject_id=subject_id,
                    glucose_5pt=[bg0, bg30, bg60, bg90, bg120],
                    insulin_5pt=[iri0, iri30, iri60, iri90, iri120],
                )

            metrics, metrics_df, fit_df, dense_df = prepare_single_output(subject_id, bw_kg, summary)

            show_metrics(metrics)
            plot_single(fit_df, dense_df)

            with st.expander("時点別データ"):
                st.dataframe(fit_df, use_container_width=True)
            with st.expander("高密度シミュレーションデータ"):
                st.dataframe(dense_df, use_container_width=True)

            st.download_button(
                "1例の時点別CSVをダウンロード",
                data=fit_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{subject_id}_fit_table.csv",
                mime="text/csv",
            )
            st.download_button(
                "1例の高密度シミュレーションCSVをダウンロード",
                data=dense_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{subject_id}_dense_table.csv",
                mime="text/csv",
            )
            st.download_button(
                "1例の結果Excelをダウンロード",
                data=dataframe_to_excel_bytes(
                    {
                        "metrics": metrics_df,
                        "fit_table": fit_df,
                        "dense_table": dense_df,
                    }
                ),
                file_name=f"{subject_id}_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"計算に失敗しました: {e}")

else:
    st.subheader("CSV / Excel 一括入力")
    uploaded = st.file_uploader("CSV または Excel ファイルをアップロードしてください", type=["csv", "xlsx"])

    sample_df = pd.DataFrame(
        columns=REQUIRED_COLS_WITH_BW,
        data=[["Case_001", 60.0, 90, 160, 180, 150, 120, 5, 40, 55, 35, 20]],
    )

    st.download_button(
        "入力テンプレートCSVをダウンロード",
        data=sample_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ogtt_template_with_bw.csv",
        mime="text/csv",
    )

    if uploaded is not None:
        try:
            df = read_uploaded_table(uploaded)

            st.markdown("**読み込みデータ（先頭）**")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("一括計算を開始", type="primary"):
                result_df, error_df, zip_bytes = run_batch_with_progress(df)

                st.success("計算が完了しました。")
                st.dataframe(result_df, use_container_width=True)

                if not error_df.empty:
                    st.warning("一部症例でエラーがありました。")
                    st.dataframe(error_df, use_container_width=True)

                st.download_button(
                    "一括結果ZIPをダウンロード",
                    data=zip_bytes,
                    file_name="ogtt_batch_results.zip",
                    mime="application/zip",
                )

                st.download_button(
                    "結果CSVをダウンロード",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="batch_results.csv",
                    mime="text/csv",
                )

                st.download_button(
                    "結果Excelをダウンロード",
                    data=dataframe_to_excel_bytes({"batch_results": result_df}),
                    file_name="batch_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"処理に失敗しました: {e}")
