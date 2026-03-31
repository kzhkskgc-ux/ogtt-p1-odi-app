# OGTT P1 / oDI Streamlit App

OGTT の 0, 30, 60, 90, 120 分の血糖・インスリン値から、以下を算出する Streamlit アプリです。

- P1 (glucose effectiveness)
- p2, p3, SI
- KGI, Ka, GI0, GI half-life
- Insulinogenic index
- Matsuda index
- oDI
- 糖吸収曲線 Ra(t)

## リポジトリ構成

- `app.py` : Streamlit 本体
- `model_core.py` : 数理モデルと一括計算ロジック
- `requirements.txt` : Python 依存関係
- `sample_input.csv` : 一括入力のサンプル
- `.streamlit/config.toml` : Streamlit の基本設定
- `.gitignore` : Git 管理から除外する設定

## 前提

- 血糖は **mg/dL**
- インスリンは **μU/mL**

## 指標の定義

- **Insulinogenic index** = `(IRI30 - IRI0) / (BG30 - BG0)`
- **Matsuda index** = `10000 / sqrt[(FPG×FIRI)×(mean PG×mean IRI)]`
- **oDI** = `Insulinogenic index × Matsuda index`

## ローカル起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 一括入力の必要列

- `ID`
- `O-BG(0)`, `O-BG(30)`, `O-BG(60)`, `O-BG(90)`, `O-BG(120)`
- `O-IRI(0)`, `O-IRI(30)`, `O-IRI(60)`, `O-IRI(90)`, `O-IRI(120)`

## GitHub に上げる手順

### 1. GitHub で新しい空のリポジトリを作成
例: `ogtt-p1-odi-app`

### 2. このフォルダに移動して Git 初期化
```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: OGTT P1/oDI Streamlit app"
```

### 3. GitHub リポジトリを接続して push
```bash
git remote add origin https://github.com/YOUR_USERNAME/ogtt-p1-odi-app.git
git push -u origin main
```

## Streamlit Community Cloud で公開する手順

1. GitHub にこのリポジトリを push する
2. Streamlit Community Cloud にサインインする
3. **Deploy an app** を選ぶ
4. GitHub リポジトリを選択する
5. Branch は `main` を選ぶ
6. Main file path は `app.py` を指定する
7. 必要なら **Advanced settings** で Python version を選ぶ
8. Deploy を押す

## Community Cloud 用の注意

- `requirements.txt` はリポジトリ直下に置いてあります
- Python 依存関係は `requirements.txt` からインストールされます
- Python version は Community Cloud の **Advanced settings** で選択します
- デプロイ後に Python version を変える場合は、通常は再デプロイが必要です

## よくあるエラー

### 1. 依存関係エラー
`requirements.txt` の不足が原因になりやすいです。

### 2. 列名エラー
一括入力は列名が完全一致している必要があります。

### 3. 数式上の NaN
- `BG30 - BG0 = 0` のとき Insulinogenic index は NaN
- Matsuda index の分母が 0 以下のとき NaN

## 研究用注記

本アプリは研究用プロトタイプです。モデル推定結果は、入力品質、補間、初期値、境界条件に影響されます。論文・学会発表に用いる場合は、推定アルゴリズムと前提条件を明記してください。
