# cham-pnb-experiments

卒論（CHAM に対する PNB/IDOD を用いた測定）で使用した実験コード一式。  
順方向バイアス測定・鍵ビット中立度測定・後方バイアス測定を行い、CSV/ログとして結果を出力する。

---

## 1. 目的（何のコードか）
本リポジトリは、軽量ブロック暗号 CHAM（例：CHAM-64/128）に対して以下を測定するための実装である。

- **コード1：順方向バイアス測定**  
  指定した ID/OD 条件下で、順方向の統計量（偏り）を測定する。
- **コード2：鍵ビット中立度の測定（Neutrality）**  
  鍵ビットごとの中立度（例：p_same, epsilon など）を測定し、CSVとして出力する。
- **コード3：後方バイアス測定（Backward bias）**  
  PNB（Probabilistic Neutral Bits）集合を入力として、後方方向の偏りを測定する。

※各コードの実ファイル名は以下に対応付けて記載する。  
- コード1：forward bias.cpp /  順方向バイアス測定
- コード2：IDOD key neurality.cpp /  鍵ビット中立度測定
- コード3：backward bias.cpp /後方バイアス測定
