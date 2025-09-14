fmml-2025summer-competition
===========================

本リポジトリは、FMML 2025 Summer コンペティション用の作業リポジトリです。配布データを用いたシグナル作成、バックテスト、提出物の管理を行います。

目次
----
- リポジトリ構成
- セットアップと使い方
- 開発フロー（推奨）
- コンペの要点（超要約）
- 注意事項

リポジトリ構成
--------------

```
fmml-2025summer-competition/
├── README.md                  # このファイル
├── competition_overview.md    # コンペ概要の要約
├── docs/archive/competition_overview_clean.md
├── docs/archive/competition_overview_with_code.md
├── docs/                      # ドキュメント分割（overview/rules/api/sample_strategy）
├── data/                      # 配布データや作業生成物（Git管理外, .gitkeepのみ）
├── notebooks/                 # 分析・検証用ノートブック
└── src/                       # 提出用Strategyや実験スクリプト
```

セットアップと使い方
------------------
1. data 配下に配布データを配置します（大容量はGitにコミットしません）。
2. notebooks 配下でEDA・指標検証・バックテスト試行を行います。
3. 提出コードは src 配下に配置し、`BaseStrategy` を継承した `Strategy` を実装します。
4. 提出前にローカルで基本テストを通し、要件に合致しているかを確認します。
5. ドキュメントは `docs/` を参照。サンプル実装は `src/strategy_rsi_sample.py`。

開発フロー（推奨）
----------------
1. 仕様の確認: `competition_overview.md` を参照（詳細は `docs/archive/`）。
2. シグナル案の作成: まずは RSI ベースのサンプルから着手し、改良案を検証。
3. 再現性の確保: 乱数シード固定、依存バージョンの明記、パラメータの探索コードは `get_model` 内に記述。
4. 提出形式の準拠: `preprocess` / `get_model` / `get_signal` を実装（`get_orders` は固定実装を使用）。

コンペの要点（超要約）
--------------------
- 評価: CPCV による Sharpe Ratio の平均で一次評価。上位はフォワードテスト。
- データ: 配布データのみ使用（外部データ・手ラベリング禁止）。
- 実行時間: 前処理〜バックテスト完了まで最長 9 時間。
- テスト: 未来参照禁止、`preprocess` で最終行を削除しない等のチェックあり。

注意事項
------
- `data/` は Git 追跡外です（`.gitkeep` のみ追跡）。
- ノートブックのチェックポイントや一時生成物はコミットしないでください。
- 概要の詳細は `docs/archive/` を参照。運用は `docs/` と要約版 `competition_overview.md` を参照してください。


