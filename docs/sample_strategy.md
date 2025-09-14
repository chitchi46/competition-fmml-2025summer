RSI サンプル戦略（要点）
====================

- `get_model`: RSI 期間（例: 9/14/18）から最良を選択。
- `get_signal`: RSI を [-1, 1] に線形変換し `signal` として付与。
- 発展版: |signal| >= 0.5 のとき反転。
- `get_orders`: 固定実装（離散化→年率リスク→サイズ比→リスクパリティ→発注）。

コードは `src/strategy_rsi_sample.py` を参照。


