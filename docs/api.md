Strategy API（mlbacktester 準拠）
==============================

BaseStrategy の主メソッド
---------------------
- `preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame`
  - OHLCV の共通前処理。インデックス: `(timestamp, symbol)`。
- `get_model(self, train_df: pd.DataFrame) -> Any`
  - 学習区間でパラメータ探索。pickle 保存可能な形式を返す。
- `get_signal(self, preprocessed_df: pd.DataFrame, model: Any) -> pd.DataFrame`
  - 売買シグナル列（例: `signal`）を追加して返却。
- `get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info) -> list[Order]`
  - 固定実装を使用。ボラティリティ・サイズ比・リスクパリティで最終サイズ算出。

補助クラス
--------
- `Order(type, side, size, price, symbol, ...)`
- `AssetInfo(..., signed_pos_sizes, open_positions, ...)`

よくある制約
----------
- `signal` 列名は固定で扱う前提。
- `models` は `get_signal` 入力へそのまま渡される。


