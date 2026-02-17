# Oracle

`Oracle` は、Goで作ったシンプルな時系列未来予測アプリです。  
過去の値から次の値を予測する小型ニューラルネット（回帰）を使います。

## できること

- 1列の数値データ（CSV/テキスト）を読み込み
- 学習して未来の `N` ステップを予測
- 予測値と簡易95%レンジを表示
- ホールドアウト検証（MAE/RMSE/MAPE）
- JSON形式での結果出力
- 予測結果CSVの保存

## 実行方法

```bash
go run . -data data/sample.csv -steps 5 -lag 6 -hidden 12 -epochs 1800 -lr 0.008 -seed 42
```

### ホールドアウト検証付き

```bash
go run . -data data/sample.csv -steps 5 -holdout 6
```

### JSON出力 + CSV保存

```bash
go run . -data data/sample.csv -steps 5 -format json -out forecast.csv
```

## 入力データ形式

- 各行の「最初に解釈できる数値」を使用します
- 例:
  - `12.3`
  - `2025-01-01, 12.3` （この場合は `2025` が読まれるため非推奨）
  - `value` のようなヘッダー行は自動でスキップされます

推奨は「1行1数値」です。

## 主なオプション

- `-data`: データファイルパス
- `-steps`: 何ステップ先まで予測するか
- `-lag`: 予測に使う過去点数
- `-hidden`: 隠れ層ユニット数
- `-epochs`: 学習反復回数
- `-holdout`: 末尾何点を検証用に使うか（0で無効）
- `-lr`: 学習率
- `-seed`: 乱数シード
- `-format`: `text` または `json`
- `-out`: 予測結果CSVの保存先（省略時は保存しない）

## テスト

```bash
go test ./...
```
