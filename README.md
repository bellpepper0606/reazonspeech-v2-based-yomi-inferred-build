# reazonspeech-v2-based-yomi-inferred-build

`bellpepper0606/reazonspeech-v2-based-yomi-inferred` の構築時に使った処理を、参考用に整理したものです。

Hugging Face に公開しているデータセットはこちらです。

- https://huggingface.co/datasets/bellpepper0606/reazonspeech-v2-based-yomi-inferred

## これは何か

ReazonSpeech v2 の音声と文字起こしを元に、読み推定と形態素解析を行うための構築スクリプトです。

処理の流れはだいたい以下です。

1. ReazonSpeech v2 の文字起こしインデックスを読む
2. 音声シャードをシャード単位で取得する
3. 音声に対して読み推定モデルを実行する
4. 元の文字起こしに対して形態素解析を行う
5. 推定読みと形態素解析結果を照合する
6. 読み・品詞などを中間出力する

処理中に次のシャードを先読みするようにして、長時間処理でもなるべく止まりにくいようにしています。

## 注意

このスクリプトは、ReazonSpeech v2 の利用条件を確認し、同意済みの環境で実行したことを前提としています。

処理過程では ReazonSpeech v2 の元の文字起こしを参照しますが、Hugging Face で公開している派生データセットには、元字幕テキストや形態素の表層形は含めていません。

公開データセットには、元データ参照ID、読み、品詞、表層形の文字数のみを含めています。

このリポジトリは、データセット構築時の処理を説明するための参考実装です。元データの再配布を目的とするものではありません。

## ファイル

- `scripts/reazonSpeech_build_reference.py`
  - 構築時に使った処理を整理した参考スクリプトです。

## 実行について

実行には CUDA 環境、NeMo、UniDic、fugashi などが必要です。

また、対象データセットは環境変数で変更できます。

```bash
REAZONSPEECH_CONFIG=tiny python scripts/reazonSpeech_build_reference.py
```

指定できる値は `tiny`, `small`, `medium`, `large`, `all` です。

構築時は `all` を対象にしていますが、動作確認では `tiny` など小さい設定から試すのがよいと思います。

## 関連記事

Zenn 側で、データセットの概要や構築方針について説明しています。

- https://zenn.dev/bellpepper0606/articles/e028355e06ad2a
