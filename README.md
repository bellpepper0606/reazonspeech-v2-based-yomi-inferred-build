# reazonspeech-v2-based-yomi-inferred-build

## 概要

本リポジトリは、Hugging Face Hub で公開している `bellpepper0606/reazonspeech-v2-based-yomi-inferred` の構築時に使用した処理を整理したものです。

対象データセット:

- https://huggingface.co/datasets/bellpepper0606/reazonspeech-v2-based-yomi-inferred

ReazonSpeech v2 の音声・文字起こしデータを元に、読み推定および形態素解析を行い、読み・品詞などの情報を生成するための構築用スクリプトを含みます。

## 利用制限

本リポジトリのスクリプトは、ReazonSpeech v2 の利用条件を確認し、同意した環境で実行することを前提としています。

処理過程では ReazonSpeech v2 の元の文字起こしを参照しますが、公開している派生データセットには、元字幕テキスト（表層形）は含まれていません。

公開データセットには、元データのエントリを参照するID、読み、品詞、表層形の文字数のみを保存しています。

本リポジトリは、派生データセットの構築手順を説明するための参考実装であり、元データの再配布を目的とするものではありません。

## 処理内容

スクリプトでは、主に以下の処理を行います。

1. ReazonSpeech v2 の文字起こしインデックスを読み込む
2. 音声シャードをシャード単位で処理する
3. 音声に対して読み推定モデルを適用する
4. 文字起こしに対して形態素解析を行う
5. 推定読みと形態素解析結果を照合する
6. 読み・品詞などの中間結果を出力する

長時間の処理を想定し、シャード単位の処理、チェックポイント、次シャードの先読みを行います。

## ファイル構成

- `scripts/reazonSpeech_build_reference.py`
  - データセット構築時に使用した処理を整理した参考スクリプトです。
- `requirements.txt`
  - 主な依存パッケージの一覧です。

## 実行環境

実行には CUDA 環境、NeMo、UniDic、fugashi などが必要です。

対象とする ReazonSpeech の設定は、環境変数 `REAZONSPEECH_CONFIG` で指定できます。

```bash
REAZONSPEECH_CONFIG=tiny python scripts/reazonSpeech_build_reference.py
```

指定できる値は、`tiny`, `small`, `medium`, `large`, `all` です。

構築時は `all` を対象としました。動作確認を行う場合は、`tiny` など小さい設定を指定してください。

## 関連記事

データセットの概要や構築方針については、以下の記事で説明しています。

- https://zenn.dev/bellpepper0606/articles/e028355e06ad2a
