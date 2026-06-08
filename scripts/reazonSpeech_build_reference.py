# ==========================================
# ReazonSpeech v2 読み推定データセット 構築用スクリプト
# ==========================================
#
# 本スクリプトは、bellpepper0606/reazonspeech-v2-based-yomi-inferred
# の構築時に使用した処理を整理したものです。
#
# ReazonSpeech 公式データセットローダーの定義を参照し、
# シャード単位で音声を処理します。処理後の音声シャードは、
# ローカルキャッシュから削除します。
#
# 本スクリプトは、reazon-research/reazonspeech の利用条件を確認し、
# 同意した環境で実行することを前提とします。
# 処理過程では元の文字起こしを参照しますが、公開している派生データセットには、
# 元字幕テキストおよび形態素表層形は含めていません。
# ==========================================

import contextlib
import gc
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import unicodedata
import urllib.request

import jaconv
import nemo.collections.asr as nemo_asr
import soundfile as sf
import torch
from fugashi import Tagger
from huggingface_hub import hf_hub_download
from kanjize import number2kanji
from Levenshtein import distance as lev_dist
from tqdm.auto import tqdm


# ==========================================
# 設定
# ==========================================
OUTPUT_DIR = "./output_data"
SHARD_CACHE_DIR = "./shard_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SHARD_CACHE_DIR, exist_ok=True)

# 以下の出力は構築時の中間結果です。
# 公開用データセットでは、元字幕テキストおよび形態素表層形を除外します。
OUTPUT_FILE_TSV = os.path.join(OUTPUT_DIR, "reazon_all_parakeet.tsv")
OUTPUT_FILE_JSON = os.path.join(OUTPUT_DIR, "reazon_all_parakeet.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

MODEL_ID = "kizuna-intelligence/hiragana-parakeet-tdt-ctc-0.6b-ja-beta"
MODEL_FILENAME = "hiragana-parakeet-tdt-ctc-0.6b-ja.nemo"

DATASET_CONFIG = os.environ.get("REAZONSPEECH_CONFIG", "all")
NBEST_RANGE = int(os.environ.get("NBEST_RANGE", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "24"))
BATCH_TIMEOUT_SEC = float(os.environ.get("BATCH_TIMEOUT_SEC", "10.0"))

# ReazonSpeech 公式ローダースクリプトで定義されている取得先。
# 構築時と同じ定義を参照します。
BASE_URL = "https://corpus.reazon-research.org/"
DATASET_INFO = {
    "tiny": {"tsv": "reazonspeech-v2/tsv/tiny.tsv", "audio": "reazonspeech-v2/data/{:03x}.tar", "nfiles": 1},
    "small": {"tsv": "reazonspeech-v2/tsv/small.tsv", "audio": "reazonspeech-v2/data/{:03x}.tar", "nfiles": 12},
    "medium": {"tsv": "reazonspeech-v2/tsv/medium.tsv", "audio": "reazonspeech-v2/data/{:03x}.tar", "nfiles": 116},
    "large": {"tsv": "reazonspeech-v2/tsv/large.tsv", "audio": "reazonspeech-v2/data/{:03x}.tar", "nfiles": 579},
    "all": {"tsv": "reazonspeech-v2/tsv/all.tsv", "audio": "reazonspeech-v2/data/{:03x}.tar", "nfiles": 4096},
    "small-v1": {"tsv": "reazonspeech-v1/tsv/small.tsv", "audio": "reazonspeech-v1/data/{:03x}.tar", "nfiles": 1},
    "medium-v1": {"tsv": "reazonspeech-v1/tsv/medium.tsv", "audio": "reazonspeech-v1/data/{:03x}.tar", "nfiles": 64},
    "all-v1": {"tsv": "reazonspeech-v1/tsv/all.tsv", "audio": "reazonspeech-v1/data/{:03x}.tar", "nfiles": 4096},
}

logging.getLogger("nemo_logger").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def download_file(url, dest_path, max_retry=5):
    """指定したURLをローカルに保存します。失敗時は部分ファイルを削除してリトライします。"""
    for attempt in range(max_retry):
        try:
            urllib.request.urlretrieve(url, dest_path)
            return True
        except Exception as e:
            print(f"  ダウンロード失敗 ({attempt + 1}/{max_retry}): {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt < max_retry - 1:
                time.sleep(5 * (attempt + 1))
    return False


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                ckpt = json.load(f)
            print(
                f"チェックポイントから再開: シャード {ckpt['shard_idx']}, "
                f"行 {ckpt['inner_idx']} (累計 {ckpt['total_processed']} 件)"
            )
            return ckpt
        except Exception:
            print("チェックポイントを読み込めなかったため、最初から開始します。")
    return {"shard_idx": 0, "inner_idx": 0, "total_processed": 0}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f)


def safe_extract(tar, path):
    """tar展開時に、展開先が指定ディレクトリ配下に収まることを確認します。"""
    base = os.path.abspath(path)
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(path, member.name))
        if not (target == base or target.startswith(base + os.sep)):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")
    tar.extractall(path=path)


def extract_shard(tar_path, extract_dir):
    with tarfile.open(tar_path, "r") as t:
        safe_extract(t, extract_dir)

    audio_files = []
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.endswith((".flac", ".wav")):
                audio_files.append(os.path.join(root, fname))
    return sorted(audio_files)


def _prefetch_worker(shard_idx, dataset_info, result):
    """指定シャードをバックグラウンドで取得し、展開まで行います。"""
    tar_url = BASE_URL + dataset_info["audio"].format(shard_idx)
    tar_path = os.path.join(SHARD_CACHE_DIR, f"shard_{shard_idx:04d}.tar")
    extract_dir = os.path.join(SHARD_CACHE_DIR, f"shard_{shard_idx:04d}")

    if os.path.exists(extract_dir):
        audio_files = []
        for root, _, files in os.walk(extract_dir):
            for fname in files:
                if fname.endswith((".flac", ".wav")):
                    audio_files.append(os.path.join(root, fname))
        if audio_files:
            result.update({
                "ok": True,
                "tar_path": tar_path,
                "extract_dir": extract_dir,
                "audio_files": sorted(audio_files),
            })
            return

    if not os.path.exists(tar_path):
        ok = False
        for attempt in range(5):
            if download_file(tar_url, tar_path):
                ok = True
                break
            time.sleep(5 * (attempt + 1))
        if not ok:
            result.update({"ok": False, "tar_path": tar_path})
            return

    try:
        audio_files = extract_shard(tar_path, extract_dir)
    except Exception as e:
        print(f"  先読み展開エラー (シャード {shard_idx}): {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        shutil.rmtree(extract_dir, ignore_errors=True)
        result.update({"ok": False, "tar_path": tar_path})
        return

    result.update({
        "ok": True,
        "tar_path": tar_path,
        "extract_dir": extract_dir,
        "audio_files": audio_files,
    })


def start_prefetch(shard_idx, dataset_info, total_shards):
    """指定シャードの先読みスレッドを起動します。"""
    if shard_idx >= total_shards:
        return None, None

    result = {"ok": False, "tar_path": None}
    t = threading.Thread(
        target=_prefetch_worker,
        args=(shard_idx, dataset_info, result),
        daemon=True,
    )
    t.start()
    return t, result


def load_tsv_index(tsv_path):
    """ヘッダーなし filename<TAB>transcription 形式のTSVを読み込みます。"""
    index = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                index[parts[0]] = parts[1]
    return index


def load_audio(path):
    try:
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)
        return arr, sr
    except Exception:
        import librosa
        return librosa.load(path, sr=None, mono=True)


def text_for_mecab(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\d+", lambda m: number2kanji(int(m.group())), text)


def process_batch(batch_data, model, tagger):
    results = []
    temp_files = []

    try:
        audio_paths = []
        for item in batch_data:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, item["audio"], item["sr"])
                audio_paths.append(tmp.name)
                temp_files.append(tmp.name)

        with torch.no_grad():
            with suppress_stdout_stderr():
                outputs = model.transcribe(audio_paths, batch_size=len(audio_paths))
                if outputs is None:
                    return []
                hypotheses = outputs[0] if isinstance(outputs, tuple) else outputs

        for item, hyp in zip(batch_data, hypotheses):
            try:
                raw_kanji = item["transcription"]
                if hasattr(hyp, "text"):
                    recognized_text = hyp.text
                elif isinstance(hyp, str):
                    recognized_text = hyp
                else:
                    print(f"\n認識結果の形式が不明 ({item['path']}): {hyp}")
                    continue

                if not recognized_text:
                    continue

                processed_text = text_for_mecab(raw_kanji)
                y_pred = jaconv.hira2kata(recognized_text)
                best_dist, best_yomi, best_nodes = float("inf"), "", None

                for nodes in tagger.nbestToNodeList(processed_text, NBEST_RANGE):
                    y_cand = ""
                    for node in nodes:
                        f = node.feature
                        kana = f.kana
                        if kana and kana != "*":
                            y_cand += kana
                        elif node.surface not in "。、？！ ":
                            y_cand += "".join(
                                re.findall(r"[\u30A0-\u30FF]+", jaconv.hira2kata(node.surface))
                            )
                    d = lev_dist(y_pred, y_cand)
                    if d < best_dist:
                        best_dist, best_yomi, best_nodes = d, y_cand, nodes
                        if d == 0:
                            break

                if best_nodes:
                    tokens_json = []
                    for node in best_nodes:
                        f = node.feature
                        surface = node.surface
                        kana_raw = f.kana
                        kana_hira = jaconv.kata2hira(kana_raw) if (kana_raw and kana_raw != "*") else surface
                        pos = f.pos1 if f.pos1 != "*" else "未知語"
                        lemma = f.lemma if (f.lemma and f.lemma != "*") else surface
                        tokens_json.append({
                            "surface": surface,
                            "kana": kana_hira,
                            "pos": pos,
                            "lemma": lemma,
                        })

                    results.append({
                        "id": os.path.basename(item["path"]),
                        "kanji": raw_kanji,
                        "yomi": recognized_text,
                        "yomi_gold": jaconv.kata2hira(best_yomi),
                        "dist": best_dist,
                        "tokens": tokens_json,
                    })
            except Exception as e:
                print(f"\nデータ処理エラー ({item.get('path', 'unknown')}): {e}")
                continue
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass

    return results


def reset_inference_engine():
    """推論エンジンを再初期化します。"""
    global model, tagger

    print("\n[RESET] 推論エンジンをリセット中...")

    if model is not None:
        del model
    if tagger is not None:
        del tagger

    gc.collect()
    torch.cuda.empty_cache()

    print("[RESET] モデルを再ロード...")
    model_path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILENAME, local_dir="./models")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
    model.eval()

    print("[RESET] 辞書を再ロード...")
    import unidic
    tagger = Tagger(f'-d "{unidic.DICDIR}"')

    print("[RESET] リセット完了\n")


if __name__ == "__main__":
    if DATASET_CONFIG not in DATASET_INFO:
        raise ValueError(f"Unknown DATASET_CONFIG: {DATASET_CONFIG}")

    print(f"使用デバイス: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"対象データセット: {DATASET_CONFIG} ({DATASET_INFO[DATASET_CONFIG]['nfiles']} シャード)")
    print("本スクリプトは、ReazonSpeech の利用条件に同意した環境で実行することを前提とします。")
    print("中間出力には構築時の文字起こし情報が含まれます。公開用データセットでは元字幕テキストを除外します。")

    print(f"\nモデルロード中: {MODEL_ID}")
    model_path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILENAME, local_dir="./models")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
    model.eval()

    print("辞書ロード中... (UniDic)")
    import unidic
    tagger = Tagger(f'-d "{unidic.DICDIR}"')
    print("準備完了\n")

    tsv_url = BASE_URL + DATASET_INFO[DATASET_CONFIG]["tsv"]
    tsv_path = os.path.join(SHARD_CACHE_DIR, f"{DATASET_CONFIG}.tsv")
    if not os.path.exists(tsv_path):
        print(f"TSVインデックスを取得中: {tsv_url}")
        if not download_file(tsv_url, tsv_path):
            print("TSVのダウンロードに失敗しました。終了します。")
            sys.exit(1)

    print("TSVインデックス読み込み中...")
    tsv_index = load_tsv_index(tsv_path)
    print(f"  {len(tsv_index)} 件のエントリを読み込みました")

    ckpt = load_checkpoint()

    f_tsv = open(OUTPUT_FILE_TSV, "a", encoding="utf-8")
    f_json = open(OUTPUT_FILE_JSON, "a", encoding="utf-8")
    if os.path.getsize(OUTPUT_FILE_TSV) == 0:
        f_tsv.write("id\tkanji\tyomi\tyomi_gold\tdist\n")

    dataset_info = DATASET_INFO[DATASET_CONFIG]
    total_shards = dataset_info["nfiles"]
    total_samples = len(tsv_index)
    pbar = tqdm(total=total_samples, initial=ckpt["total_processed"], desc="Total")

    current_thread, current_result = start_prefetch(ckpt["shard_idx"], dataset_info, total_shards)

    try:
        for shard_idx in range(ckpt["shard_idx"], total_shards):
            tar_path = os.path.join(SHARD_CACHE_DIR, f"shard_{shard_idx:04d}.tar")
            extract_dir = os.path.join(SHARD_CACHE_DIR, f"shard_{shard_idx:04d}")

            print(f"\n{'=' * 60}")
            print(f"シャード {shard_idx + 1}/{total_shards}")

            if current_thread is not None:
                current_thread.join()
                dl_ok = current_result["ok"]
            else:
                dl_ok = os.path.exists(tar_path)

            next_thread, next_result = start_prefetch(shard_idx + 1, dataset_info, total_shards)
            if shard_idx + 1 < total_shards:
                print(f"  次シャード ({shard_idx + 2}/{total_shards}) の取得をバックグラウンドで開始")

            if not dl_ok:
                print(f"  シャード {shard_idx} の取得/展開に失敗。スキップします。")
                ckpt["shard_idx"] = shard_idx + 1
                ckpt["inner_idx"] = 0
                save_checkpoint(ckpt)
                current_thread, current_result = next_thread, next_result
                continue

            if current_result and current_result.get("audio_files") is not None:
                audio_files = current_result["audio_files"]
                extract_dir = current_result["extract_dir"]
            else:
                os.makedirs(extract_dir, exist_ok=True)
                try:
                    audio_files = extract_shard(tar_path, extract_dir)
                except Exception as e:
                    print(f"  展開エラー: {e}. スキップします。")
                    ckpt["shard_idx"] = shard_idx + 1
                    ckpt["inner_idx"] = 0
                    save_checkpoint(ckpt)
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    current_thread, current_result = next_thread, next_result
                    continue

            print(f"  {len(audio_files)} ファイル展開完了")

            start_inner = ckpt["inner_idx"] if shard_idx == ckpt["shard_idx"] else 0
            files_to_process = audio_files[start_inner:]
            batch_data = []
            shard_pbar = tqdm(
                total=len(audio_files),
                initial=start_inner,
                desc=f"Shard {shard_idx + 1}/{total_shards}",
                leave=False,
            )

            for inner_offset, audio_path in enumerate(files_to_process):
                inner_idx = start_inner + inner_offset
                rel_key = "/".join(audio_path.replace("\\", "/").split("/")[-2:])
                transcription = tsv_index.get(rel_key) or tsv_index.get(os.path.basename(audio_path), "")

                try:
                    audio_arr, sr = load_audio(audio_path)
                except Exception as e:
                    tqdm.write(f"[SKIP] 壊れファイル: {audio_path} ({e})")
                    shard_pbar.update(1)
                    continue

                batch_data.append({
                    "path": audio_path,
                    "transcription": transcription,
                    "audio": audio_arr,
                    "sr": sr,
                })

                if len(batch_data) >= BATCH_SIZE:
                    batch_start_time = time.time()
                    batch_results = process_batch(batch_data, model, tagger)
                    elapsed_time = time.time() - batch_start_time
                    if elapsed_time > BATCH_TIMEOUT_SEC:
                        print(
                            f"\n[RESET] 処理遅延検知 "
                            f"({elapsed_time:.2f}s / {BATCH_TIMEOUT_SEC}s limit)。モデルをリセットします。"
                        )
                        reset_inference_engine()

                    n_sent = len(batch_data)
                    del batch_data
                    batch_data = []

                    for res in batch_results:
                        f_tsv.write(f"{res['id']}\t{res['kanji']}\t{res['yomi']}\t{res['yomi_gold']}\t{res['dist']}\n")
                        f_json.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_tsv.flush()
                    f_json.flush()

                    ckpt["inner_idx"] = inner_idx + 1
                    ckpt["total_processed"] += n_sent
                    save_checkpoint(ckpt)

                    pbar.update(n_sent)
                    shard_pbar.update(n_sent)

            if batch_data:
                batch_results = process_batch(batch_data, model, tagger)
                n_sent = len(batch_data)
                del batch_data
                batch_data = []

                for res in batch_results:
                    f_tsv.write(f"{res['id']}\t{res['kanji']}\t{res['yomi']}\t{res['yomi_gold']}\t{res['dist']}\n")
                    f_json.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_tsv.flush()
                f_json.flush()

                ckpt["total_processed"] += n_sent
                pbar.update(n_sent)
                shard_pbar.update(n_sent)

            shard_pbar.close()

            ckpt["shard_idx"] = shard_idx + 1
            ckpt["inner_idx"] = 0
            save_checkpoint(ckpt)
            print(f"  シャード {shard_idx + 1} 完了 (累計 {ckpt['total_processed']} 件)")

            try:
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                shutil.rmtree(extract_dir, ignore_errors=True)
                print("  キャッシュ削除完了")
            except Exception as e:
                print(f"  キャッシュ削除エラー（無視）: {e}")

            gc.collect()
            torch.cuda.empty_cache()
            current_thread, current_result = next_thread, next_result

    except KeyboardInterrupt:
        print("\nユーザーによる中断")

    finally:
        pbar.close()
        f_tsv.close()
        f_json.close()
        print(f"\n処理終了 (累計 {ckpt['total_processed']} 件処理済み)")
