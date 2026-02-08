# UltraShape-1.0 リポジトリ構造とプログラムの役割

## 概要
UltraShape 1.0 は、3D形状生成のための2段階（粗生成 → 詳細化）パイプラインを実装したPython/PyTorchプロジェクトです。  
このドキュメントでは、リポジトリ構造と主要プログラムの役割をまとめます。

## ルート直下の構成

| パス | 役割 |
| --- | --- |
| `main.py` | PyTorch Lightningを用いた学習のエントリーポイント。設定読み込み、学習ループ、DDP/DeepSpeed設定を担う。 |
| `train.sh` | 分散学習用のシェルラッパー。`scripts/train_deepspeed.sh` を呼び出してVAE/DiT学習を起動する。 |
| `configs/` | 学習・推論用のYAML設定ファイル。`train_vae_refine.yaml`、`train_dit_refine.yaml`、`infer_dit_refine.yaml` を格納。 |
| `scripts/` | 推論・学習・データ準備用の補助スクリプト群。 |
| `ultrashape/` | 主要なPythonパッケージ本体。モデル、データローダ、前処理/後処理、ユーティリティが集約されている。 |
| `inputs/` | サンプルの入力画像や粗メッシュの格納場所。 |
| `docs/` | プロジェクトページ用の静的アセット（HTML/CSS/画像など）。 |
| `requirements.txt` | Python依存ライブラリ一覧。 |
| `LICENSE`, `Notice.txt` | ライセンス関連情報。 |

## scripts/ 配下のプログラム

| ファイル | 役割 |
| --- | --- |
| `scripts/infer_dit_refine.py` | 画像と粗メッシュを入力し、詳細化されたメッシュを生成するCLI推論スクリプト。 |
| `scripts/gradio_app.py` | Gradio UIを提供するインタラクティブ推論アプリ。モデルのキャッシュや低VRAMモードに対応。 |
| `scripts/sampling.py` | 学習用データ作成のために、メッシュから点群・SDFをサンプリングするスクリプト。 |
| `scripts/run.sh` | 推論スクリプトの実行例。`infer_dit_refine.py` を呼び出す。 |
| `scripts/train_deepspeed.sh` | DeepSpeedによる分散学習の実行ラッパー。 |
| `scripts/install_env.sh` | 環境構築用の補助スクリプト。 |

## ultrashape/ パッケージの構成

| パス | 役割 |
| --- | --- |
| `ultrashape/models/` | 学習・推論で使用するモデル群。 |
| `ultrashape/models/autoencoders/` | 形状VAEやサーフェス抽出器などのエンコーダ/デコーダ関連。 |
| `ultrashape/models/denoisers/` | DiT系のデノイザーモデル（拡散モデル本体）。 |
| `ultrashape/models/diffusion/` | Flow Matching系の学習ロジックやトランスポート関連実装。 |
| `ultrashape/data/` | Objaverse用のデータセット定義（VAE/DiT学習用のDataModule）。 |
| `ultrashape/pipelines.py` | 推論パイプライン（`UltraShapePipeline` / `DiTPipeline`）の実装。 |
| `ultrashape/preprocessors.py` | 画像のリサイズ・リセンタリングなど前処理ロジック。 |
| `ultrashape/postprocessors.py` | メッシュ簡略化・フローター除去など後処理ロジック。 |
| `ultrashape/surface_loaders.py` | メッシュからサーフェス点群やシャープエッジ点群を抽出するローダ。 |
| `ultrashape/rembg.py` | 画像背景除去（rembg）を行うユーティリティ。 |
| `ultrashape/schedulers.py` | Flow Matching拡散スケジューラ実装。 |
| `ultrashape/utils/` | EMA・ボクセル化・学習補助・可視化など共通ユーティリティ。 |

## 補足: 設定ファイルの役割

| ファイル | 役割 |
| --- | --- |
| `configs/train_vae_refine.yaml` | VAE学習用設定。 |
| `configs/train_dit_refine.yaml` | DiT詳細化モデル学習用設定。 |
| `configs/infer_dit_refine.yaml` | 推論時のモデル/パイプライン設定。 |
