# 🌟 概要

YomiToku は日本語に特化した AI 文章画像解析エンジン(Document AI)です。画像内の文字の全文 OCR およびレイアウト解析機能を有しており、画像内の文字情報や図表を認識、抽出、変換します。

- 🤖 日本語データセットで学習した 4 種類(文字位置の検知、文字列認識、レイアウト解析、表の構造認識)の AI モデルを搭載しています。4 種類のモデルはすべて独自に学習されたモデルで日本語文書に対して、高精度に推論可能です。
- 🇯🇵 各モデルは日本語の文書画像に特化して学習されており、7000 文字を超える日本語文字の認識をサポート、手書き文字、縦書きなど日本語特有のレイアウト構造の文書画像の解析も可能です。（日本語以外にも英語の文書に対しても対応しています。）
- 📈 レイアウト解析、表の構造解析、読み順推定機能により、文書画像のレイアウトの意味的構造を壊さずに情報を抽出することが可能です。
- 📄 多様な出力形式をサポートしています。html やマークダウン、json、csv のいずれかのフォーマットに変換可能です。また、文書内に含まれる図表、画像の抽出の出力も可能です。文書画像を全文検索可能なサーチャブルPDFに変換する処理もサポートしています。
- ⚡ GPU 環境で高速に動作し、効率的に文書の文字起こし解析が可能です。また、VRAM も 8GB 以内で動作し、ハイエンドな GPU を用意する必要はありません。軽量モデルを用いれば CPU でも高速に推論が可能です。

## 🙋 FAQ

### Q. インターネットに接続できない環境での動作は可能ですか？

A. 可能です。YomiToku は初回実行時に Hugging Face Hub にアクセスし、自動でモデルファイルのダウンロードを行いますが、この際にインターネットに接続します。しかし、事前に手動でダウンロードすることでインターネットへ接続できない環境でも動作可能です。詳しくは[Module Usase](module.ja.md)の「インターネットに接続できない環境での利用」を参照してください。

### Q. 商用利用は可能ですか？

YomiToku を商用環境でご利用いただく場合、以下の方法で **製品版の商用ライセンス** を提供しています。  
手書き認識の精度向上、画像の自動向き補正、レイアウト解析の強化など、**製品版のみで利用可能な追加機能** を多数搭載しています。

商用利用か否かの判断は以下のガイドラインに従い実施してください。

- **[ライセンスの商用/非商用の判断のためのガイドライン](commercial_use_guideline.ja.md)**

#### オンプレミス環境・ローカル PC での商用利用

オンプレミス環境やローカル PC での商用利用をご希望の場合は、専用の **オンプレ向け商用ライセンス** をご用意しています。  
詳細は以下よりお問い合わせください。

- <https://www.mlism.com/>

#### クラウド上での商用利用（AWS Marketplace）

YomiToku の商用版は **AWS Marketplace** でも提供しています。  
すべての解析処理は **お客様の AWS 環境内で完結** し、外部ネットワークや第三者サーバーへの送信は一切発生しません。  
機密文書・社内資料・個人情報を扱うワークロードでも安心してご利用いただけます。

提供している製品は次のとおりです。いずれも別の AWS Marketplace 製品のため、利用する製品をそれぞれサブスクライブしてください。

| 製品 | 内容 |
| :--- | :--- |
| [YomiToku-Pro - Document Analyzer](https://aws.amazon.com/marketplace/pp/prodview-64qkuwrqi4lhi) | 文書画像の OCR・レイアウト解析（通常版） |
| [YomiToku-Pro Lite - Document Analyzer](https://aws.amazon.com/marketplace/pp/prodview-zh2ubewvb6hxe) | Document Analyzer の軽量版（通常版との精度・処理時間の比較は[技術記事](https://mlism.com/blog/tech/yomitoku-pro-lite-gpu-marketplace)を参照） |
| [YomiToku-Pro - Table Semantic Parser](https://aws.amazon.com/marketplace/pp/prodview-mqprrj24bp4w6) | 帳票の表構造（セル・キーと値・グリッド）の抽出 |

製品の一覧は [AWS Marketplace の検索結果](https://aws.amazon.com/marketplace/search/results?searchTerms=yomitoku) からも確認できます。

デプロイや解析の手順は、専用のクライアントライブラリ **YomiToku-Client** の案内を参照してください。

- **YomiToku-Client の README（製品ごとの設定とクイックスタート）**  
  <https://github.com/MLism-Inc/yomitoku-client/blob/main/README.md>
- **YomiToku-Client ドキュメント（詳しい利用手順）**  
  <https://mlism-inc.github.io/yomitoku-client/>
