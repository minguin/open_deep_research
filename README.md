# Open Deep Research

Welcome to this open replication of [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/)!

Read more about this implementation's goal and methods [in our blog post](https://huggingface.co/blog/open-deep-research).

This agent achieves 55% pass@1 on GAIA validation set, vs 67% for Deep Research.

To install it, first run
```bash
pip install -r requirements.txt
```

And install smolagents dev version
```bash
pip install smolagents[dev]
```

Then you're good to go! Run the run.py script, as in:
```bash
python run.py --model-id "o1" "Your question here!"
```

## 変更点
- smolagentsのinstall及びrequirements.txtも書き換えている
  ```bash
  pip install smolagents
  ```
- モデルの変更（o1→gpt-4o-mini）
  - ```reasoning_effort="high"```を外す
  - ```HF_TOKEN```も多分不要
- 結果の日本語化
  - 途中のプロンプトを日本語に直すのはナンセンスなので、最終結果を日本語に翻訳させる
  - TODO:途中SerpAPIで結果取得するときは基本Queryが英語になっているはずなので、ここも要調整
- 実行例
  ```bash
  python run.py --model-id "gpt-4o-mini" "下記のOSSのDeepReserachが最近出てきていますが、各リポジトリのReadmeや実際のコードをすべて読んでい、どのような仕組みやLLMのモデル、技術スタックを使って実現しているか、教えて下さい。
  - https://github.com/huggingface/smolagents/tree/gaia-submission-r1/examples/open_deep_research
  - https://github.com/jina-ai/node-DeepResearch
  - https://github.com/dzhng/deep-research
  - https://github.com/langchain-ai/open_deep_research

  またこれらのライブラリはWeb検索がベースだと思っていますが、社内データのようなインターナルなデータを含めてのDeepResearchの仕組みを作っていきたいです。そういうことを考えたときに、例えばPDFやテキストベースで営業活用のノウハウテキストがあったとして、上記OSSのコードのどこの部分に手をいれると実現できそうか、教えて下さい。"
  ```