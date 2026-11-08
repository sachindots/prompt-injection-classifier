# Prompt-injection classifier

Fine-tune a small encoder model to detect prompt injections, measure it against off-the-shelf baselines, publish it on the Hugging Face Hub, and plug it into `owasp-rag-eval` as its `defenses.injection_score`.

**Keywords this project earns:** PyTorch, Hugging Face Transformers, fine-tuning, LoRA/PEFT, text classification, model evaluation, Hugging Face Hub.

**Depends on:** `../owasp-rag-eval` for the attack suite and the benign test questions.

## Data

- `deepset/prompt-injections` on Hugging Face. Check its size and language mix first, and record what you find in the README.
- Reserve the OWASP attack and golden files for external evaluation. In the current suite, direct attack questions and malicious documents are positive. The benign user questions accompanying injected documents and the golden questions are negative. Instructions to control assistant behavior or plant a false answer both count as injections.
- Split with a fixed seed into train, validation and test. No text may appear in two splits after normalising case and whitespace.
- Keep an extra held-out set built from the `owasp-rag-eval` attacks. It is never used for training or threshold choice. It shows how well the classifier generalises to attacks it never saw.

## Models

| Role | Model |
|---|---|
| Cheap baseline | TF-IDF + logistic regression (scikit-learn) |
| Off-the-shelf baseline | `meta-llama/Llama-Prompt-Guard-2-86M` (gated: accept the license on Hugging Face first) |
| Yours | `answerdotai/ModernBERT-base` fine-tuned with LoRA via PEFT |

## Metrics

Report every model on the same test set and the same held-out set:

- precision, recall, F1 at the chosen threshold
- ROC AUC and PR AUC
- false-positive rate on benign questions (a defense that blocks normal users is a failed defense)
- CPU latency per text, p50 and p95
- model size on disk

Choose the threshold that maximizes validation F1 for each model. Use the same strict score > threshold comparison as the RAG caller and report benign false positives with their denominator.

## Stack

Python 3.12, uv, PyTorch, `transformers`, `peft`, `datasets`, scikit-learn, `huggingface_hub`, pytest, ruff. Train on a free Colab or Kaggle GPU. The notebook only installs the package and calls `train.py`, so all real code stays in the package.

## Layout

```
src/injclf/data.py       load, normalise, dedupe, split
src/injclf/baselines.py  TF-IDF + LR, Prompt Guard
src/injclf/train.py      LoRA fine-tune
src/injclf/evaluate.py   metrics + latency, writes results/*.json
src/injclf/predict.py    injection_score(text) -> float
tests/                   split leakage, metric correctness, predict contract
results/                 committed outputs of real runs
```

## Milestones

1. **Data.** Loader, dedupe and splits, plus a test proving no text leaks across splits. Write split sizes and the label balance to `results/data_stats.json`.
2. **Baselines.** Run TF-IDF + LR and Prompt Guard through `evaluate.py`. Write `results/baselines.json`.
3. **Fine-tune.** LoRA on ModernBERT, with training config and seed in `configs/`. Write `results/modernbert_lora.json`.
4. **Threshold and latency.** Pick the threshold on validation and report test and held-out results at it. Add a CPU latency benchmark.
5. **Publish and integrate.** Push the model with a model card listing data, metrics and limitations. Implement `predict.injection_score`, then wire it into `../owasp-rag-eval/src/owasp_rag/defenses.py` and re-run that project's attack eval with `--defenses classifier`.

**Done means:**
- `make train eval` reproduces every number.
- The README results table matches `results/`.
- The model is on the Hub.
- A Limitations section covers dataset size, language mix and the fact that untested adaptive attacks exist.

## Check yourself after each milestone

- Why does PR AUC matter more than ROC AUC when positives are rare?
- What do LoRA rank and alpha control, and what did changing them do to your results?
- Why is choosing the threshold on the test set a leak?
- How does max sequence length truncation change what the model sees in a long injected document?
- Your classifier beats Prompt Guard on your test set. Why might that not hold in production?

## Resume bullet (fill from results/)

"Fine-tuned ModernBERT with LoRA (PyTorch, Hugging Face PEFT) to detect prompt injections, reaching F1 __ vs __ for Llama Prompt Guard 2 at a __% false-positive rate on benign queries, with p95 CPU latency of __ ms. Published on Hugging Face Hub and deployed as a RAG defense."

## Chatbot gateway

The original experiment selects by validation F1. Gateway decisions use the separately calibrated per-source benign-flag budget in `configs/gateway.toml`. The caller forwards text only on an explicit allow. Measure end-to-end latency and report attack detections alongside benign blocks.
