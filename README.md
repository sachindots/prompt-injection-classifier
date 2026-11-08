# Prompt-injection classifier

Build the data splits for training a detector that scores user questions and retrieved passages independently.

```sh
uv sync --locked
uv run python -m injclf.data --config configs/data.toml
uv run pytest -q
```

The config pins the deepset revision and locates the sibling RAG dataset relative to the config file. The loader keeps the published test set and stratifies the remaining development rows by label. External examples have first claim on duplicate text, followed by test examples. Deduplication ignores case and whitespace while preserving the original input for the model. Conflicting labels stop the run.

| Partition | Benign | Injection |
| --- | ---: | ---: |
| Training | 274 | 162 |
| Validation | 69 | 41 |
| Published test | 56 | 60 |
| External OWASP | 156 | 3 |

Counts and source fingerprints come from [the data run](results/data_stats.json). External benign examples include the ordinary user questions paired with malicious documents. The current external attacks are development cases in the sibling project and serve as a smoke test.

## Limitations

Deepset contains English, German, and other languages, including mixed-language texts. Its card gives no per-example language labels. Translations and paraphrases can cross partitions despite exact-text deduplication. The dataset card declares different licenses in its top-level and nested metadata. [Upstream dataset card](https://huggingface.co/datasets/deepset/prompt-injections/blob/4f61ecb038e9c3fb77e21034b22511b523772cdd/README.md)

## Run the workbench

Run `uv run python -m injclf.baselines` from the project root and open `artifacts/workbench.html` in a browser. The command trains the TF-IDF logistic regression baseline, chooses the threshold on validation, and writes the measured results to `results/tfidf.json`. The page scores pasted text on the device. Its Evaluation tab shows the saved run.

## Run the trained classifier

Run `uv run python -m injclf.serve` and open the printed local URL. This workbench uses the saved ModernBERT checkpoint for scoring. The standalone HTML in the previous section runs the lexical baseline.

To rebuild the weights on a CUDA GPU and verify the merged checkpoint on CPU, run:

```sh
uv run python -m injclf.train
uv run python -m injclf.evaluate
```

Training selects the epoch and threshold by validation F1, breaking epoch ties with validation average precision. Inference scores overlapping windows and uses the highest score for the text. The saved configuration contains the window settings and decision threshold.

| Model | Validation F1 | Published-test F1 | External RAG F1 | External benign texts flagged |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 0.941 | 0.841 | 0.071 | 78 / 156 |
| ModernBERT with LoRA | 0.988 | 0.942 | 0.049 | 116 / 156 |

The selected checkpoint caught all three external attacks. Its false positives on benign security questions make automatic blocking unsuitable for this RAG workload. Inputs from a separate, representative development set would be needed to select a threshold for that workload. See the [baseline run](results/tfidf.json), [training run](results/modernbert_lora.json), and [CPU benchmark](results/modernbert_cpu.json).
