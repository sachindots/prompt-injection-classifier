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
