# Prompt-injection classifier

For the gateway bundle, extract the archive, run `uv sync --locked` from its root, then `uv run python -m injclf.serve`. Open [the integration guide](http://127.0.0.1:8765/#how-to). The bundle includes the serving checkpoint. Rebuilding the experiments also requires the sibling RAG data and the sources described below.

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

Run `uv run python -m injclf.serve` and open the printed local URL. The live workbench uses the saved gateway encoder, classification head, and decision policy for scoring. The standalone HTML in the previous section runs the lexical baseline.

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

## Put Boundary before a chatbot

Start `uv run python -m injclf.serve` from this checkout. Open [API docs](http://127.0.0.1:8765/#docs) for the wire contract and [How to](http://127.0.0.1:8765/#how-to) for a provider-independent integration. Keep the inspection call in the backend that holds the provider credentials.

```python
from injclf.gateway import guarded_call

reply = guarded_call(user_message, send_to_provider)
```

Supply your existing provider function as `send_to_provider`. A block raises `PermissionError` before that function runs. Inspection failures propagate so the caller can return a temporary error. Check each retrieved passage separately before adding it to model context. The legacy local `/score` endpoint uses the original full model and returns its raw score.

### Reproduce the gateway policy

Run `uv run python -m injclf.policy` after training the original encoder. This fits linear heads over frozen encoder features using the pinned sources in `configs/gateway.toml`. The Stack Exchange title snapshot includes per-question author, link, and license. CyberSecEval attack variants and OWASP cheat sheets are split by group. Normalized text duplicates are excluded across partitions. Development subsets are used for fitting and cutoff selection. The original external RAG suite remains reserved for evaluation.

The selection criterion averages F1 on the original and gateway validation sets subject to the per-source false-positive budget. Matching scores prefer a shorter encoder. The human extraction challenge contains only positives, so report detection counts there. Sources provide the proxy labels for ordinary security questions and passages. These benchmarks measure text-level detection and cannot establish coverage for adaptive or multi-turn attacks.

Restart the server after rebuilding the policy. With an idle server running, use `uv run python -m injclf.benchmark` for warmed end-to-end HTTP latency and decision parity on the published test and external RAG sets. The server binds to loopback and limits inference to one active request per process. Under contention it returns an explicit busy response. Public hosting needs a separately operated authenticated gateway with TLS, rate limits, and load testing.

The selected six-layer gateway detected 59 of 128 grouped audit attacks and flagged 2 of 185 benign audit texts. On the external RAG suite it detected 2 of 3 attacks and flagged 0 of 156 benign texts. See the denominators and the positive-only extraction challenge in [the policy run](results/gateway.json). The [HTTP benchmark](results/gateway_latency.json) records the hardware and timing protocol.
