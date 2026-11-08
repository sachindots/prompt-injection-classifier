# Prompt-injection classifier

Boundary scores user messages and retrieved passages before a chatbot receives them. The workbench runs locally and shows which attacks the detector missed and which benign inputs it flagged.

## Start the gateway

Clone the repository, extract the [serving checkpoint](https://github.com/sachindots/prompt-injection-classifier/releases/download/gateway-v1/boundary-checkpoint.zip) into its root, and start the server:

```sh
git clone https://github.com/sachindots/prompt-injection-classifier.git
cd prompt-injection-classifier
curl -fL -o boundary-checkpoint.zip https://github.com/sachindots/prompt-injection-classifier/releases/download/gateway-v1/boundary-checkpoint.zip
unzip boundary-checkpoint.zip
uv sync --locked
uv run python -m injclf.serve
```

Open the [workbench](http://127.0.0.1:8765/) or the [integration guide](http://127.0.0.1:8765/#how-to). Startup verifies the encoder and head against the committed policy. Rebuilding the training experiments also requires the sibling RAG dataset and the public sources below.

## Prepare the training data

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

### Observe decisions before enforcing

Keep your existing provider protections enabled. Use `shadow_call` on the backend to send every message through after inspection, including flagged messages and requests that time out. The provider call waits for inspection to finish or fail.

```python
import logging
from injclf.gateway import shadow_call

logging.basicConfig(level=logging.INFO)
reply = shadow_call(user_message, send_to_provider)
```

The `injclf.gateway` logger records `shadow allow`, `shadow block`, or `shadow inspection_error`. These events contain only the outcome. Count inspection errors separately and compare would-be blocks with labeled benign messages and attacks before switching to `guarded_call`. Enforcing requests still require an explicit allow.

### Reproduce the gateway policy

Run `uv run python -m injclf.policy` after training the original encoder. This fits linear heads over frozen encoder features using the pinned sources in `configs/gateway.toml`. The Stack Exchange title snapshot includes per-question author, link, and license. CyberSecEval attack variants and OWASP cheat sheets are split by group. Normalized text duplicates are excluded across partitions. Development subsets are used for fitting and cutoff selection. The original external RAG suite remains reserved for evaluation.

The selection criterion averages F1 on the original and gateway validation sets subject to the per-source false-positive budget. Matching scores prefer a shorter encoder. The human extraction challenge contains only positives, so report detection counts there. Sources provide the proxy labels for ordinary security questions and passages. These benchmarks measure text-level detection and cannot establish coverage for adaptive or multi-turn attacks.

Restart the server after rebuilding the policy. With an idle server running, use `uv run python -m injclf.benchmark` for warmed end-to-end HTTP latency and decision parity on the published test and external RAG sets. The server binds to loopback and limits inference to one active request per process. Under contention it returns an explicit busy response. Public hosting needs a separately operated authenticated gateway with TLS, rate limits, and load testing.

The selected six-layer gateway detected 59 of 128 grouped audit attacks and flagged 2 of 185 benign audit texts. On the external RAG suite it detected 2 of 3 attacks and flagged 0 of 156 benign texts. See the denominators and the positive-only extraction challenge in [the policy run](results/gateway.json). The [HTTP benchmark](results/gateway_latency.json) records the hardware and timing protocol.

## Public training and evaluation resources

The additional training experiment uses the published training partitions from [Agentic Prompt-Injection 5K](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k) by Enes Deniz (CC BY 4.0) and [neuralchemy’s core dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) (Apache 2.0). Both contain attack and benign examples. The run removes normalized exact-text overlap with the original corpus and reserves the public exam first. Semantic overlap and the pretraining history of external models remain uncertain.

The public exam checks different failure modes:

- [Microsoft BIPIA](https://github.com/microsoft/BIPIA) provides indirect-injection scenarios. The evaluation uses the published derived clean and attacked contexts.
- [InjecGuard / NotInject](https://github.com/leolee99/InjecGuard) measures false alarms on benign prompts containing words common in attacks.
- [Microsoft LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) provides adaptive email attacks. The sampled Phase 2 set measures attack detection only.
- [Agentic boundary pairs](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-boundary-pairs) test distinctions between benign and malicious instructions in similar settings.

The exact evaluation snapshot, per-class sampling limit, and source revisions are in `configs/quality.toml`. Examples come from the [Horizon Labs evaluation suite](https://huggingface.co/datasets/Horizon-Labs/prompt-injection-eval-suite). The [Horizon Labs specialist](https://huggingface.co/Horizon-Labs/prompt-injection-guard-small) has previously been evaluated by its publisher on this suite. These results support local comparisons and do not establish independence from every model developer.

To reproduce, run `python -m injclf.refine` in the project environment, then `python -m injclf.quality gateway`, `python -m injclf.quality refined`, and `python -m injclf.quality horizon`. Each detector's threshold is fixed before the exam runs. Counts and per-source accuracy are reported separately for attack-only, benign-only and mixed sets.

On BIPIA, the deployed gateway caught 191 of 200 attacks and blocked 155 of 178 clean passages. Horizon caught 54 of 200 and blocked 2 of 178. On the boundary-pair set, Horizon caught 119 of 120 attacks and blocked 2 of 120 benign texts. The deployed gateway caught 14 and blocked 9. Neither detector met the recall and false-alarm targets across workloads. The gateway remains the low-latency default. Test the blocking policy on representative traffic before enabling enforcement. Compare the [deployed gateway](results/quality_gateway.json), [additional fine-tuning](results/quality_refined.json), and [Horizon specialist](results/quality_horizon.json). Per-text timings in these reports include tokenization and scoring. The Horizon run used CPU and the Boundary runs used GPU.

## Does the detector catch an instruction inserted into an ordinary document?

Run `uv run python -m injclf.paired` in the training checkout. The experiment fits a TF-IDF logistic regression model on clean BIPIA documents and copies with published BIPIA attack instructions inserted. It chooses the cutoff on separate validation documents. The live gateway keeps its deployed cutoff. A pair counts as correct only when the detector allows the original and blocks its injected copy.

The primary test uses CPython source modules and posts from the 20 Newsgroups test partition. It inserts instructions from BIPIA's test templates and scores each document independently, as a chatbot integration would. The secondary report uses BIPIA test documents, many of which appeared in the earlier public benchmark. Both runs record source-level counts and split fingerprints. [Microsoft BIPIA](https://github.com/microsoft/BIPIA) · [CPython source distribution](https://www.python.org/ftp/python/3.12.0/) · [20 Newsgroups dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) · [Experiment results](results/paired.json)

On the 100 fresh pairs, the paired TF-IDF model got both decisions right for 68. It caught 76 attacks and blocked 9 clean documents. The live gateway got both decisions right for 6 pairs, caught 80 attacks and blocked 74 clean documents. The release still uses the existing gateway. The paired model needs evaluation on direct user prompts and customer workloads before it can take over blocking.
