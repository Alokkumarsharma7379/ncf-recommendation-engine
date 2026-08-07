# NCF Recommendation Engine

> **Implicit-feedback recommendation research pipeline:** turn Amazon review events into pairwise training examples, learn user–item rankings with a PyTorch NeuMF model, and expose full-catalog recommendations through FastAPI and a minimal browser client.

This repository is a compact, inspectable implementation of **Neural Collaborative Filtering (NCF)** for implicit feedback. It focuses on the ranking problem that production recommenders actually face: observations tell us which items a user interacted with, but not which unobserved items the user disliked. Consequently, the system learns relative preference from positive–negative pairs rather than regressing one-to-five-star ratings.

> **Implementation-status note:** this document distinguishes implemented behavior from intended architecture. The training, evaluation, checkpoint, full-catalog API, and frontend paths are implemented. `src/optimization/hpo.py`, `src/optimization/quantize.py`, `src/baselines/svd.py`, `src/api/models.py`, and `configs/train.yaml` are currently empty scaffolds. Therefore Optuna search, SVD training, and INT8 export are architectural targets—not runnable repository features yet. The benchmark numbers retained below are historical project claims and have no benchmark artifact or script in the current tree.

---

## 📌 High-level architecture

### Why implicit ranking rather than rating regression?

Amazon reviews are converted to binary interaction events (`rating = 1`). This is deliberate:

- **Missing is not negative.** A user may not have seen an item; assigning a zero rating to every missing matrix cell introduces false negatives.
- **Ordering is the serving objective.** The API returns a top-$K$ list, so a pairwise objective that raises an observed item above an unobserved item is better aligned than minimizing rating RMSE.
- **Feedback is selection-biased.** Explicit stars exist only after consumption and mix preference with user-specific rating habits. Interaction occurrence is a more broadly available, if noisy, signal.
- **Sampling makes sparse learning tractable.** Rather than materializing the entire $|U|\times|I|$ matrix, preprocessing draws unobserved candidates for each positive event.

The cost is that sampled negatives are **assumed**, not verified, negatives. Popularity exposure, position bias, duplicate interactions, and accidental false negatives remain sources of bias.

### End-to-end system map

```text
Amazon Reviews 2023 / Office_Products.jsonl
  fields read: user_id, parent_asin, rating, timestamp
                         │
                         ▼
              chunked JSONL ingestion (50,000 rows)
              parent_asin → item_id; rating → 1
                         │
                         ▼
            iterative 5-core user/item pruning
                         │
                         ▼
       contiguous user/item encoding + metadata pickle
                         │
                         ▼
       leave-one-out split (one encoded item per user)
                ┌────────┴─────────┐
                │                  │
                ▼                  ▼
       training positives      held-out positive
                │              + 99 sampled items
                ▼                  │
  BPR triplets, target 4:1          └──────────────┐
  sampled once during preprocessing               │
                │                                 │
                ▼                                 │
       BPRDataset / DataLoader                    │
                │                                 │
                ▼                                 │
    NeuMF: GMF branch ║ MLP branch                │
                │                                 │
                ▼                                 │
     PyTorch loop: Adam + StepLR + BPR             │
                │                                 │
                ├──────── evaluate on 100 items ◄─┘
                │          Precision@10, NDCG@10
                ▼
   saved_models/best_model.pt (best Precision@10)
                │
       ┌────────┴──────────────────────────────┐
       │                                       │
       ▼                                       ▼
  [planned] Optuna/TPE                  FastAPI startup load
  [planned] INT8 Linear quantization           │
                                               ▼
                              score every catalog item in one batch
                                               │
                                               ▼
                                    torch.topk → JSON response
                                               │
                                               ▼
                              static HTML + JavaScript frontend
```

The requested idealized path is **raw JSONL → K-core → LOO/BPR (4:1) → NeuMF → training → Optuna → INT8 → FastAPI → frontend**. In the current code, Optuna and quantization are disconnected empty stages, and FastAPI loads the floating-point checkpoint.

### Repository map

| Path | Responsibility | Current state |
|---|---|---|
| `src/data/download.py` | Stream the Office Products review JSONL from Hugging Face | Implemented |
| `src/data/preprocess.py` | Chunk loading, 5-core filtering, encoding, split, sampling, Parquet output | Implemented |
| `src/data/dataset.py` | `NCFDataset`, `BPRDataset`, `get_loader`, `get_bpr_loader` | Implemented |
| `src/models/ncf.py` | `NeuMF` GMF/MLP fusion network | Implemented |
| `src/training/loss.py` | BCE factory and pairwise `bpr_loss` | Implemented |
| `src/training/metrics.py` | sampled-candidate Precision@$K$, NDCG@$K$, evaluation | Implemented |
| `src/training/trainer.py` | `NCFTrainer`, optimization, scheduling, evaluation, checkpointing | Implemented |
| `scripts/train.py` | fixed-configuration training entry point | Implemented |
| `src/api/inference.py` | checkpoint load and full-catalog top-$K$ scoring | Implemented |
| `src/api/main.py` | FastAPI lifespan, validation, CORS, health and prediction routes | Implemented |
| `frontend/index.html` | Tailwind/vanilla-JS demo client | Implemented |
| `src/optimization/{hpo,quantize}.py` | HPO and quantization | Empty scaffold |
| `src/baselines/svd.py` | SVD baseline | Empty scaffold |
| `configs/train.yaml` | external configuration | Empty scaffold; training uses `CFG` in Python |

---

## 🧮 Mathematical foundations

### Problem definition and notation

Let $\mathcal U$ be users, $\mathcal I$ items, and $\mathcal S\subseteq\mathcal U\times\mathcal I$ observed interactions. The implicit target is

$$
y_{ui}=\begin{cases}
1,&(u,i)\in\mathcal S,\\
0,&\text{unobserved—not necessarily disliked.}
\end{cases}
$$

The model maps $(u,i)$ to $\hat y_{ui}\in(0,1)$. During BPR training only score differences matter; the final sigmoid is nevertheless part of this implementation's `NeuMF.forward`.

### GMF branch

Independent embedding tables provide $\mathbf p_u^G,\mathbf q_i^G\in\mathbb R^{d_G}$. Their element-wise Hadamard interaction is

$$
\mathbf{\phi}^{GMF}_{ui} = \mathbf{p}_u^G \odot \mathbf{q}_i^G.
$$

Unlike classical matrix factorization, which immediately sums this vector into $\mathbf p_u^\top\mathbf q_i$, NeuMF preserves its $d_G$ coordinates for a learned fusion layer. The default $d_G=8$.

### MLP branch

The MLP uses separate embeddings $\mathbf p_u^M,\mathbf q_i^M\in\mathbb R^{32}$ because the default first layer width is 64. Its input is

$$
\mathbf z_0=[\mathbf p_u^M;\mathbf q_i^M]\in\mathbb R^{64}.
$$

For layers $\ell=1,\ldots,L$,

$$
\mathbf z_\ell=\operatorname{Dropout}_{0.2}\!\left(
\operatorname{ReLU}\!\left(
\operatorname{BN}_\ell(\mathbf W_\ell^\top\mathbf z_{\ell-1}+\mathbf b_\ell)
\right)\right),
$$

or, suppressing batch normalization and dropout for the canonical form,

$$
\mathbf{\phi}^{MLP} = a_L \left( \mathbf{W}_L^T \left( \dots a_1(\mathbf{W}_1^T [\mathbf{p}_u^M ; \mathbf{q}_i^M] + \mathbf{b}_1) \dots \right) + \mathbf{b}_L \right).
$$

With the default `layers=[64, 32, 16, 8]`, the actual dense blocks are $64\to32\to16\to8$; each block is `Linear → BatchNorm1d → ReLU → Dropout(0.2)`.

### NeuMF fusion

The branch outputs are concatenated and projected by $\mathbf h$:

$$
\hat{y}_{ui} = \sigma \left( \mathbf{h}^T [\mathbf{\phi}^{GMF}_{ui} ; \mathbf{\phi}^{MLP}_{ui}] + b_o \right),
\qquad \sigma(x)=\frac{1}{1+e^{-x}}.
$$

The default fused vector has $8+8=16$ coordinates. Embeddings are initialized from $\mathcal N(0,0.01^2)$, MLP linear weights with Xavier uniform initialization, and the output weight with Kaiming uniform initialization.

### Bayesian Personalized Ranking loss

For triplets $(u,i,j)$ in $\mathcal D$, $i$ is observed and $j$ is sampled from unobserved catalog items. Define the preference margin

$$
\Delta_{uij}=\hat y_{ui}-\hat y_{uj}.
$$

Under $P(i\succ_u j\mid\Theta)=\sigma(\Delta_{uij})$, maximum log-likelihood with a Gaussian parameter prior yields

$$
\mathcal{L}_{BPR}
=-\sum_{(u,i,j)\in\mathcal D}\ln\sigma\left(\hat y_{ui}-\hat y_{uj}\right)
+\lambda\lVert\Theta\rVert_2^2.
$$

The repository computes the mean data term with numerical stabilization,

$$
-\frac{1}{|B|}\sum_{(u,i,j)\in B}
\log\left(\sigma(\hat y_{ui}-\hat y_{uj})+10^{-8}\right),
$$

while Adam's `weight_decay=10^{-5}` supplies parameter regularization. Every batch therefore performs two NeuMF forward passes, one positive and one negative.

> **Research caveat:** applying sigmoid before the BPR difference bounds $\Delta_{uij}$ to $(-1,1)$. Many BPR implementations rank unbounded logits instead. Removing the output sigmoid for BPR can improve gradient range, but would change checkpoint semantics and API scores.

### Ranking metrics

For user $u$, let $R_u^K$ be the top-$K$ returned items, $G_u$ the relevant set, and $\mathbb 1[\cdot]$ the indicator function.

$$
HR@K=\frac{1}{|\mathcal U|}\sum_{u\in\mathcal U}\mathbb 1[R_u^K\cap G_u\neq\varnothing],
$$

$$
Precision@K=\frac{1}{|\mathcal U|}\sum_{u\in\mathcal U}\frac{|R_u^K\cap G_u|}{K}.
$$

With exactly one held-out positive per user, a hit contributes $1/K$ to repository `Precision@K`; hence $HR@K=K\cdot Precision@K$. The code does **not** report HR directly.

For relevance $rel_m$ at rank $m$,

$$
DCG@K = \sum_{m=1}^K \frac{2^{rel_m} - 1}{\log_2(m + 1)},
\qquad
NDCG@K = \frac{DCG@K}{IDCG@K}.
$$

Here one binary relevant item makes $IDCG@K=1$. A held-out item at zero-based index $r$ contributes $1/\log_2(r+2)$; a miss contributes zero. Metrics rank one positive against 99 sampled negatives, not the complete catalog.

---

## 🛠️ Dataset pipeline and preprocessing mechanics

### Source and schema

`download()` retrieves `Office_Products.jsonl` from McAuley Lab's Amazon Reviews 2023 category files. `load_raw(path, chunksize=50_000)` reads only:

| Input field | Meaning | Transformation |
|---|---|---|
| `user_id` | source user identifier | contiguous integer `user` |
| `parent_asin` | parent product identifier | renamed `item_id`, then encoded |
| `rating` | explicit review rating | overwritten with binary `1` |
| `timestamp` | event time | loaded but currently discarded before splitting |

Chunks limit ingestion memory, although the retained chunks are ultimately concatenated in RAM. Encoder dictionaries and cardinalities are serialized in `meta.pkl`; BPR triplets and tests are stored as Parquet.

> **Data-directory caveat:** the tracked `data/raw/ml-100k/` files are MovieLens 100K artifacts, whereas the active downloader/preprocessor expects Amazon `data/raw/Office_Products.jsonl`. Reproduction should run the downloader rather than infer the active source from the checked-in raw directory.

### Iterative $k$-core filtering

With default $k=5$, one pass computes user and item degrees and retains only nodes whose current degree is at least five:

$$
\mathcal U' = \{u:\deg(u)\ge k\},\qquad
\mathcal I' = \{i:\deg(i)\ge k\}.
$$

Removing low-degree users can reduce item degrees and vice versa, so `kcore_filter` repeats both filters until the number of rows stops changing. At the fixed point, every remaining user and item has at least five interaction **rows**. The implementation does not deduplicate repeated user–item pairs before counting.

### ID encoding

`encode_ids` assigns contiguous zero-based IDs in first-occurrence order. The embedding tables therefore have exactly `n_users × dimension` and `n_items × dimension` rows. API callers must provide this encoded user ID, not the original Amazon identifier. The reverse mappings are not explicitly stored, but can be reconstructed by inverting `meta["encoders"]`.

### Leave-one-out split

The intended LOO protocol keeps one interaction per user for evaluation and trains on the rest:

$$
\mathcal S_u^{train}=\mathcal S_u\setminus\{i_u^*\},\qquad
\mathcal S_u^{test}=\{i_u^*\}.
$$

This gives every retained user training history and prevents the held-out edge itself from entering a training triplet. In a temporal implementation, $i_u^*$ should be the latest event and the split simulates next-item retrieval without future-event leakage.

**Actual behavior:** `leave_one_out_split` sorts by encoded `(user, item)` and takes the greatest encoded item for each user. It does **not** use the loaded timestamp. Thus it prevents direct interaction overlap but is not a chronological split and cannot guarantee temporal leakage prevention. For production-grade offline validation, retain `timestamp`, sort by `(user, timestamp)`, and hold out the latest event.

Each test row stores the positive first and 99 uniformly sampled catalog items. The sampling excludes only that row's positive—not all of the user's known positives—so a sampled “negative” may be a training interaction. It also assumes at least 100 catalog items because sampling is without replacement.

### 4:1 BPR negative sampling

For every training-positive row, `sample_bpr_triplets` repeats `(user, pos_item)` four times and samples item IDs uniformly from $\{0,\ldots,|I|-1\}$:

$$
j\sim\operatorname{Uniform}(\mathcal I),\qquad (u,j)\notin\mathcal S^{train}.
$$

Candidates colliding with a known training pair are filtered rather than resampled. Consequently, output is **at most** four triplets per positive, not always exactly four. Sampling runs once in preprocessing with NumPy seed 42; it is **not refreshed per epoch**. The DataLoader reshuffles the fixed triplets each epoch. Per-epoch negative sampling would increase negative diversity and is a recommended extension.

### Generated artifacts

```text
data/processed/
├── train.parquet   # columns: user, pos_item, neg_item
├── test.parquet    # columns: user, pos_item, neg_items (list of 99)
└── meta.pkl        # encoders.user, encoders.item, n_users, n_items
```

`pickle` is unsafe for untrusted input. Only load `meta.pkl` produced by this pipeline.

---

## ⚡ Model engineering, training, and optimization

### Default experiment configuration

The authoritative configuration is the `CFG` dictionary in `scripts/train.py`; `configs/train.yaml` is empty.

| Parameter | Default | Operational meaning |
|---|---:|---|
| `seed` | 42 | NumPy and PyTorch RNG seed |
| `epochs` | 20 | full DataLoader passes |
| `batch_size` | 1024 | BPR triplets per update |
| `layers` | `[64, 32, 16, 8]` | MLP input/output widths |
| `latent_dim_gmf` | 8 | GMF embedding width |
| MLP dropout | 0.2 | hard-coded per dense block in `NeuMF` |
| `lr` | $10^{-3}$ | Adam learning rate |
| `weight_decay` | $10^{-5}$ | Adam L2-style regularization |
| scheduler | StepLR(10, 0.5) | halve LR every 10 epochs |
| `eval_every` | 1 | evaluate each epoch |
| `k` | 10 | ranking cutoff |
| `num_workers` | 0 | DataLoader workers |
| `loss_type` | `bpr` | pairwise training path |

`set_seed` initializes NumPy, CPU PyTorch, and all CUDA RNGs when available. It does not enable deterministic algorithms or seed DataLoader workers, so bit-for-bit GPU reproducibility is not guaranteed.

### Training lifecycle

1. Select CUDA when available, otherwise CPU.
2. Auto-run preprocessing only when `train.parquet` is absent.
3. Load metadata, Parquet training triplets, and sampled evaluation rows.
4. Instantiate `NeuMF`, Adam, StepLR, and `NCFTrainer`.
5. For each batch: clear gradients, score positives and negatives, calculate BPR loss, backpropagate, update parameters.
6. Step the scheduler after each epoch.
7. Evaluate each epoch against 100 candidates per user.
8. Save `saved_models/best_model.pt` only when **Precision@10 strictly improves**. The checkpoint contains epoch, model state, optimizer state, and metrics.

### Optuna HPO: intended design versus implementation

The dependency pins Optuna 3.6.1, but `src/optimization/hpo.py` contains no objective, sampler, study, or search space. Therefore there are no real repository search-space defaults to report and no command that launches HPO.

A production implementation should use `optuna.samplers.TPESampler(seed=42)` and validate a conditional space such as the following **proposal** (not current code):

| Variable | Suggested distribution | Rationale |
|---|---|---|
| learning rate | log-uniform $[10^{-5},10^{-2}]$ | optimizer sensitivity across orders of magnitude |
| GMF dimension | categorical `{8, 16, 32, 64}` | capacity/latency trade-off |
| MLP depth/units | categorical valid pyramids, e.g. `[64,32,16,8]`, `[128,64,32,16]`, `[256,128,64,32]` | preserves even first width and valid fusion |
| dropout | uniform $[0,0.5]$ | regularization |
| weight decay | log-uniform $[10^{-7},10^{-3}]$ | regularization strength |

The objective should maximize validation NDCG@10, report intermediate epochs for pruning, rebuild models from scratch per trial, and persist the study in a database. Search results must not be tuned against the final test set.

### Dynamic INT8 quantization

The intended CPU transformation is conceptually:

```python
quantized = torch.quantization.quantize_dynamic(
    float_model.cpu().eval(),
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

Dynamic quantization stores eligible `nn.Linear` weights as signed 8-bit values and quantizes activations dynamically at runtime; embeddings and `BatchNorm1d` are not covered by `{nn.Linear}`. Linear-weight payloads can approach a 4× reduction relative to FP32, but whole-checkpoint compression is smaller because embeddings and metadata remain floating point and quantization adds scale/zero-point overhead.

**Current state:** `src/optimization/quantize.py` is empty, no quantized checkpoint is tracked, and `src/api/inference.py` constructs ordinary `NeuMF` then loads `best_model.pt`. The often-cited CPU latency change **190 ms → 67 ms** is historical documentation, not reproducible from an included benchmark. Quantized state dictionaries may also require a quantized model construction path rather than loading into the current float architecture.

---

## 🚀 Production serving and API design

### Startup and request lifecycle

FastAPI's asynchronous lifespan calls `inference.load_model()` once before serving traffic. Loading is synchronous and must find both `data/processed/meta.pkl` and `saved_models/best_model.pt` relative to the process working directory.

For `POST /predict`:

1. **Pydantic validation** accepts an integer `user_id ≥ 0` and `1 ≤ top_k ≤ 100`.
2. **Domain validation** rejects encoded users outside `[0, n_users)`.
3. **Catalog batch construction** creates all item IDs and repeats the requested user ID to the same length.
4. **Forward pass** scores the entire catalog under `torch.no_grad()` using the float model on CUDA if available, otherwise CPU.
5. **Selection** uses `torch.topk(top_k)`—not a full sort—to identify the highest scores.
6. **Serialization** rounds scores to six decimals and validates them into `PredictResponse`.

> **Important edge case:** `top_k` is constrained to 100 by Pydantic but not clamped to `n_items`; catalogs with fewer than `top_k` items will cause `torch.topk` to fail. The inference path also does not exclude items already seen by the user.

Although the lifespan is async, `/predict` is declared with synchronous `def`, and scoring is compute-bound. FastAPI will execute such handlers in its threadpool; this is not asynchronous model execution or dynamic batching. Concurrent requests can duplicate full-catalog work.

### Endpoints

#### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

This endpoint reports process health only; it does not explicitly verify model readiness.

#### `POST /predict`

Request schema:

```json
{
  "user_id": 42,
  "top_k": 10
}
```

Response schema:

```json
{
  "user_id": 42,
  "recommendations": [
    {"item_id": 731, "score": 0.913284},
    {"item_id": 114, "score": 0.887102}
  ]
}
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 42, "top_k": 10}'
```

| Status | Meaning |
|---:|---|
| 200 | ranked recommendations returned |
| 422 | Pydantic constraint failure or encoded user out of range |
| 503 | model has not been loaded |
| 500 | unhandled checkpoint, metadata, `topk`, or inference error |

Interactive OpenAPI documentation is exposed at `http://localhost:8000/docs`.

### Frontend

`frontend/index.html` uses vanilla JavaScript, Tailwind via CDN, and an API base fixed to `http://localhost:8000`. It polls `/health`, posts encoded user IDs and top-$K$, times browser-observed round trips, and renders ranked scores. FastAPI currently permits all CORS origins; restrict this allow-list and disable wildcard credential combinations before deployment.

---

## 📊 Benchmark results and trade-offs

No benchmark script, SVD implementation, quantized artifact, model file, raw run logs, hardware description, warm-up protocol, or statistical repetitions are present. The values below are therefore **reported historical figures, not independently reproducible results from the current revision**. Model file sizes were never recorded; inventing them would be misleading.

| Model | NDCG@10 | Precision@10 | CPU latency | Model file size | Evidence / trade-off |
|---|---:|---:|---:|---:|---|
| SVD baseline | 0.6920 | 0.6320 | ~5 ms | Not recorded | Historical README only; `src/baselines/svd.py` is empty |
| NeuMF FP32 | **0.8100** | **0.7400** | 190 ms | Not recorded | Higher reported ranking quality; full catalog and neural layers increase latency |
| NeuMF dynamic INT8 | 0.8085 | 0.7390 | **67 ms** | Not recorded | Small reported quality loss and 2.84× reported speedup; quantization code/artifact absent |

Two cautions are essential:

1. Repository `Precision@10` with one relevant item is bounded by 0.1, so reported values `0.6320`, `0.7400`, and `0.7390` cannot have been produced by the current `precision_at_k` function. They may actually represent HR@10 or a different multi-positive evaluation protocol.
2. Latency depends on catalog cardinality, CPU, thread count, batch shape, cold/warm state, and concurrency. Browser “Request Latency” additionally includes HTTP and rendering overhead.

A credible benchmark should record commit SHA, data fingerprint, split, seeds, candidate protocol, CPU model, PyTorch version, thread settings, p50/p95/p99 latency after warm-up, serialized bytes, and mean/uncertainty across repeated runs.

---

## 🛠️ Complete local setup and reproduction

### Prerequisites

- Git
- Python 3.12 (the repository currently requests `3.12.1` through `.python-version`)
- Sufficient disk/RAM for the Amazon Office Products JSONL and in-memory processed frame
- Optional CUDA-capable PyTorch environment

Run commands from the repository root because data and checkpoint paths are relative.

### 1. Clone and create an isolated environment

```bash
git clone <repository-url>
cd ncf-recommendation-engine

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pandas.to_parquet` requires a Parquet engine. If installation does not bring one transitively, install PyArrow:

```bash
python -m pip install pyarrow
```

### 2. Download the active dataset

```bash
python src/data/download.py
```

Expected target: `data/raw/Office_Products.jsonl`. The downloader streams HTTP chunks and skips an existing file, but it does not checksum or resume partial downloads.

### 3. Preprocess

```bash
python src/data/preprocess.py
```

This executes chunked loading, binary conversion, iterative 5-core filtering, ID encoding, non-temporal LOO, fixed 4:1 candidate sampling, 99-negative evaluation sampling, and artifact serialization.

Optional artifact sanity check:

```bash
python - <<'PY'
import pickle
import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")
with open("data/processed/meta.pkl", "rb") as handle:
    meta = pickle.load(handle)
print(train.shape, train.columns.tolist())
print(test.shape, test.columns.tolist())
print(meta["n_users"], meta["n_items"])
PY
```

### 4. Train

```bash
python scripts/train.py
```

Edit `CFG` in `scripts/train.py` to change defaults; the YAML file is not wired in. Successful training creates `saved_models/best_model.pt`. If CUDA is present, the model and batches move to it automatically.

### 5. Start the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Smoke-test it:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 0, "top_k": 10}'
```

For development only, append `--reload`. Avoid it in performance measurements.

### 6. Launch the frontend

Do not rely on `file://` behavior; serve the static file:

```bash
python -m http.server 3000 --directory frontend
```

Open `http://localhost:3000`. The frontend expects the API at `http://localhost:8000`.

### Reproducibility checklist

- Archive the raw JSONL checksum and source URL.
- Preserve `meta.pkl` with its matching checkpoint; encoded IDs depend on input order.
- Record the Git commit, Python/PyTorch versions, hardware, and CUDA/cuDNN versions.
- Do not regenerate preprocessing between model comparison runs unless every baseline uses the identical split.
- Treat sampled-candidate metrics separately from full-catalog metrics.
- Run latency tests without Uvicorn reload and after warm-up.

---

## ⚠️ System limitations and future extensions

### Cold start

NeuMF has no usable representation for users/items absent from its encoder and no content features for new entities. Practical extensions include:

- popularity or trending fallbacks for anonymous/new users;
- item text/image/category towers and user-profile features;
- fold-in or online embedding updates;
- a feature store with explicit unknown-ID handling;
- distillation into a retrieval model that generalizes from metadata.

### Full-catalog candidate bottleneck

The API allocates a user tensor and runs NeuMF for every item per request, giving roughly $O(|I|)$ scoring cost and transfer/memory pressure. `torch.topk` avoids a complete $O(|I|\log|I|)$ sort, but it cannot eliminate scoring.

At large scale, use a two-stage architecture:

1. **Candidate retrieval:** popularity, co-visitation, or a two-tower dot-product model indexed with FAISS/HNSW/ScaNN.
2. **NeuMF re-ranking:** apply the richer interaction model to hundreds or thousands of candidates.

The nonlinear MLP interaction is not directly reducible to a single item ANN vector, which is why a separately trained/retrieval-compatible tower is useful.

### Offline evaluation gaps

- The split is item-ID ordered rather than time ordered.
- Test negatives can include known training positives.
- Sampled-negative rankings can overestimate or reorder performance versus full-catalog evaluation.
- There is no validation/test separation for tuning and final reporting.
- Repeated user–item events are not deduplicated.
- Precision naming and historical benchmark magnitude disagree.

Future work should add temporal train/validation/test splits, user-positive exclusion, full-catalog metrics where feasible, bootstrap confidence intervals, coverage/novelty/diversity measures, and counterfactual or online A/B evaluation.

### Training and modeling gaps

- Refresh BPR negatives per epoch, and consider popularity-aware, hard, or in-batch negatives.
- Return logits during BPR optimization and apply calibration only at presentation boundaries.
- Mask already-consumed items during recommendation.
- Replace the mutable default `layers` argument with an immutable value.
- Make dropout configurable and externalize configuration to the YAML file.
- Add early stopping, AMP, gradient clipping, distributed training, experiment tracking, and resume support.
- Implement and test the SVD baseline and Optuna objective before drawing model-selection conclusions.

### Serving and MLOps gaps

- Implement INT8 export/load and verify operator support and accuracy before deployment.
- Add model-version metadata, artifact checksums, schema compatibility checks, and a registry.
- Clamp `top_k`, bound batch/catalog memory, and exclude consumed items.
- Add readiness/liveness separation, structured logs, tracing, Prometheus metrics, timeouts, rate limits, authentication, and restricted CORS.
- Introduce micro-batching, caching for repeated users, replicas, and autoscaling.
- Monitor feature/interaction drift, recommendation distribution, latency percentiles, error rate, and business outcomes.
- Containerize with pinned dependencies and CI tests for preprocessing, tensor shapes, checkpoint compatibility, API contracts, and performance regression.

---

## Key public interfaces

```python
# Data
download(target_dir: str = "data/raw") -> pathlib.Path
load_raw(path: pathlib.Path, chunksize: int = 50_000) -> pandas.DataFrame
kcore_filter(df: pandas.DataFrame, k: int = 5) -> pandas.DataFrame
encode_ids(df: pandas.DataFrame) -> tuple[pandas.DataFrame, dict, dict]
leave_one_out_split(df: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]
sample_bpr_triplets(df, n_items, ratio, rng) -> pandas.DataFrame
run(raw_path=Path("data/raw/Office_Products.jsonl"), out_dir=Path("data/processed")) -> None

# Model and training
NeuMF(n_users, n_items, layers=[64, 32, 16, 8], latent_dim_gmf=8)
NeuMF.forward(user: torch.Tensor, item: torch.Tensor) -> torch.Tensor
bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor
evaluate(model, test_df, device, k=10) -> dict[str, float]
NCFTrainer.fit(train_loader, test_df, epochs, eval_every=1, k=10) -> dict[str, list]

# Serving
load_model() -> None
predict(user_id: int, top_k: int = 10) -> list[dict]
```

## Responsible interpretation

Interaction prediction is not a proxy for user welfare, item quality, or fairness. Review data reflects exposure and platform selection effects. Before production use, evaluate subgroup performance, popularity concentration, feedback loops, privacy/retention requirements, abuse modes, and mechanisms for user control and explanation.
