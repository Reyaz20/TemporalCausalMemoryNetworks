# TCMN

## What's real vs. simplified

|Piece|This weekend's version|Full TCMN (future work)|
|-|-|-|
|Temporal modeling|LSTM + sinusoidal positional encoding|Causal-attention temporal memory|
|Causal/confounder handling|One confounder (country) embedded separately, checked post-hoc|Learned causal graph over all state variables|
|Training data|6 real orders + 250 synthetic orders (same causal pattern injected)|Full historical order/employee data|
|Explanation layer|RAG retrieval + a scripted explanation|RAG + a live LLM call (Ollama/GPT)|

## Setup (5 minutes)

```bash
cd tcmn\_project
python3 -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run order

```bash
# 1. Generate the training dataset (6 real orders + 250 synthetic, same causal pattern)
python3 src/generate\_synthetic\_orders.py

# 2. Sanity-check the data pipeline (trajectory tensor shapes)
python3 src/data\_loader.py

# 3. Sanity-check the model's forward pass
python3 src/model.py

# 4. Train TCMN-lite and see accuracy + the confounder check
python3 src/train.py

# 5. Run the RAG explanation demo (needs internet on first run, downloads \~80MB model)
python3 src/rag\_demo.py
```

## In order

1. `data/orders.csv` — the real 6 orders from your case study.
2. `src/data\_loader.py` output — show the flat table becoming a `(N, 3, 1)` trajectory tensor.
3. `src/train.py` output — the training curve, final accuracy, and the **confounder check table**
(this is your strongest result: predicted Cancelled-probability differs by country, showing
the model picked up on the confounder you designed the causal graph around).
4. `src/rag\_demo.py` output — business-rule retrieval, then trajectory retrieval, then the
explanation stub.

## If `rag\_demo.py` fails (due to no internet in the room)

Everything else in the project (data loading, training, evaluation) has no internet dependency.
Only step 5 needs to reach huggingface.co once to download the embedding model — download it
ahead of time if you're not sure about venue wifi.

