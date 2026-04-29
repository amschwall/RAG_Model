# RAG_Model
# Virtual environment setup
It is recommended to run this project through a Python virtual environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```
# Dependency installation
Next, install the required dependencies within your virtural environment.

Dependencies for training:
```bash
pip install numpy==1.26.4 torch==2.2.0 transformers==4.46.0 sentencepieve=0.1.99 tqdm==4.65
```
NOTE: The dependencies for training were installed directly into the venv via the terminal.


Dependencies for evaluation:
```bash
pip install nltk, pandas, bert-score, sentence-transformers
```

# Running tokenizer training
```bash
python train_tokenizer.py
```
# Running pretraining

```bash
python pretrain.py
```

# Running finetuning

```bash
python finetune_pretrained.py
python fine_tune_only.py
```

# Evalutaing finetuned models

```bash
python evaluate.py
```

# Running RAG model (including evaluation)

```bash
python rag.py
```
# Outputs
All created files during tokenizer training, pretraining, finetuning, RAG training, and evaluation are output into the project directory.
