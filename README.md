# 🇮🇳 BharatLLM Forge

> Build your own Indian LLM by merging open-weight models + fine-tuning on Indian data.
> No billion-dollar compute budget needed.

## What This Does

Takes existing open-weight LLMs (Mistral, LLaMA 3, etc.), merges them using
TIES-DARE algorithm, extends their vocabulary for Indic scripts, then fine-tunes
on Indian datasets (IndicCorp, Samanantar, legal/agri/medical corpora).

Result: A 7B parameter model that understands Hindi, Bengali, Tamil, Marathi,
Telugu, Gujarati + English + Hinglish code-switching.

---

## Setup (Run Once)

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/bharatlm-forge
cd bharatlm-forge

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Login to HuggingFace (to download gated models)
huggingface-cli login
```

---

## Quick Start

```bash
# Step 1: Download source models
python scripts/download_models.py --models mistral-7b llama3-8b indicbert

# Step 2: Download Indian datasets
python dataset_pipeline/download_datasets.py --langs hi bn ta mr te

# Step 3: Run the merge engine
python merge_engine/run_merge.py --config configs/bharat_base_merge.yaml

# Step 4: Fine-tune on Indian data
python finetune/train.py --config configs/finetune_bharat.yaml

# Step 5: Evaluate
python eval/evaluate.py --model outputs/bharat-llm-7b

# Step 6: Chat with your model!
python scripts/chat.py --model outputs/bharat-llm-7b
```

---

## Project Structure

```
bharatlm/
├── merge_engine/
│   ├── ties_dare.py          # TIES-DARE merge algorithm (core)
│   ├── slerp.py              # SLERP interpolation
│   ├── frankenmerge.py       # Layer-wise model composition
│   ├── vocab_fusion.py       # Indic vocabulary extension
│   └── run_merge.py          # Main merge runner
├── dataset_pipeline/
│   ├── download_datasets.py  # Fetch IndicCorp, Samanantar, etc.
│   ├── lang_detect.py        # FastText language detection
│   ├── text_cleaner.py       # Unicode normalization, noise removal
│   ├── deduplicator.py       # MinHash LSH deduplication
│   ├── quality_filter.py     # Perplexity + heuristic filtering
│   ├── tokenizer_builder.py  # SentencePiece Indic tokenizer
│   └── pipeline.py           # Full pipeline orchestrator
├── finetune/
│   ├── train.py              # QLoRA fine-tuning
│   ├── dataset_loader.py     # Load + format training data
│   └── rlhf_reward.py        # Reward model for RLHF
├── eval/
│   ├── evaluate.py           # Run all benchmarks
│   ├── indic_mmlu.py         # Indic-MMLU evaluation
│   └── human_eval.py         # Human evaluation interface
├── scripts/
│   ├── download_models.py    # Model downloader
│   └── chat.py               # Interactive chat interface
└── configs/
    ├── bharat_base_merge.yaml
    └── finetune_bharat.yaml
```

---

## Hardware Requirements

| Task | Min VRAM | Recommended | Time |
|------|----------|-------------|------|
| Download models | 0 GB | - | 2-4 hours |
| TIES-DARE merge | 24 GB | 40 GB | 30-60 min |
| Dataset processing | 8 GB RAM | 32 GB RAM | 4-8 hours |
| QLoRA fine-tuning | 16 GB | 40 GB | 12-48 hours |
| Inference (4-bit) | 6 GB | 12 GB | Real-time |

**Budget option**: Use Google Colab A100 ($10-20 for merge+finetune)
**Best option**: Runpod.io or Lambda Labs (A100 80GB ~$2/hour)
# Bharat
