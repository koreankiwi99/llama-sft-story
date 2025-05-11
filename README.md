# 🦙 LLaMA-SFT-Story

This repository contains a minimal codebase for supervised fine-tuning (SFT) of a LLaMA model on movie-related data.

The fine-tuned model is used as part of a movie recommendation chatbot system.

---

## 🔗 Model on Hugging Face

You can try or download the fine-tuned model here:  
👉 [https://huggingface.co/koreankiwi99/llama-sft-story](https://huggingface.co/koreankiwi99/llama-sft-story)

---

## 🤖 Full Chatbot Pipeline

To see how this model is used in the full movie recommendation chatbot (including semantic retrieval, Cypher query generation, and UI), visit:  
👉 [https://github.com/EPFL-AI-Team/STORY](https://github.com/EPFL-AI-Team/STORY)

---

## 📂 Repo Structure

```bash
llama-sft-story/
├── configs/               # Config files for SFT training
├── scripts/               # Helper scripts (e.g. for launching jobs)
├── train_llama_sft.py     # Main training script
├── test_llama.py          # Optional evaluation script
├── requirements.txt       # Python dependencies
└── .gitignore             # Ignored files
```

---

## ⚙️ Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📎 GitHub Repo

👉 [https://github.com/koreankiwi99/llama-sft-story](https://github.com/koreankiwi99/llama-sft-story)
