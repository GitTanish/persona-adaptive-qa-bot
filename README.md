# 🤖 Persona-Adaptive QA Bot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19ywBAOhRfozZQ23SaEjXNJxt6TcGVZVg?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> Fine-tuning LLMs to adapt communication style based on audience persona — built on a free T4 GPU with QLoRA

---

## 🎯 What This Does

This notebook demonstrates how to build a **persona-adaptive question-answering system** that adjusts its explanation style based on the target audience (students, healthcare professionals, educators, etc.).

**Key Achievement:** 67% persona-style adaptation while maintaining 52% factual accuracy — trained in ~2 hours on a free Google Colab T4 GPU for $0.

### The Problem
Standard LLMs provide one-size-fits-all answers. The same technical concept needs:
- 👨‍🎓 **Simple explanations** for students
- 👨‍⚕️ **Clinical precision** for healthcare professionals  
- 👨‍🏫 **Pedagogical structure** for educators

### The Solution
Fine-tune Phi-3-mini (3.8B params) on persona-labeled QA pairs using QLoRA — efficient training without expensive hardware.

---

## 🚀 Quick Start

**Click the Colab badge above** → Run all cells → Get results in ~2 hours!

The notebook includes:
1. ✅ Environment setup (one cell)
2. ✅ Data loading & preprocessing
3. ✅ Model fine-tuning with QLoRA
4. ✅ Adapter merging for deployment
5. ✅ Evaluation with metrics
6. ✅ Interactive inference demo

**No local setup needed. Runs entirely in your browser.**

---

## 📊 Results

Evaluation on 100-item held-out test set:

| Metric | Score | Description |
|--------|-------|-------------|
| **Persona-Style Hit Rate** | **67.0%** | Model adapts vocabulary, tone, and depth |
| **Semantic Overlap** | **51.7%** | Maintains factual accuracy |
| Exact Match | 0.0% | Expected for generative tasks |

### Sample Outputs

**Question:** *"In the study of cartography, who are important contributors?"*

**👨‍🏫 For Educators:**
> "In the study of cartography, it is important to recognize the individuals who contributed to our understanding of geographic representation and map-making techniques..."

**📚 For General Readers:**
> "In the study of maps and their history, several key figures stand out. Tales de Mileto is believed to have made significant contributions..."

**💻 For Computer Science Students:**
> "When analyzing historical contributions to cartography from a computational perspective, we can examine how algorithmic approaches evolved..."

---

## 🏗️ How It Works

```
Salesforce Webscale-RL (2K QA pairs)
           ↓
Phi-3-mini-4k (3.8B params, 4-bit)
           ↓
QLoRA Fine-tuning (r=16, α=32)
           ↓
Merge Adapters → FP16 Model
           ↓
Production-Ready (No PEFT needed)
```

**Training Details:**
- ⚡ Time: ~120 minutes on free T4 GPU
- 💰 Cost: $0 (Google Colab free tier)
- 💾 Memory: ~8GB VRAM with 4-bit quantization
- 🎯 Method: QLoRA (99% memory savings vs full fine-tuning)

**Why 2 hours?** QLoRA trades speed for memory efficiency. Traditional full fine-tuning would require 80GB+ VRAM and cost $50-200 per run. QLoRA makes this possible on free hardware!

---

## 🔧 Technical Stack

- **Model:** `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- **Dataset:** `Salesforce/Webscale-RL` (persona-labeled QA pairs)
- **Method:** QLoRA with 4-bit NF4 quantization
- **Libraries:** 
  - 🤗 Hugging Face (Transformers, PEFT, TRL, Datasets, Accelerate)
  - bitsandbytes (quantization)
  - PyTorch

**LoRA Config:** r=16, α=32, dropout=0.05, target_modules="all-linear"

---

## 📁 What's in the Notebook

```
📓 Persona_Adaptive_QA_Training.ipynb
│
├── Step 1: Install dependencies
├── Step 2: Import libraries
├── Step 3: Load model & tokenizer (4-bit)
├── Step 4: Prepare dataset with persona labels
├── Step 5: Configure QLoRA & training
├── Step 6: Fine-tune model
├── Step 7: Merge adapters (optional)
├── Step 8: Load final model
└── Step 9: Evaluate & demo
```

**Everything runs sequentially. Just hit "Run All"!**

---

## ⏱️ Performance Notes

Training time varies by hardware:

| Hardware | Training Time | Cost |
|----------|---------------|------|
| Colab T4 (free) | ~90-120 min | $0 |
| Colab A100 (Pro) | ~15-20 min | ~$0.50 |
| Local RTX 4090 | ~30-40 min | Electricity |
| Cloud A100 | ~15-20 min | $2-4 |

**Why QLoRA is slower on T4:**
- 4-bit quantization adds computational overhead
- T4 has less compute than A100 (8.1 vs 19.5 TFLOPS)
- Memory efficiency trades off with speed

**Still impressive:** This approach democratizes LLM development — no expensive hardware needed!

---

## 🎓 What I Learned

- **Parameter-Efficient Fine-Tuning:** QLoRA reduces memory by 99% vs full fine-tuning
- **Prompt Engineering:** Designing chat templates for persona conditioning
- **Model Merging:** Combining LoRA adapters with base weights for production
- **Evaluation Design:** Creating interpretable metrics for generative style adaptation
- **Resource Optimization:** Trading computation time for memory efficiency

**Key Insight:** You don't need A100s or enterprise budgets to build production-quality AI. Parameter-efficient methods democratize LLM development.

---

## 🚀 Future Improvements

- [ ] Multi-adapter routing (separate adapters per persona)
- [ ] Larger training set (10K+ examples)
- [ ] More personas (legal, finance, engineering experts)
- [ ] Gradio demo for interactive testing
- [ ] RAG integration for up-to-date knowledge
- [ ] Flash Attention 2 for faster training

**Contributions welcome!** Open an issue or PR with your ideas.

---

## 📚 References & Credits

### Research Papers
- **Phi-3:** [Microsoft Research](https://arxiv.org/abs/2404.14219)
- **QLoRA:** [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- **Webscale-RL:** [Salesforce AI Research](https://arxiv.org/abs/2510.06499)

### Special Thanks
- **Tim Dettmers** for QLoRA and bitsandbytes
- **Younes Belkada** for PEFT/TRL contributions
- **Microsoft Research** for open-sourcing Phi-3
- **Salesforce AI Research** for the Webscale-RL dataset
- **Hugging Face** for the incredible open-source ecosystem

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Model & Dataset Licenses:**
- Phi-3: [MIT License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- Webscale-RL: [CC BY 4.0](https://huggingface.co/datasets/Salesforce/Webscale-RL)

---

## 📧 Contact

**Tanish Saroj**  
[LinkedIn](https://www.linkedin.com/in/tanishsaroj) | [Email](mailto:newb13982@gmail.com)

**Project Link:** [https://github.com/GitTanish/persona-adaptive-qa-bot](https://github.com/GitTanish/persona-adaptive-qa-bot)

---

### ⭐ Found this helpful? Give it a star!

This project proves you can do serious ML work on free hardware. If it helped you, please star the repo and share your results!

**Tags:** `llm` `fine-tuning` `qlora` `peft` `nlp` `persona-adaptation` `transformers` `huggingface` `phi3` `machine-learning`
