# 🧠 Medical Chatbot using DialoGPT-Large with LoRA Fine-tuning

This project fine-tunes the `microsoft/DialoGPT-large` model using **LoRA (Low-Rank Adaptation)** and compares it with **Prompt Tuning** approaches on a medical Q&A dataset ([MedQuad](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)).

---
##Depolyed link:- https://huggingface.co/spaces/Adarsh123rv12/medical-chatbot

## 🧪 Objective

To build a compact, efficient, and medically-aware chatbot for question-answering tasks in the healthcare domain. The experiment involved testing different fine-tuning methods and selecting the best based on performance and resource efficiency.

---

## 🔬 Methods Compared

### 1.  LoRA Fine-tuning (Best)
- Lightweight parameter-efficient fine-tuning (PEFT)
- Injects trainable adapters into attention layers
- Enables full backprop without modifying original weights

### 2.  Prompt Tuning (Baseline)
- Prepends learned prompts to the input
- Very low memory usage
- Lower generation quality in multi-turn and factual answers

---

##  Evaluation Results (LoRA)

Evaluated on 100 samples using the Hugging Face `evaluate` library.

| Metric        | Value         |
|---------------|---------------|
| **Perplexity**     | 56.38          |
| **ROUGE-L Score**  | 0.9893         |
| **Avg Latency**    | 1.706 sec / query |
| **Model Size**     | 1647.34 MB    |

---

##  Why LoRA was Better

-  **Higher Accuracy**: ROUGE-L score almost perfect (0.9893)
-  **More Coherent Answers**: Better alignment with reference medical answers
-  **Lower Perplexity**: Indicates better language modeling
-  **Trainable Parameters Reduced**: No need to fine-tune all model weights
-  **Faster Iteration**: Easy to integrate with `Trainer` API

---

##  Model Details

- **Base Model**: `microsoft/DialoGPT-large`
- **Dataset**: `keivalya/MedQuad-MedicalQnADataset`
- **Fine-tuning method**: LoRA using `peft` library
- **Frameworks**: PyTorch, HuggingFace Transformers, Accelerate

---

##  How to Use the Fine-Tuned Model

Once uploaded to Hugging Face Hub, load it using:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="your-username/DialoGPT-large-medical-chat")

response = pipe("What are the symptoms of diabetes?<|endoftext|>")
print(response[0]['generated_text'])
