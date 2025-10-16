# AgriBot — The Smart Agriculture Assistant

AgriBot is a focused chatbot for agriculture. It answers practical, day-to-day questions for farmers, students, and extension officers across crop management, soil fertility, irrigation, pest & disease control, postharvest handling, and sustainability. The goal is simple: quick, reliable, in-domain guidance presented in a friendly UI.

---

## Why this project

- **Real need:** Farmers often need fast, contextual answers, not long reports.
- **Scope control:** The bot stays **in domain** (agronomy). If a query is off-topic, it politely declines.
- **Lightweight stack:** Small instruction-tuned model + Streamlit UI for easy demos and grading.

---

## What AgriBot does

- Answers short agriculture Q&A with clear, actionable tips.
- Uses a small curated Q&A set to ground responses in core agronomy topics.
- Includes basic guardrails to prevent unrelated or unsafe answers.
- Provides a clean Streamlit interface with adjustable decoding settings (beams, max tokens).

---

## Data (small, curated)

- question–answer pairs covering: soil fertility, irrigation, pests/diseases, crop management, postharvest, sustainability, economics, safety, weather/climate, plus a few out-of-domain examples for guardrails.
- Split: stratified by topic buckets (≈ 22 train / 11 validation) to keep coverage balanced.
- Preprocessing: whitespace cleanup, de-dupe, standardize schema `{question, answer, intent/topic}`.
- Inputs format (instruction style),


---

## Model & training (summary)

- Models explored: `t5-small` and **`flan-t5-small`** (instruction-tuned).
- Best checkpoint: **FLAN-T5-small** with beam search (no sampling) for stable outputs.
- Training: custom `tf.GradientTape` loop, Adam, **clip-norm 1.0**, batch **8**, early stopping (patience 2).
- Inference: deterministic (beam search, `num_beams=4`) to make results reproducible.

**Observed results (validation):**
- Baseline T5-small: higher perplexity, weaker BLEU.
- **FLAN-T5-small**: noticeably better perplexity and BLEU; more on-task phrasing.

---

## App (Streamlit)

A lightweight Streamlit front end wraps the model with an input box, decoding controls, and an answer panel.

### Run locally

From your project root (Windows PowerShell shown):

```powershell
# 0) (optional) create/activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1) install deps
pip install -r requirements.txt

# 2) start the app (adjust path if your app file differs)
streamlit run agribot_streamlit/app.py
# or, if streamlit isn't found:
python -m streamlit run agribot_streamlit/app.py

.
├── agribot_streamlit/
│   ├── app.py                 # Streamlit UI
│   ├── ui_utils.py            # helpers (optional)
│   └── assets/                # logos, css, etc.
├── data/                      # local only (not committed)
├── notebooks/ or scripts/     # training / evaluation
├── artifacts/                 # saved models, logs (optional)
├── requirements.txt
└── README.md

