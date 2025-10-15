# # app.py — AgriBot (Streamlit + PyTorch + FLAN-T5-small)
# # Clean, robust, and demo-ready. Deterministic decoding + domain guardrail + logging.

# import os
# import time
# import traceback
# from typing import List

# import pandas as pd
# import torch
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # ------------------------- Config (safe defaults) -------------------------
# # If you need fully offline: set MODEL_NAME to a local folder you saved with save_pretrained()
# MODEL_NAME = os.getenv("AGRIBOT_MODEL", "google/flan-t5-small")
# MAX_IN = 128
# DEFAULT_BEAMS = 4
# DEFAULT_MAX_NEW = 120

# # Lightweight domain guardrail (heuristic). Keep this simple and transparent.
# DOMAIN_KEYWORDS: List[str] = [
#     "crop", "soil", "irrigat", "pest", "weed", "harvest", "postharvest", "fertil",
#     "manure", "compost", "mulch", "farm", "agric", "seed", "variety", "maize",
#     "beans", "tomato", "sorghum", "millet", "cassava", "disease", "aphid", "lime", "ph"
# ]

# # ------------------------- Streamlit Page Setup -------------------------
# st.set_page_config(page_title="AgriBot — Agriculture Q&A", layout="centered")

# # Minimal inline styles for a clean look without build steps
# st.markdown(
#     """
#     <style>
#       .small-note { color: #64748b; font-size: 0.85rem; }
#       .answer-box { border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; background: #fff; }
#       .warn-box   { border: 1px solid #fde68a; border-radius: 12px; padding: 12px; background: #fffbeb; color:#854d0e; }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.title(" AgriBot — Agriculture Q&A")
# st.caption("Ask agriculture questions only (crops, soils, irrigation, pests, postharvest…). Out-of-domain queries are politely declined.")

# # ------------------------- Helpers -------------------------
# def is_in_domain(text: str) -> bool:
#     t = (text or "").lower()
#     return any(k in t for k in DOMAIN_KEYWORDS)

# def to_source(question: str) -> str:
#     # Keep this prompt format consistent with our training/experiments
#     return f"question: {question} domain: agriculture"

# # ------------------------- Model Loading (cached) -------------------------
# @st.cache_resource(show_spinner=True)
# def load_model_and_tokenizer(name: str):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     st.info(f"Loading model: {name} on {device}…")
#     tok = AutoTokenizer.from_pretrained(name, use_fast=True)
#     mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
#     mdl.to(device)
#     mdl.eval()
#     # Warm-up small generate to allocate and compile kernels
#     with torch.no_grad():
#         ids = tok("question: warmup domain: agriculture", return_tensors="pt").to(device)
#         _ = mdl.generate(**ids, max_new_tokens=2, num_beams=1, do_sample=False)
#     st.success(f"Model ready • device: {device}")
#     return tok, mdl, device

# # Ensure we surface errors on the page (not just the terminal)
# try:
#     tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)
# except Exception as e:
#     st.error("Model load error:\n\n" + "".join(traceback.format_exception(e)))
#     st.stop()

# # ------------------------- Sidebar Controls -------------------------
# with st.sidebar:
#     st.subheader("Settings")
#     beams = st.slider("Beams (deterministic)", 1, 8, value=DEFAULT_BEAMS, step=1)
#     max_new = st.slider("Max new tokens", 32, 256, value=DEFAULT_MAX_NEW, step=8)
#     log_csv = st.toggle("Log Q&A to CSV", value=True, help="Saves to artifacts/log.csv")
#     st.divider()
#     st.caption(f"Model: **{MODEL_NAME}**")
#     st.caption(f"Device: **{device}**")
#     st.caption("Tip: deterministic decoding (no sampling) → stable answers for grading.")

# # ------------------------- Examples -------------------------
# examples = [
#     "How can I improve soil fertility organically?",
#     "What time of day is best to irrigate tomatoes?",
#     "How do I control aphids without synthetic pesticides?",
#     "Which crops are suitable for crop rotation after maize?",
#     "How can I harvest rainwater for smallholder irrigation?"
# ]
# with st.expander("Quick examples"):
#     for ex in examples:
#         st.write(f"• {ex}")

# # ------------------------- Input Area -------------------------
# q = st.text_area(
#     "Your question",
#     placeholder="e.g., How can I improve soil fertility organically?",
#     height=120
# )

# col1, col2 = st.columns([1, 1])
# with col1:
#     ask = st.button("Generate", type="primary")
# with col2:
#     if st.button("Clear"):
#         st.experimental_rerun()

# # ------------------------- Inference -------------------------
# def generate_answer(question: str, beams: int, max_new_tokens: int):
#     question = (question or "").strip()
#     if not question:
#         return {"ok": False, "error": "Please enter an agriculture question."}
#     if not is_in_domain(question):
#         return {
#             "ok": True,
#             "guardrail": True,
#             "answer": ("I'm focused on agriculture topics like crops, soils, irrigation, and pests. "
#                        "Please ask a farming-related question.")
#         }

#     t0 = time.time()
#     with torch.no_grad():
#         enc = tokenizer(to_source(question), return_tensors="pt",
#                         truncation=True, max_length=MAX_IN).to(device)
#         out = model.generate(
#             **enc,
#             num_beams=int(beams),
#             do_sample=False,           # deterministic for stable grading
#             max_new_tokens=int(max_new_tokens)
#         )
#         ans = tokenizer.decode(out[0], skip_special_tokens=True)
#     elapsed_ms = round((time.time() - t0) * 1000)
#     return {"ok": True, "guardrail": False, "answer": ans, "elapsed_ms": elapsed_ms}

# if ask:
#     with st.spinner("Thinking…"):
#         try:
#             res = generate_answer(q, beams, max_new)
#         except Exception as e:
#             st.error("Generate error:\n\n" + "".join(traceback.format_exception(e)))
#             st.stop()

#     if not res.get("ok"):
#         st.error(res.get("error", "Something went wrong."))
#     else:
#         meta = f"Generated in {res.get('elapsed_ms','–')} ms • beams={beams} • max_new={max_new}"
#         if res.get("guardrail"):
#             st.markdown("<div class='warn-box'>Out-of-domain guardrail — please ask a farming-related question.</div>", unsafe_allow_html=True)
#         st.markdown("**Answer**")
#         st.markdown(f"<div class='answer-box'>{res['answer']}</div>", unsafe_allow_html=True)
#         st.markdown(f"<div class='small-note'>{meta}</div>", unsafe_allow_html=True)

#         # Optional logging for your report/demo
#         if log_csv:
#             try:
#                 os.makedirs("artifacts", exist_ok=True)
#                 row = {
#                     "question": q,
#                     "answer": res["answer"],
#                     "guardrail": res.get("guardrail", False),
#                     "elapsed_ms": res.get("elapsed_ms", None),
#                     "beams": beams,
#                     "max_new": max_new,
#                 }
#                 path = os.path.join("artifacts", "log.csv")
#                 if os.path.exists(path):
#                     df = pd.read_csv(path)
#                     df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
#                 else:
#                     df = pd.DataFrame([row])
#                 df.to_csv(path, index=False)
#                 st.caption(f"Logged to {path}")
#             except Exception as e:
#                 st.warning(f"Could not log to artifacts/log.csv: {e}")

# # ------------------------- Footer / Safety Note -------------------------
# st.divider()
# st.caption("AgriBot provides general agronomy guidance. Always consider local extension services and regulations.")

# app.py — AgriBot (Streamlit + PyTorch + FLAN-T5-small)
# Clean, robust, and demo-ready. Deterministic decoding + domain guardrail + logging.

import os
import time
import traceback
from typing import List

import pandas as pd
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------- Streamlit Page Setup (FIRST CALL) -------------------------
st.set_page_config(
    page_title="AgriBot — Agriculture Q&A",
    page_icon=None,          # no emoji/favicon
    layout="centered"
)

# ------------------------- Config (safe defaults) -------------------------
# If you need fully offline: set MODEL_NAME to a local folder saved via save_pretrained()
MODEL_NAME = os.getenv("AGRIBOT_MODEL", "google/flan-t5-small")
MAX_IN = 128
DEFAULT_BEAMS = 4
DEFAULT_MAX_NEW = 120

# Lightweight domain guardrail (heuristic). Keep this simple and transparent.
DOMAIN_KEYWORDS: List[str] = [
    "crop", "soil", "irrigat", "pest", "weed", "harvest", "postharvest", "fertil",
    "manure", "compost", "mulch", "farm", "agric", "seed", "variety", "maize",
    "beans", "tomato", "sorghum", "millet", "cassava", "disease", "aphid", "lime", "ph"
]

# ------------------------- Minimal styles -------------------------
st.markdown(
    """
    <style>
      .small-note { color: #64748b; font-size: 0.85rem; }
      .answer-box { border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; background: #fff; }
      .warn-box   { border: 1px solid #fde68a; border-radius: 12px; padding: 12px; background: #fffbeb; color:#854d0e; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AgriBot — Agriculture Q&A")
st.caption("Ask agriculture questions only (crops, soils, irrigation, pests, postharvest…). Out-of-domain queries are politely declined.")

# ------------------------- Helpers -------------------------
def is_in_domain(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in DOMAIN_KEYWORDS)

def to_source(question: str) -> str:
    # Keep this prompt format consistent with training/experiments
    return f"question: {question} domain: agriculture"

# ------------------------- Model Loading (cached) -------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Loading model: {name} on {device}…")
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    mdl.to(device)
    mdl.eval()
    # Warm-up small generate to allocate/compile kernels
    with torch.no_grad():
        ids = tok("question: warmup domain: agriculture", return_tensors="pt").to(device)
        _ = mdl.generate(**ids, max_new_tokens=2, num_beams=1, do_sample=False)
    st.success(f"Model ready • device: {device}")
    return tok, mdl, device

try:
    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)
except Exception as e:
    st.error("Model load error:\n\n" + "".join(traceback.format_exception(e)))
    st.stop()

# ------------------------- Sidebar Controls -------------------------
with st.sidebar:
    st.subheader("Settings")
    beams = st.slider("Beams (deterministic)", 1, 8, value=DEFAULT_BEAMS, step=1)
    max_new = st.slider("Max new tokens", 32, 256, value=DEFAULT_MAX_NEW, step=8)
    log_csv = st.toggle("Log Q&A to CSV", value=True, help="Saves to artifacts/log.csv")
    st.divider()
    st.caption(f"Model: **{MODEL_NAME}**")
    st.caption(f"Device: **{device}**")
    st.caption("Tip: deterministic decoding (no sampling) → stable answers for grading.")

# ------------------------- Examples -------------------------
examples = [
    "How can I improve soil fertility organically?",
    "What time of day is best to irrigate tomatoes?",
    "How do I control aphids without synthetic pesticides?",
    "Which crops are suitable for crop rotation after maize?",
    "How can I harvest rainwater for smallholder irrigation?"
]
with st.expander("Quick examples"):
    for ex in examples:
        st.write(ex)   # no bullet dot

# ------------------------- Input Area -------------------------
q = st.text_area(
    "Your question",
    placeholder="e.g., How can I improve soil fertility organically?",
    height=120
)

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button("Generate", type="primary")
with col2:
    if st.button("Clear"):
        st.experimental_rerun()

# ------------------------- Inference -------------------------
def generate_answer(question: str, beams: int, max_new_tokens: int):
    question = (question or "").strip()
    if not question:
        return {"ok": False, "error": "Please enter an agriculture question."}
    if not is_in_domain(question):
        return {
            "ok": True,
            "guardrail": True,
            "answer": ("I'm focused on agriculture topics like crops, soils, irrigation, and pests. "
                       "Please ask a farming-related question.")
        }

    t0 = time.time()
    with torch.no_grad():
        enc = tokenizer(to_source(question), return_tensors="pt",
                        truncation=True, max_length=MAX_IN).to(device)
        out = model.generate(
            **enc,
            num_beams=int(beams),
            do_sample=False,           # deterministic for stable grading
            max_new_tokens=int(max_new_tokens)
        )
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
    elapsed_ms = round((time.time() - t0) * 1000)
    return {"ok": True, "guardrail": False, "answer": ans, "elapsed_ms": elapsed_ms}

if ask:
    with st.spinner("Thinking…"):
        try:
            res = generate_answer(q, beams, max_new)
        except Exception as e:
            st.error("Generate error:\n\n" + "".join(traceback.format_exception(e)))
            st.stop()

    if not res.get("ok"):
        st.error(res.get("error", "Something went wrong."))
    else:
        meta = f"Generated in {res.get('elapsed_ms','–')} ms • beams={beams} • max_new={max_new}"
        if res.get("guardrail"):
            st.markdown("<div class='warn-box'>Out-of-domain guardrail — please ask a farming-related question.</div>", unsafe_allow_html=True)
        st.markdown("**Answer**")
        st.markdown(f"<div class='answer-box'>{res['answer']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-note'>{meta}</div>", unsafe_allow_html=True)

        # Optional logging for your report/demo
        if log_csv:
            try:
                os.makedirs("artifacts", exist_ok=True)
                row = {
                    "question": q,
                    "answer": res["answer"],
                    "guardrail": res.get("guardrail", False),
                    "elapsed_ms": res.get("elapsed_ms", None),
                    "beams": beams,
                    "max_new": max_new,
                }
                path = os.path.join("artifacts", "log.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    df = pd.DataFrame([row])
                df.to_csv(path, index=False)
                st.caption(f"Logged to {path}")
            except Exception as e:
                st.warning(f"Could not log to artifacts/log.csv: {e}")

# ------------------------- Footer / Safety Note -------------------------
st.divider()
st.caption("AgriBot provides general agronomy guidance. Always consider local extension services and regulations.")
