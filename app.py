import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"

st.set_page_config(
    page_title="FinBERT Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide",
)

SAMPLES = {
    "Positive": (
        "Apple reported record quarterly revenue of $124.3 billion, beating Wall Street "
        "expectations on strong iPhone and Services growth, and raised its quarterly dividend."
    ),
    "Negative": (
        "Shares of the automaker plunged 12% in premarket trading after the company cut its "
        "full-year profit outlook amid rising supply chain costs and weaker European demand."
    ),
    "Neutral": (
        "The company announced that its annual general meeting of shareholders will be held "
        "on May 15 at 10:00 a.m. local time at its corporate headquarters."
    ),
}


@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    tokenizer, model, device = load_model()
    all_probs = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


def main():
    st.title("📈 Financial Sentiment Analysis with FinBERT & LIME")
    st.markdown(
        "Analyze financial news headlines or paragraphs using **ProsusAI/finbert** "
        "and optionally inspect word-level feature importance with **LIME**."
    )

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = SAMPLES["Positive"]

    st.subheader("Try a Sample Financial News")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 Positive Sample", use_container_width=True):
            st.session_state["input_text"] = SAMPLES["Positive"]
    with col2:
        if st.button("🔴 Negative Sample", use_container_width=True):
            st.session_state["input_text"] = SAMPLES["Negative"]
    with col3:
        if st.button("🟡 Neutral Sample", use_container_width=True):
            st.session_state["input_text"] = SAMPLES["Neutral"]

    user_text = st.text_area(
        "Financial News / Text",
        key="input_text",
        height=140,
        placeholder="Enter financial news, earnings release, or market commentary...",
    )

    with st.sidebar:
        st.header("Settings")
        enable_lime = st.checkbox("Enable LIME Explainability", value=False)
        num_features = st.slider(
            "LIME Top Words",
            min_value=5,
            max_value=20,
            value=10,
            disabled=not enable_lime,
        )
        num_samples = st.slider(
            "LIME Perturbation Samples",
            min_value=50,
            max_value=500,
            value=150,
            step=50,
            disabled=not enable_lime,
            help="Higher values give more stable LIME explanations but take longer on CPU.",
        )

    if st.button("Analyze Sentiment", type="primary"):
        text_to_analyze = user_text.strip()
        if not text_to_analyze:
            st.warning("Please enter some financial text to analyze.")
            return

        tokenizer, model, _ = load_model()
        id2label = {int(k): v.capitalize() for k, v in model.config.id2label.items()}
        class_names = [id2label[i] for i in range(len(id2label))]

        probs = predict_proba([text_to_analyze])[0]
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])

        st.subheader("Prediction Result")
        badge_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
        st.success(
            f"{badge_map.get(pred_label, '🔹')} **Predicted Sentiment:** {pred_label} ({pred_conf:.2%})"
        )

        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        ordered_labels = ["Positive", "Neutral", "Negative"]
        m_cols = st.columns(3)
        for idx, lbl in enumerate(ordered_labels):
            val = prob_dict.get(lbl, 0.0)
            m_cols[idx].metric(lbl, f"{val:.2%}")
            m_cols[idx].progress(val)

        prob_df = pd.DataFrame(
            {
                "Sentiment": ordered_labels,
                "Probability": [prob_dict.get(lbl, 0.0) for lbl in ordered_labels],
            }
        ).set_index("Sentiment")
        st.bar_chart(prob_df)

        if enable_lime:
            st.subheader("LIME Explainability")
            with st.spinner("Generating LIME explanation..."):
                explainer = LimeTextExplainer(class_names=class_names)
                exp = explainer.explain_instance(
                    text_to_analyze,
                    predict_proba,
                    num_features=num_features,
                    num_samples=num_samples,
                    labels=(pred_idx,),
                )
                weights = exp.as_list(label=pred_idx)
                lime_df = pd.DataFrame(weights, columns=["Word / Token", "Weight"])
                st.dataframe(lime_df, use_container_width=True)
                st.bar_chart(lime_df.set_index("Word / Token"))
                components.html(exp.as_html(), height=380, scrolling=True)


if __name__ == "__main__":
    main()
