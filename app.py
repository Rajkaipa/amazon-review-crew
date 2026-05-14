# app.py
# Streamlit web UI for the review sentiment crew.

import streamlit as st
from crew_lib import analyze_review_text, parse_verdict

# ---------- Page config ----------
st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
)

# ---------- Custom styling ----------
st.markdown("""
<style>
    .verdict-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        text-align: center;
        margin: 1rem 0;
    }
    .verdict-positive { background-color: #d1f5db; border: 2px solid #2e7d32; }
    .verdict-negative { background-color: #fde0e0; border: 2px solid #c62828; }
    .verdict-neutral  { background-color: #f0f0f0; border: 2px solid #757575; }
    .verdict-label    { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .verdict-justification { font-size: 1rem; color: #444; font-style: italic; }
    .stProgress > div > div > div { background-color: #1976d2; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar with examples ----------
st.sidebar.title("🛒 About")
st.sidebar.markdown(
    "This app uses a **multi-agent CrewAI pipeline** "
    "(Analyst → Classifier) with `gpt-oss-20b` via OpenRouter "
    "to classify any product review.\n\n"
    "Backed by an evaluation harness: **73.3% accuracy** on 60 "
    "real Amazon reviews."
)

st.sidebar.divider()
st.sidebar.subheader("Try an example")

EXAMPLES = {
    "👍 Clearly positive": (
        "Absolutely love this product! Best purchase I've made all year. "
        "The build quality feels premium and it arrived in just two days. "
        "Highly recommend to anyone on the fence."
    ),
    "👎 Clearly negative": (
        "Terrible experience. Item broke after one week of normal use. "
        "Customer service refused a refund and stopped replying. "
        "Save your money."
    ),
    "🤷 Mixed / neutral": (
        "It's okay. Does what the description says, nothing more, "
        "nothing less. Average build, average price. I neither love "
        "it nor hate it."
    ),
}

# Initialize session state for the text area
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

for label, text in EXAMPLES.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.review_text = text

# ---------- Main panel ----------
st.title("🛒 Review Sentiment Analyzer")
st.markdown(
    "Paste any product review and the multi-agent crew will analyze it. "
    "Expect 20–60 seconds (free-tier LLM)."
)

review_text = st.text_area(
    "Review text",
    value=st.session_state.review_text,
    height=200,
    placeholder="e.g. 'I bought these headphones last week and...'",
    key="review_input",
)

analyze_clicked = st.button(
    "🔍 Analyze Review",
    type="primary",
    use_container_width=False,
    disabled=not review_text.strip(),
)

# ---------- Results ----------
if analyze_clicked and review_text.strip():
    with st.spinner("Agents are working… (this can take up to a minute)"):
        try:
            analysis, classification = analyze_review_text(review_text)
            sentiment, confidence, justification = parse_verdict(classification)
        except Exception as e:
            st.error(f"Something went wrong while running the crew: {e}")
            st.stop()

    if sentiment is None:
        st.warning(
            "The model produced output that couldn't be parsed. "
            "Raw classifier output:"
        )
        st.code(classification)
    else:
        # Verdict card
        color_class = {
            "POSITIVE": "verdict-positive",
            "NEGATIVE": "verdict-negative",
            "NEUTRAL":  "verdict-neutral",
        }[sentiment]
        emoji = {"POSITIVE": "👍", "NEGATIVE": "👎", "NEUTRAL": "🤷"}[sentiment]

        st.markdown(
            f"""
            <div class="verdict-card {color_class}">
                <div class="verdict-label">{emoji} {sentiment}</div>
                <div class="verdict-justification">{justification or ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence as a progress bar with a numeric label
        if confidence is not None:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(min(max(confidence, 0.0), 1.0), text="Confidence")
            with col2:
                st.metric(label="", value=f"{confidence:.0%}")

        # Detailed analysis (collapsed by default)
        with st.expander("🔬 Show detailed agent analysis"):
            st.markdown("**Analyst's structured breakdown:**")
            st.markdown(analysis or "_(no analysis captured)_")
            st.divider()
            st.markdown("**Raw classifier output:**")
            st.code(classification)

# ---------- Footer ----------
st.divider()
st.caption(
    "Built with CrewAI · MCP · OpenRouter · Streamlit. "
    "[View source on GitHub](https://github.com/Rajkaipa/amazon-review-crew)"
)