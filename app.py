import streamlit as st
import requests
import re
import torch
import trafilatura

from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Live News Summarizer",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .article-card {
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 25px;
        background-color: #ffffff;
    }

    .article-title {
        font-size: 24px;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .article-meta {
        color: #666;
        font-size: 14px;
        margin-bottom: 15px;
    }

    .summary-title {
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .summary-text {
        font-size: 16px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD BART MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    model = model.to(device)

    model.eval()

    return tokenizer, model


with st.spinner("Loading summarization model..."):

    tokenizer, model = load_model()


# ============================================================
# NEWS API KEY
# ============================================================

# For local testing:
# You can temporarily replace this with:
#
# NEWS_API_KEY = "your_api_key"
#
# For deployment, use Streamlit secrets.

try:

    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

except Exception:

    NEWS_API_KEY = ""


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(
    article,
    chunk_size=900,
    stride=100
):

    if stride >= chunk_size:

        raise ValueError(
            "Stride must be smaller than chunk size."
        )

    tokens = tokenizer(
        article,
        add_special_tokens=False,
        truncation=False
    )["input_ids"]

    chunks = []

    start = 0

    step = chunk_size - stride

    while start < len(tokens):

        end = start + chunk_size

        chunk = tokens[start:end]

        chunks.append(chunk)

        if end >= len(tokens):

            break

        start += step

    return chunks


# ============================================================
# DIRECT BART SUMMARIZATION
# ============================================================

def direct_bart_summarize(
    article,
    max_input_length=1024,
    max_summary_length=120,
    min_summary_length=30
):

    inputs = tokenizer(
        article,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        summary_ids = model.generate(
            **inputs,
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return summary


# ============================================================
# CHUNK SUMMARIZATION
# ============================================================

def summarize_chunk(
    chunk_text,
    max_input_length=1024,
    max_summary_length=150,
    min_summary_length=40
):

    inputs = tokenizer(
        chunk_text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        summary_ids = model.generate(
            **inputs,
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return summary


# ============================================================
# HIERARCHICAL BART
# ============================================================

def hierarchical_summarize(article):

    chunks = create_chunks(article)

    chunk_summaries = []

    for chunk in chunks:

        chunk_text = tokenizer.decode(
            chunk,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        summary = summarize_chunk(
            chunk_text
        )

        chunk_summaries.append(summary)

    combined_summary = " ".join(
        chunk_summaries
    )

    final_summary = direct_bart_summarize(
        combined_summary,
        max_input_length=1024,
        max_summary_length=120,
        min_summary_length=30
    )

    return final_summary


# ============================================================
# MAIN SUMMARIZATION ROUTER
# ============================================================

def summarize_news(article):

    tokens = tokenizer(
        article,
        add_special_tokens=False,
        truncation=False
    )["input_ids"]

    token_count = len(tokens)

    if token_count <= 1024:

        summary = direct_bart_summarize(
            article
        )

        method = "Direct BART"

    else:

        summary = hierarchical_summarize(
            article
        )

        method = "Hierarchical BART"

    return {

        "summary": summary,

        "method": method,

        "input_tokens": token_count

    }


# ============================================================
# CLEAN ARTICLE TEXT
# ============================================================

def clean_article_text(text):

    if not text:

        return None

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    boilerplate_patterns = [

        r"Add this site to your preferred sources.*$",

        r"You may also like.*$",

        r"Read more.*$",

        r"Subscribe to.*$",

        r"Follow us on.*$"

    ]

    for pattern in boilerplate_patterns:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

    return text


# ============================================================
# ARTICLE VALIDATION
# ============================================================

def validate_article(
    text,
    min_chars=500
):

    if not text:

        return False

    text = text.strip()

    if len(text) < min_chars:

        return False

    if len(text.split()) < 100:

        return False

    return True


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_article_text(url):

    # --------------------------------------------------------
    # Method 1: Trafilatura
    # --------------------------------------------------------

    try:

        downloaded = trafilatura.fetch_url(
            url
        )

        if downloaded:

            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False
            )

            text = clean_article_text(
                text
            )

            if validate_article(text):

                return text

    except Exception:

        pass


    # --------------------------------------------------------
    # Method 2: Newspaper4k
    # --------------------------------------------------------

    try:

        article = Article(url)

        article.download()

        article.parse()

        text = article.text

        text = clean_article_text(
            text
        )

        if validate_article(text):

            return text

    except Exception:

        pass


    return None


# ============================================================
# FETCH NEWS FROM NEWS API
# ============================================================

def fetch_news(
    query,
    page_size=5
):

    if not NEWS_API_KEY:

        raise ValueError(
            "NewsAPI key not found. "
            "Add NEWS_API_KEY to Streamlit secrets."
        )

    url = "https://newsapi.org/v2/everything"

    params = {

        "q": query,

        "language": "en",

        "sortBy": "publishedAt",

        "pageSize": page_size

    }

    headers = {

        "X-Api-Key": NEWS_API_KEY

    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":

        raise Exception(
            data.get(
                "message",
                "NewsAPI request failed"
            )
        )

    return data.get(
        "articles",
        []
    )


# ============================================================
# FETCH + EXTRACT + SUMMARIZE
# ============================================================

def fetch_and_summarize(
    query,
    page_size=5
):

    articles = fetch_news(
        query=query,
        page_size=page_size
    )

    results = []

    seen_urls = set()

    progress_bar = st.progress(
        0
    )

    total_articles = len(articles)

    for i, article in enumerate(
        articles
    ):

        title = article.get(
            "title",
            "Untitled Article"
        )

        url = article.get(
            "url"
        )

        if not url:

            continue

        if url in seen_urls:

            continue

        seen_urls.add(url)

        progress_bar.progress(
            (i + 1) / total_articles
        )

        text = extract_article_text(
            url
        )

        if not text:

            continue

        try:

            result = summarize_news(
                text
            )

            results.append({

                "title": title,

                "source": article.get(
                    "source",
                    {}
                ).get(
                    "name",
                    "Unknown Source"
                ),

                "publishedAt":
                    article.get(
                        "publishedAt",
                        ""
                    ),

                "url": url,

                "summary":
                    result["summary"],

                "method":
                    result["method"],

                "input_tokens":
                    result["input_tokens"]

            })

        except Exception:

            continue

    progress_bar.empty()

    return results


# ============================================================
# STREAMLIT FRONTEND
# ============================================================

st.markdown(
    '<div class="main-title">📰 Live News Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Search for a topic and get AI-generated summaries '
    'of the latest news articles.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    article_count = st.slider(
        "Number of articles",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    st.write(
        "### 🤖 Summarization"
    )

    st.write(
        "Model: BART-large-CNN"
    )

    st.write(
        "Long articles automatically use "
        "Hierarchical BART."
    )

    st.divider()

    st.caption(
        "News is retrieved using NewsAPI."
    )


# ============================================================
# SEARCH INPUT
# ============================================================

query = st.text_input(
    "🔎 What news are you interested in?",
    placeholder="e.g. artificial intelligence, Tesla, climate change..."
)


# ============================================================
# SEARCH BUTTON
# ============================================================

search_clicked = st.button(
    "🔍 Get Latest News",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS SEARCH
# ============================================================

if search_clicked:

    if not query.strip():

        st.warning(
            "Please enter a news topic."
        )

    else:

        with st.spinner(
            "Fetching and summarizing latest news..."
        ):

            try:

                results = fetch_and_summarize(
                    query=query.strip(),
                    page_size=article_count
                )

                st.session_state[
                    "news_results"
                ] = results

                st.session_state[
                    "search_query"
                ] = query.strip()

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "news_results" in st.session_state:

    results = st.session_state[
        "news_results"
    ]

    search_query = st.session_state.get(
        "search_query",
        ""
    )

    st.divider()

    if results:

        st.subheader(
            f"Latest news for: {search_query}"
        )

        st.caption(
            f"{len(results)} articles summarized"
        )

        for i, result in enumerate(
            results
        ):

            st.markdown(
                '<div class="article-card">',
                unsafe_allow_html=True
            )

            # Article title

            st.markdown(
                f'<div class="article-title">'
                f'{result["title"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Metadata

            published = result[
                "publishedAt"
            ]

            if published:

                published = published.replace(
                    "T",
                    " "
                ).replace(
                    "Z",
                    ""
                )

            st.markdown(
                f'<div class="article-meta">'
                f'📰 {result["source"]} '
                f'&nbsp; • &nbsp; '
                f'🕒 {published}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Summary

            st.markdown(
                '<div class="summary-title">'
                'AI Summary'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="summary-text">'
                f'{result["summary"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write("")

            # Technical information

            col1, col2 = st.columns(2)

            with col1:

                st.caption(
                    f"Model: {result['method']}"
                )

            with col2:

                st.caption(
                    f"Input tokens: "
                    f"{result['input_tokens']}"
                )

            # Original article

            st.link_button(
                "Read Original Article ↗",
                result["url"]
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "No articles could be processed "
            "for this topic. Try another search."
        )