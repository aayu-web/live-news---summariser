# 📰 Live News Summarizer

An AI-powered **Live News Summarization System** that retrieves the latest news based on a user's topic and generates concise summaries using **BART (`facebook/bart-large-cnn`)**.

The project combines **NLP, abstractive text summarization, live news retrieval, web article extraction, and Streamlit** into a complete end-to-end application.

---

## 🚀 Demo

> Add your deployed Streamlit URL here after deployment.

**Live Demo:** Coming Soon

**GitHub:** https://github.com/aayu-web/live-news---summariser

---

## 📌 Project Overview

Traditional text summarization datasets contain historical articles, which makes them unsuitable for answering questions about **current events**.

This project solves that problem by combining a pretrained abstractive summarization model with a live news retrieval pipeline.

### How it works

```text
User enters a topic
        ↓
NewsAPI retrieves latest articles
        ↓
Article URLs are extracted
        ↓
Full article content is downloaded
        ↓
Article text is cleaned and validated
        ↓
BART summarizes the article
        ↓
Long articles → Hierarchical BART
Short articles → Direct BART
        ↓
Summaries displayed in Streamlit
```

---

## ✨ Features

* 🔎 Search news using a custom topic
* 📰 Retrieve the latest articles using NewsAPI
* 🌐 Extract full article content from publisher websites
* 🧹 Clean and validate extracted article text
* 🤖 Abstractive summarization using BART
* 📚 Automatic handling of long articles using hierarchical summarization
* ⚡ Direct summarization for shorter articles
* 🔗 Links to the original news articles
* 📊 Displays source, publication time, summarization method, and input size
* 🖥️ Interactive Streamlit interface
* 🔐 API keys handled using Streamlit Secrets

---

## 🧠 Model

The project uses:

**Model:** `facebook/bart-large-cnn`

BART is a Transformer-based sequence-to-sequence model designed for text generation tasks such as abstractive summarization.

The model can accept up to approximately **1024 BART tokens** in a single input.

Because news articles can be much longer than this limit, the project uses two summarization strategies.

### 1. Direct BART

For articles containing **≤ 1024 tokens**, the article is passed directly to BART.

```text
Article
   ↓
BART
   ↓
Summary
```

### 2. Hierarchical BART

For articles containing **> 1024 tokens**, the article is divided into overlapping chunks.

Each chunk is summarized individually, and the resulting summaries are combined and passed through BART again to produce the final summary.

```text
Long Article
     ↓
Tokenization + Chunking
     ↓
┌──────────┐
│ Chunk 1  │ → Summary 1
├──────────┤
│ Chunk 2  │ → Summary 2
├──────────┤
│ Chunk 3  │ → Summary 3
└──────────┘
     ↓
Combined summaries
     ↓
Final BART summarization
     ↓
Final Summary
```

---

## 📊 Model Evaluation

The summarization pipeline was evaluated on a **100-article held-out subset** from the CNN/DailyMail test set.

ROUGE metrics were used for evaluation.

| Model             |    ROUGE-1 |    ROUGE-2 |    ROUGE-L | ROUGE-Lsum |
| ----------------- | ---------: | ---------: | ---------: | ---------: |
| TF-IDF            |     0.2352 |     0.0845 |     0.1651 |     0.1938 |
| Direct BART       |     0.3556 |     0.1590 |     0.2698 |     0.3088 |
| Hierarchical BART |     0.3409 |     0.1449 |     0.2613 |     0.2965 |
| Hybrid BART       | **0.3580** | **0.1606** | **0.2715** | **0.3114** |

The results show that the BART-based approaches substantially outperform the TF-IDF extractive baseline on this evaluation subset.

> ROUGE measures lexical overlap with reference summaries and should not be interpreted as a direct measurement of factual accuracy.

---

## 🛠️ Tech Stack

### Machine Learning / NLP

* Python
* PyTorch
* Hugging Face Transformers
* BART
* CNN/DailyMail Dataset
* ROUGE

### Live News Pipeline

* NewsAPI
* Trafilatura
* Newspaper4k
* Requests

### Frontend

* Streamlit

### Development

* Google Colab
* VS Code
* Git
* GitHub

---

## 📂 Project Structure

```text
live-news---summariser/
│
├── app.py
├── requirements.txt
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

`secrets.toml` is intentionally excluded from Git using `.gitignore`.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/aayu-web/live-news---summariser.git
cd live-news---summariser
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure NewsAPI

Create a NewsAPI key and configure Streamlit Secrets.

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
NEWS_API_KEY = "your_api_key_here"
```

Never commit this file to GitHub.

### 4. Run the application

```bash
streamlit run app.py
```

The application will open locally at:

```text
http://localhost:8501
```

---

## 🔑 API

This project uses **NewsAPI** to retrieve current news articles.

The application sends the user's query to the NewsAPI `/everything` endpoint and sorts retrieved articles by publication time.

The article's publisher URL is then used to extract the full article text before summarization.

---

## 🧹 Article Extraction

News APIs may provide only limited article content.

Therefore, the application retrieves the original article URL and attempts full-text extraction using:

1. **Trafilatura**
2. **Newspaper4k** as a fallback

Extracted text is then cleaned and validated before being passed to the summarization model.

This helps prevent incomplete or extremely short content from being summarized.

---

## 🔄 Summarization Pipeline

The complete backend pipeline is:

```text
Query
 ↓
NewsAPI
 ↓
Article Metadata
 ↓
Publisher URL
 ↓
Trafilatura
 ↓
Newspaper4k fallback
 ↓
Text Cleaning
 ↓
Article Validation
 ↓
Token Count
 ↓
 ┌───────────────────────┐
 │                       │
 ≤ 1024              > 1024
 │                       │
Direct BART        Hierarchical BART
 │                       │
 └───────────┬───────────┘
             ↓
          Summary
             ↓
       Streamlit UI
```

---

## 🎯 Why This Project?

The main goal was to move beyond a static NLP model and build a **complete AI application**.

Instead of simply training/evaluating a summarization model on an existing dataset, this project connects the model to real-world data:

```text
Dataset → Model Development
                ↓
         Validated Model
                ↓
        Live Data Pipeline
                ↓
         End-to-End AI App
```

This demonstrates the integration of:

* NLP
* Transformer models
* Text preprocessing
* API integration
* Web scraping/article extraction
* Long-document processing
* Backend logic
* Interactive UI
* Deployment

---

## ⚠️ Limitations

* NewsAPI results depend on the availability and relevance of indexed articles.
* Some publisher websites may block automated article extraction.
* Extracted article content can occasionally be incomplete.
* BART has a limited input context window, requiring chunking for long articles.
* Summaries are generated by a pretrained model and may occasionally contain factual errors.
* ROUGE evaluates similarity to reference summaries rather than factual correctness.
* Processing multiple live articles can take time because BART-large-CNN is computationally expensive.

---

## 🔮 Future Improvements

Potential improvements include:

* Better news relevance ranking
* Duplicate article detection
* Multi-source article comparison
* Factuality verification
* Multilingual summarization
* More efficient long-document summarization
* Article clustering by topic
* Sentiment analysis
* Automatic key-point extraction
* Summarizing multiple articles into a single news brief
* Deployment with GPU-backed inference

---

## 👨‍💻 Author

**Aayush Jha**

BTech Computer Science Engineering
AI/ML Enthusiast

### Areas of Interest

* Machine Learning
* Deep Learning
* Natural Language Processing
* Computer Vision
* Generative AI
* Data Science

---

## ⭐ If you find this project interesting

Feel free to explore the repository, try the application, or connect with me on LinkedIn.

