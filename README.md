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

For articles containing **> 1024 tokens**, the article is divided into
