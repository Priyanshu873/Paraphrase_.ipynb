# 🔍 Paraphrase Detection & Sentence Similarity using Sentence Transformers

## 📌 Project Overview

This project demonstrates how to use **Sentence Transformers** to perform:

* Paraphrase Detection
* Sentence Similarity Analysis
* Outlier Sentence Detection

It leverages pre-trained transformer models to convert sentences into embeddings and analyze semantic similarity.

---

## 🚀 Features

* ✅ Generate sentence embeddings using transformer models
* ✅ Identify paraphrases using similarity scores
* ✅ Compute cosine similarity between sentences
* ✅ Detect outlier sentences in a group

---

## 🧠 Models Used

* `all-MiniLM-L6-v2` → Fast and efficient for general embeddings
* `distiluse-base-multilingual-cased-v1` → Supports multilingual sentence embeddings

---

## ⚙️ Installation

Install required libraries:

```bash
pip install sentence-transformers
```

---

## 📊 How It Works

### 1. Sentence Embedding

Sentences are converted into numerical vectors using pre-trained models.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, convert_to_tensor=True)
```

---

### 2. Paraphrase Mining

Finds similar sentence pairs with similarity scores.

```python
from sentence_transformers.util import paraphrase_mining
paraphrases = paraphrase_mining(model, sentences)
```

---

### 3. Cosine Similarity

Measures similarity between sentence embeddings.

```python
from sentence_transformers import util
cosine_similarity = util.pytorch_cos_sim(embeddings, embeddings)
```

---

### 4. Outlier Detection

Identifies the least similar sentence in a group.

```python
avg_sim = cosine_similarity.mean(dim=1)
outlier_index = avg_sim.argmin().item()
```

---

## 📁 Dataset

Sample sentences are manually defined in the notebook for demonstration purposes.

Example:

* "The cat sits outside"
* "The cat plays in the garden"
* "I love pasta"
* "A man is playing guitar"

---

## 📈 Output

* Similar sentence pairs with scores
* Cosine similarity matrix
* Outlier sentence identification

---

## 🎯 Use Cases

* NLP Projects
* Chatbots & Virtual Assistants
* Semantic Search
* Duplicate Question Detection
* Content Recommendation Systems

---

## 🧑‍💻 Author

**Your Name**

* LinkedIn: https://www.linkedin.com/in/priyanshu-mishra-477338328/
* Email: priyanshumishra19086@gmail.com

---

## ⭐ Contribution

Feel free to fork this repository and contribute by improving models or adding new features.

---

## 📌 License

This project is open-source and available under the MIT License.
