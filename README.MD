# 📝 Automatic Answer Assessment and Feedback Generation using LLMs

An intelligent Automated Essay Assessment (AES) system that evaluates descriptive answers, predicts scores, and generates structured feedback using Natural Language Processing (NLP), Machine Learning, and Large Language Models (LLMs).

---

## 📌 Overview

Manual evaluation of descriptive answers is time-consuming, subjective, and difficult to scale in educational environments. Traditional Automated Essay Scoring (AES) systems primarily focus on assigning scores and often lack transparency and meaningful feedback.

This project proposes an **LLM-driven Automatic Answer Assessment System** that not only predicts scores but also provides **rubric-based explanations and improvement suggestions** to support student learning.

The system combines semantic understanding, syntactic analysis, machine learning-based scoring, and LLM-generated feedback within a unified framework.

---

## ✨ Features

* ✅ Automated evaluation of descriptive answers
* ✅ Semantic similarity analysis using transformer embeddings
* ✅ Syntactic feature extraction
* ✅ Readability assessment
* ✅ Machine learning-based score prediction
* ✅ Rubric-wise scoring breakdown
* ✅ Explainable feedback generation
* ✅ Improvement suggestions for students
* ✅ Interactive web interface using Streamlit

---

## 🏗️ System Architecture

The system follows a modular pipeline architecture:

```
Student Answer
      │
      ▼
Preprocessing Module
      │
      ▼
Syntactic Analysis Module
      │
      ▼
Semantic Analysis Module
      │
      ▼
Scoring Module
      │
      ▼
Explanation Module
      │
      ▼
Final Score + Feedback
```

---

## ⚙️ Modules

### 1. Preprocessing Module

Performs text normalization and prepares the input for analysis.

#### Functions:

* Input validation
* Lowercasing
* Whitespace cleaning
* Sentence segmentation
* Tokenization
* Lemmatization

---

### 2. Syntactic Analysis Module

Extracts structural characteristics of the answer.

#### Features Extracted:

* Sentence count
* Word count
* Average sentence length

These features help assess organization and writing quality.

---

### 3. Semantic Analysis Module

Evaluates the meaning and relevance of the student's answer.

#### Techniques Used:

* MiniLM sentence embeddings
* Contextual encoding
* Cosine similarity

The student's answer is compared with the reference answer to determine semantic relevance.

---

### 4. Scoring Module

Predicts the final score based on extracted features.

#### Algorithm:

**Ridge Regression**

#### Input Features:

* Semantic similarity score
* Syntactic features
* Readability metrics

#### Output:

* Final score (0–10)
* Rubric-wise score allocation

---

### 5. Explanation Module

Generates understandable feedback for students.

#### Model Used:

**Phi-2**

#### Generates:

* Criterion-wise explanations
* Strengths and weaknesses
* Suggestions for improvement

---

## 🧠 Technologies Used

### Programming Language

* Python

### Machine Learning & Deep Learning

* PyTorch
* Scikit-learn

### NLP Libraries

* NLTK
* spaCy
* Sentence Transformers
* Hugging Face Transformers

### Data Processing

* Pandas
* NumPy

### Readability Analysis

* Textstat

### Model Utilities

* Accelerate
* Joblib
* tqdm

### Frontend

* Streamlit

---

## 🔬 Algorithms and Models

| Module              | Technique / Algorithm                 |
| ------------------- | ------------------------------------- |
| Preprocessing       | Tokenization, Lemmatization           |
| Syntactic Analysis  | Rule-Based Structural Quantification  |
| Semantic Analysis   | MiniLM Embeddings + Cosine Similarity |
| Scoring             | Ridge Regression                      |
| Feedback Generation | Phi-2 Transformer Model               |

---

## 📊 Results

The system successfully evaluates descriptive answers using a combination of semantic and structural analysis.

![alt text](<Irrelevant Answer.png>)

![alt text](<Partially Relevant Answer.png>)

![alt text](<Relevant Answer.png>)

### Achievements

* Achieved approximately **86% scoring accuracy**
* Generates scores in less than **2 seconds**
* Produces rubric-based explanations
* Provides meaningful improvement suggestions

### Output Includes

* Final score (out of 10)
* Content score
* Organization score
* Language score
* Structured feedback

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 💻 Usage

1. Launch the Streamlit application.
2. Enter the student's answer.
3. Provide the corresponding reference answer.
4. Submit the response.
5. View:

   * Predicted score
   * Rubric-wise evaluation
   * Generated explanations
   * Improvement suggestions

---

## 📌 Future Enhancements

* Multilingual answer assessment
* Fine-tuned transformer models for scoring
* Enhanced adversarial answer detection
* Deployment using cloud platforms
* LMS integration for educational institutions

---

## 🎓 Academic Information

**Project Title:**
*An LLM-Driven System for Automatic Essay Assessment and Feedback Generation*

**Department:** Department of Computer Applications

**Institution:** B.S. Abdur Rahman Crescent Institute of Science and Technology

**Academic Year:** 2024–2026

---

## 📜 License

This project is developed for academic and research purposes.

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository and submit a pull request.

---

### ⭐ If you found this project useful, consider giving this repository a star.
