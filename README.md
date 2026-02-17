# NewsRecAtlas
A Structural and Modeling Study of Large-Scale News Recommendation Systems

---

## About This Project

**NewsRecAtlas** is a semester-long project developed as part of  
**[Course Name] – [Course Code]**  
at **[University Name]**.

The project investigates the structural, behavioral, and bias-related properties of large-scale recommendation systems using the MIND (Microsoft News Dataset). It progresses from rigorous exploratory data analysis to advanced modeling techniques, including sequential and LLM-based methods.

This repository documents the evolution of the project as the course progresses and serves as a professional portfolio artifact.

---

## Motivation

Modern recommender systems operate under:

- Extreme sparsity  
- Severe implicit feedback imbalance  
- Long-tail popularity distributions  
- Exposure and ranking bias  
- Temporal burstiness  

Understanding these structural constraints is essential before designing effective recommendation models.

NewsRecAtlas aims to map and analyze these structural dynamics before extending into advanced modeling approaches.

---

## Dataset

**MIND-small (Microsoft News Dataset)**

- 156,965 impressions  
- 50,000 users  
- 51,282 news articles  
- Implicit click feedback  

Files used:
- `behaviors.tsv` — User impression logs
- `news.tsv` — Article metadata (category, title, abstract)

MIND-small preserves the schema of MIND-large while remaining computationally feasible for iterative experimentation.

---

## Phase 1 — Exploratory Data Analysis (Completed)

### Structural Findings

- **Interaction sparsity:** 0.999908  
- **Co-occurrence sparsity:** 0.993806  
- **Click ratio:** 4.04%  
- **Cold-start users:** 32.77%  
- **Mean candidates per impression:** 37.23  

### Observed Patterns

- Strong long-tail popularity distribution  
- Significant exposure (position) bias  
- Temporal burstiness in user interactions  
- Positive correlation between activity and browsing history depth  

### Validation

Non-trivial structural validation tests confirm:

- Impression integrity  
- Referential consistency  
- Timestamp monotonicity  
- Expected sparsity levels  

The EDA establishes a robust structural foundation for modeling decisions.

---

## Research Direction

### Course-Aligned Techniques
- Frequent itemset and co-occurrence analysis  
- Graph-based user–item modeling  
- User clustering based on engagement patterns  

### Beyond-Course Extensions
- Sequential recommendation models  
- Popularity-aware and bias-aware ranking strategies  

### LLM-Based Extension
- Transformer-based embeddings of news titles/abstracts  
- Content-aware modeling for cold-start mitigation  

---

## Key Research Questions

1. How does extreme sparsity affect traditional collaborative filtering?
2. Do sequential models capture bursty behavior better than static models?
3. How does popularity bias influence ranking diversity?
4. Can semantic embeddings improve cold-start performance?

---

## Assumptions and Constraints

- Click labels approximate user preference (implicit feedback).
- Impression candidate sets reflect outputs of a pre-existing ranking system.
- MIND-small is structurally representative of MIND-large.

---

## Repository Structure

notebooks/
01_data_setup_and_eda.ipynb

README.md


Future additions:
- Graph modeling experiments  
- Sequential recommendation implementations  
- LLM-based embedding experiments  
- Comparative evaluation analysis  

---

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn / Matplotlib
- (Planned) PyTorch / Transformers

---

## Project Status

✔ Exploratory Data Analysis Completed  
⏳ Baseline Modeling – In Progress  
🔜 Sequential & LLM Extensions – Planned  

---

## Author

[Your Name]  
[Program Name], [University Name]  
[Course Name – Course Code]
