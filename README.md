# NewsRecAtlas
A Structural and Modeling Study of Large-Scale News Recommendation Systems using MIND Dataset

---

## About This Project

**NewsRecAtlas** is a semester-long project developed as part of  
**CSCE-676 – Data Mining & Analysis**  
at **Texas A&M University - College Station**.

The project investigates the structural, behavioral, and bias-related properties of large-scale recommendation systems using the MIND (Microsoft News Dataset). It progresses from rigorous exploratory data analysis to advanced modeling techniques.

This repository documents the evolution of the project as the course progresses and covers techniques specified through and beyond the course.

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

## Setup and Reproducibility

### Clone the Repository

```bash
git clone https://github.com/your-username/news-rec-atlas.git  
cd news-rec-atlas
```
---

### Environment Requirements

Developed using:

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- Jupyter Notebook

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```
---

### Dataset Download

Download MIND-small from:

https://msnews.github.io/

After downloading:

1. Extract the folder `MINDsmall_train`
2. Place it in the root directory

Expected structure:

```bash
news-rec-atlas/  
│  
├── MINDsmall_train/  
│   ├── behaviors.tsv  
│   └── news.tsv  
│  
├── notebooks/  
│   └── 01_data_selection_setup_and_eda.ipynb  
│  
└── README.md  
```
The notebook assumes:

```bash
MINDsmall_train/behaviors.tsv  
MINDsmall_train/news.tsv  
```

---

### Running the Notebook

Open:

```bash
notebooks/01_data_selection_setup_and_eda.ipynb
```

Run all cells from top to bottom.

---

## Phase 1 — Exploratory Data Analysis (Completed)

Phase 1 examined:

- Interaction sparsity and co-occurrence sparsity  
- Click imbalance and long-tail popularity  
- Cold-start prevalence  
- Exposure bias  
- Temporal interaction patterns  
- Correlation between user behavioral features  

Validation tests were implemented to ensure structural integrity and parsing correctness.

The EDA revealed extreme sparsity, strong imbalance, and bursty engagement behavior — all of which influence modeling choices.

---
## Future Work & Research Direction

### Course-Aligned Techniques
- Frequent itemset and co-occurrence analysis  
- Graph-based user–item modeling  
- User clustering based on engagement patterns  

### Beyond-Course Extensions
- Sequential recommendation models  
- Popularity-aware and bias-aware ranking strategies  
- LLM Transformer-based embeddings of news titles/abstracts  
- Content-aware modeling for cold-start mitigation  

Further Research questions present in `01_data_selection_setup_and_eda.ipynb`

---

## Repository Structure

```bash
news-rec-atlas/  
├── notebooks/  
│   └── 01_data_selection_setup_and_eda.ipynb  
│  
├── README.md  
│ 
└── .gitignore  
```


Potential Future additions:
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

---
## Assumptions and Constraints

- Click labels approximate user preference (implicit feedback).
- Impression candidate sets reflect outputs of a pre-existing ranking system.
- MIND-small is structurally representative of MIND-large.

---

## Project Status

- Exploratory Data Analysis - Completed  
- Baseline Modeling – Future Work
- Data Mining Techniques implementation (Course) - Future Work
- Data Mining Techniques implementation (External) - Future Work
- Research Potential - Future Work

---

## Author


Akash Moses Guttedar<br>
UIN - 535005841<br>
amg_1597@tamu.edu<br>
Dept. of Electrical & Computer Engineering<br>
Texas A&M University - College Station 
