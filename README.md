# DBSCAN-Clustering-on-Mall-Customer-Dateset
"ML models implemented from scratch using NumPy and Pandas only"

# 🧠 DBSCAN Clustering from Scratch with Silhouette Optimization  

## 📘 Overview
This project demonstrates **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** on the **Mall Customers dataset**.  
Unlike K-Means, DBSCAN does not require the number of clusters `k` in advance — instead, it discovers clusters based on **density regions** and handles **noise** effectively.  

---

## 🧮 Mathematical Foundations

| Concept          | Formula / Description |
|----------|------------------------|
| **ε-Neighborhood** | $$\( N_\varepsilon(p) = \{ q \in D \ | \ dist(p, q) \leq \varepsilon \} \) $$|
| **Core Point** | A point $$\( p \)$$ is a core point if $$\( |N_\varepsilon(p)| \geq \text{MinPts} \)$$ |
| **Directly Density Reachable** | A point $$\( q \)$$ is directly density reachable from $$\( p \)$$ if $$\( q \in N_\varepsilon(p) \)$$ and $$\( p \)$$ is a core point |
| **Density Connected** | Points $$\( p \)$$ and $$\( q \)$$ are density-connected if there exists a chain of core points connecting them |
| **Silhouette Score** | $$\( s = \frac{b - a}{\max(a, b)} \)$$ <br> where: $$<br> • \( a \)$$ = average intra-cluster distance $$<br> • \( b \)$$ = average nearest-cluster distance |

---

## 📊 Dataset Information
**Dataset:** Mall_Customers.csv  
**Features used:**  
- `Annual Income (k$)`  
- `Spending Score (1–100)`  

These features help identify spending patterns and income-based customer clusters.

---

## ⚙️ Workflow

### 1️⃣ Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
