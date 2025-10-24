# DBSCAN-Clustering-on-Mall-Customer-Dateset
"ML models implemented from scratch using NumPy and Pandas only"


# 🧠 DBSCAN Clustering (Scratch Implementation)

### 📋 Project Summary
- Implemented **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** on the **Mall Customers dataset**.  
- Automatically finds clusters without needing `k` and handles noise points (`-1`) efficiently.  

---

### ⚙️ Steps Overview
1. **📚 Import Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`  
2. **🧼 Preprocess Data:** Scale features `Annual Income` & `Spending Score`  
3. **📈 Silhouette Optimization:**  
   - Loop over different `eps` (0.1–1.5) and `min_samples` (3–10)  
   - Compute `silhouette_score` for each pair  
4. **🌡️ Heatmap Visualization:**  
   - Visualize silhouette scores for `(eps, minPts)` combinations  
5. **🏆 Final Model:**  
   - Select parameters with highest silhouette score  
   - Train DBSCAN → assign cluster labels  
6. **🎨 Visualization:**  
   - Plot clusters and outliers in 2D feature space  

---

### 🧮 Key Mathematical Concepts
- **ε-Neighborhood:** $$\( N_\varepsilon(p) = \{ q \ | \ dist(p, q) \leq \varepsilon \} \)$$  
- **Core Point:** $$\( |N_\varepsilon(p)| \geq \text{MinPts} \)$$  
- **Silhouette Score:** $$\( s = \frac{b - a}{\max(a, b)} \)$$
  (where $$\( a \)$$ = intra-cluster distance, $$\( b \)$$ = nearest-cluster distance)

---

### 📊 Results
- DBSCAN identified natural clusters without predefining `k`.  
- Outliers labeled as **`-1`**.  
- Silhouette heatmap revealed optimal parameters for dense cluster formation.  

---

### 🚀 Next Steps
- Extend to multi-dimensional data.  
- Compare with **Hierarchical** and **K-Means** clustering.  
- Explore **cosine** or **manhattan** distance metrics.  

---

👨‍💻 **Author:** *Anshu Pandey*  
📘 *From Scratch Implementation with Optimization & Visual Insights*
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
