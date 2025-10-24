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

### 2️⃣ Data Preprocessing

data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


### 3️⃣ Silhouette Optimization

eps_values = np.arange(0.1, 1.5, 0.1)
minpts_values = [3, 5, 7, 10]
scores = []

for eps in eps_values:
    for min_pts in minpts_values:
        model = DBSCAN(eps=eps, min_samples=int(min_pts))
        labels = model.fit_predict(X_scaled)

        if len(set(labels)) > 1 and -1 not in set(labels):
            score = silhouette_score(X_scaled, labels)
            scores.append((eps, min_pts, score))

### 4️⃣ Heatmap Visualization

score_df = pd.DataFrame(scores, columns=["Eps", "MinPts", "Silhouette"])
pivot = score_df.pivot("MinPts", "Eps", "Silhouette")

plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("Silhouette Score Heatmap (DBSCAN)")
plt.show()

### 5️⃣ Final Model

best_params = score_df.loc[score_df['Silhouette'].idxmax()]
best_eps = best_params['Eps']
best_minpts = int(best_params['MinPts'])

final_dbscan = DBSCAN(eps=best_eps, min_samples=best_minpts)
labels = final_dbscan.fit_predict(X_scaled)
data['Cluster'] = labels

### 6️⃣ Cluster Visualization

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X['Annual Income (k$)'], 
    y=X['Spending Score (1-100)'],
    hue=data['Cluster'],
    palette='Set1',
    s=100
)
plt.title(f"DBSCAN Clustering (eps={best_eps:.2f}, minPts={best_minpts})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.show()
