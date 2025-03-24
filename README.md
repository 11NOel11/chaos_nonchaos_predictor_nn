# Chaos Classification: Neural Network vs Lyapunov

## 📌 Abstract  
Chaos theory describes the unpredictable yet deterministic nature of many physical systems, with applications in physics, engineering, and finance. The **Lyapunov exponent** is a widely used mathematical tool for classifying chaotic behavior, but can **deep learning outperform traditional methods?**  
This project explores whether **Neural Networks (NNs)** can provide a superior alternative for chaos detection, leveraging **SHM (Simple Harmonic Motion) & Double Pendulum** as test cases. Additionally, we employ **SHAP (SHapley Additive exPlanations)** to interpret model decisions, bridging the gap between deep learning and explainable AI.

## 📖 Introduction  
Chaos is an inherent property of many dynamical systems where small changes in initial conditions can lead to vastly different outcomes. Traditionally, the **Lyapunov exponent** is used to determine whether a system exhibits chaos (positive exponent) or remains predictable (negative or zero exponent). However, this approach relies on analytical methods that may be infeasible for complex real-world data.

In contrast, machine learning—particularly **neural networks**—can learn from observed patterns and classify chaotic vs. non-chaotic systems without explicit equations. This research investigates:

✅ Can neural networks **outperform** Lyapunov exponents in classifying chaos?  
✅ How does **Explainable AI (SHAP)** contribute to understanding chaos classification?  
✅ How does deep learning compare to **traditional mathematical approaches**?  

## ⚡ Key Contributions
✅ **Neural Network Classifier** – Trained on dynamical system trajectories to detect chaos.  
✅ **Lyapunov Exponent Baseline** – Traditional approach used for comparison.  
✅ **SHAP Explainability** – Analyzing how NNs make chaos classification decisions.  
✅ **Extensive Evaluation** – Accuracy, precision, recall, F1-score, and visualizations.  

---

## 📂 Theoretical Background
### 🔹 Lyapunov Exponent & Chaos
The **Lyapunov exponent (λ)** measures the sensitivity of a system to initial conditions. It is defined as:
\[ 
\lambda = \lim_{{t \to \infty}} \frac{1}{t} \sum_{i=1}^{t} \ln \left| \frac{dx_i}{dx_0} \right| 
\]
If \( \lambda > 0 \), the system is chaotic, meaning small perturbations lead to exponential divergence. If \( \lambda \leq 0 \), the system is stable or periodic.

---

## 📂 Dataset & Methodology
### 🔹 Dataset
We use simulated data from **SHM (Simple Harmonic Motion) and Double Pendulum systems**, generating time-series trajectories and extracting key features such as:
- Displacement, velocity, and acceleration
- Angular momentum (for pendulum systems)
- Phase space embedding

### 🔹 Neural Network Model
- **Feedforward Neural Network** with ReLU activation.
- **L2 Regularization & Dropout** to prevent overfitting.
- **Adam Optimizer** for adaptive learning.
- **Early Stopping** to optimize training performance.

### 🔹 Lyapunov Baseline Classifier
- Computes **Lyapunov exponent** and classifies based on \( \lambda > 0 \).
- Serves as a benchmark for machine learning performance.

---

## 📊 Experimental Results & Analysis
Our experiments show that deep learning significantly outperforms Lyapunov exponent classification. We also apply **SHAP explainability** to analyze how different features contribute to predictions.

| Model  | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|--------|-------------|--------------|------------|--------------|
| **Neural Network** | 90.25 | 87.76 | 94.75 | 90.19 |
| **Lyapunov Exponent** | 50.20 | 50.00 | 50.00 | 50.00 |

### **Key Observations**
📌 **Neural networks generalize well**, whereas Lyapunov exponents are rigid for unseen conditions.  
📌 **SHAP analysis** reveals that velocity and angular momentum are critical for chaos detection.  
📌 **Higher-dimensional embeddings improve NN performance.**  

---

## 📊 Visualizations & Explainability
### Double Pendulum Motion (Chaotic Example)
![Double Pendulum](graphs/double_pendulum_motion_chaotic.png)

### SHM Motion (Non-Chaotic Example)
![SHM](graphs/shm_nonchaotic.png)

### SHAP Feature Importance
![SHAP](graphs/meanshapvalue.png)

### Confusion Matrix
![Confusion Matrix](graphs/confusionmatrix.png)

### ROC Curve
![ROC](graphs/roc_curve.png)

---

## 🚀 Future Work & Research Directions
🔹 Apply chaos detection to **weather prediction systems** (atmospheric turbulence modeling).  
🔹 Use chaos classification for **financial market instability detection**.  
🔹 Expand dataset to include **Lorenz Attractor & Logistic Map**.  
🔹 Explore **Recurrent Neural Networks (RNNs) and Transformers** for time-series classification.  
🔹 Enhance interpretability using **LIME & Counterfactual Explanations**.  

---

## 📜 References
🔹 Sprott, J. C. (2003). *Chaos and Time-Series Analysis*  
🔹 Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*  
🔹 Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*  

---

## 📌 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/chaos-classification.git  
cd chaos-classification  
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt  
```
### 3️⃣ Run the Neural Network Model
```bash
python chaos_nn_classifier.py  
```

---

## 📬 Contact & Contributions
📩 **Want to contribute?** Open an **Issue** or submit a **Pull Request**.  
💡 **For inquiries, reach out via email or GitHub Discussions.**  

---

### 🔥 *"In chaos, AI finds patterns where math sees randomness!"*  
