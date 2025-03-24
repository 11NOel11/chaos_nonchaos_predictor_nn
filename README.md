# Chaos Classification: Neural Network vs Lyapunov

## Abstract  
Chaos theory describes the unpredictable yet deterministic nature of many physical systems, with applications in physics, engineering, and finance. The **Lyapunov exponent** is a widely used mathematical tool for classifying chaotic behavior, but can **deep learning outperform traditional methods?**  
This project evaluates chaos classification using controlled physical simulations of **Simple Harmonic Motion (SHM) and the Double Pendulum** as test cases. Additionally, we employ **SHAP (SHapley Additive exPlanations)** to interpret model decisions, bridging the gap between deep learning and explainable AI. This approach has implications for fields such as **astrophysics, climate modeling, and financial systems**.

## Introduction  
Chaos is an inherent property of many dynamical systems where small changes in initial conditions can lead to vastly different outcomes. Traditionally, the **Lyapunov exponent** is used to determine whether a system exhibits chaos (positive exponent) or remains predictable (negative or zero exponent). However, this approach relies on analytical methods that may be infeasible for complex real-world data.

In contrast, machine learning—particularly **neural networks**—can learn from observed patterns and classify chaotic vs. non-chaotic systems without explicit equations. This research investigates:

- **Can neural networks outperform Lyapunov exponents in classifying chaos?**  
- **How does Explainable AI (SHAP) contribute to understanding chaos classification?**  
- **How does deep learning compare to traditional mathematical approaches?**  

## Key Contributions  
- **Neural Network Classifier** – Trained on dynamical system trajectories to detect chaos.  
- **Lyapunov Exponent Baseline** – Traditional approach used for comparison.  
- **SHAP Explainability** – Analyzing how NNs make chaos classification decisions.  
- **Extensive Evaluation** – Accuracy, precision, recall, F1-score, and visualizations.  

---

## Theoretical Background  
### Lyapunov Exponent & Chaos  
The **Lyapunov exponent (\( \lambda \))** measures the sensitivity of a system to initial conditions. It is defined as:
\[  
\lambda = \lim_{{t \to \infty}} \frac{1}{t} \sum_{i=1}^{t} \ln \left| \frac{dx_i}{dx_0} \right|  
\]
If \( \lambda > 0 \), the system is chaotic, meaning small perturbations lead to exponential divergence. If \( \lambda \leq 0 \), the system is stable or periodic.

### Why Lyapunov Exponents Struggle in Some Cases  
While the Lyapunov exponent is a well-established tool, it has **limitations**:
- **High-Dimensional Systems** – Computing \( \lambda \) accurately becomes difficult in complex systems.
- **Noise Sensitivity** – Small measurement errors can lead to incorrect classification.
- **Unseen Scenarios** – The method struggles when applied to real-world data with uncertainties.

This motivates the need for a **data-driven alternative** like neural networks.

---

## Dataset & Methodology  
### Dataset  
We use simulated data from **SHM and Double Pendulum systems**, generating time-series trajectories and extracting key features such as:
- Displacement, velocity, and acceleration  
- Angular momentum (for pendulum systems)  
- Phase space embedding  

### Neural Network Model  
- **Feedforward Neural Network** with ReLU activation.  
- **L2 Regularization & Dropout** to prevent overfitting.  
- **Adam Optimizer** for adaptive learning.  
- **Early Stopping** to optimize training performance.  
- **Training Details** – The model was trained with an 80-20 train-test split, using a batch size of 32 and trained for 50 epochs.

### Lyapunov Baseline Classifier  
- Computes **Lyapunov exponent** and classifies based on \( \lambda > 0 \).  
- Serves as a benchmark for machine learning performance.  

---

## Experimental Results & Analysis  
Our experiments show that deep learning significantly outperforms Lyapunov exponent classification. We also apply **SHAP explainability** to analyze how different features contribute to predictions.

| Model  | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|--------|-------------|--------------|------------|--------------|
| **Neural Network** | 90.25 | 87.76 | 94.75 | 90.19 |
| **Lyapunov Exponent** | 50.20 | 50.00 | 50.00 | 50.00 |

### Key Observations  
- **Neural networks generalize well**, whereas Lyapunov exponents are rigid for unseen conditions.  
- **SHAP analysis** reveals that **velocity contributes more significantly to chaotic behavior detection than displacement in SHM.**  
- **Higher-dimensional embeddings improve NN performance.**  
- **SHAP Feature Analysis** – SHAP assigns feature importance scores, indicating that energy fluctuations and angular velocity have the highest predictive power.  
- **Computational Performance** – Training the NN took ~15 minutes on a standard CPU setup, whereas Lyapunov calculations were slower for large datasets.  

---

## Conclusion  
This study demonstrates that **deep learning provides a superior alternative to Lyapunov exponent-based chaos classification**. The neural network approach not only achieves higher accuracy but also offers better generalizability, especially in complex, high-dimensional systems where traditional methods fail. Furthermore, **SHAP-based explanations enhance interpretability**, making AI-driven classification more transparent.

The findings suggest that **deep learning can serve as a powerful tool for chaos detection in real-world applications** such as weather forecasting, financial risk assessment, and astrophysics. By combining neural networks with explainability techniques, we bridge the gap between AI and classical chaos theory, paving the way for more reliable and interpretable models in nonlinear dynamics.

### *"Deep learning offers a paradigm shift in chaos classification, enabling insights beyond conventional mathematical methods."*  

---

## Future Work & Research Directions  
- **Apply chaos detection to weather prediction systems** (e.g., atmospheric turbulence modeling).  
- **Use chaos classification for financial market instability detection, leveraging similar feature engineering approaches.**  
- **Expand dataset to include Lorenz Attractor & Logistic Map.**  
- **Explore Recurrent Neural Networks (RNNs) and Transformers for time-series classification.**  
- **Enhance interpretability using LIME & Counterfactual Explanations.**  
- **Augment the dataset with real-world sensor data rather than simulated trajectories, improving generalizability.**  

---

## References  
- Sprott, J. C. (2003). *Chaos and Time-Series Analysis*  
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*  
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*  

---
