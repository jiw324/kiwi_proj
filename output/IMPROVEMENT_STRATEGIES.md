# Significant Performance Boost Strategies

## 🎯 Current Bottleneck Analysis

### **Major Issue Identified: High-Brix Samples Performance Collapse**

From `error_bins_cv.csv`:

| Target Range | ENSEMBLE RMSE | PLS RMSE | Status |
|--------------|---------------|----------|---------|
| < 12°Brix | 1.233 | 1.403 | ✅ Good |
| 12-14°Brix | 1.129 | 1.110 | ✅ Good |
| **> 14°Brix** | **2.704** | **2.225** | ❌❌❌ **TERRIBLE** |

**🚨 CRITICAL FINDING:** Performance **DOUBLES** for high-sugar samples (>14°Brix)!
- Only 36 samples in this range (8.6% of dataset)
- RMSE 2.7 vs 1.1 for mid-range (2.4× worse)
- This is the **PRIMARY target for improvement**

---

## 🚀 Top 5 Strategies for SIGNIFICANT Improvement (Ranked by Impact)

### **1. TARGETED DATA AUGMENTATION FOR HIGH-BRIX SAMPLES** ⭐⭐⭐⭐⭐
**Expected Gain:** 20-40% overall RMSE reduction

**Problem:**
- High-Brix samples severely underrepresented (36/420 = 8.6%)
- Models lack sufficient training data for this critical range
- Water band saturation effects at high sugar concentrations

**Solutions:**
```python
# A. Collect more high-Brix samples (BEST but expensive)
Target: 150+ samples in >14°Brix range (currently only 36)

# B. SMOTE/Synthetic Oversampling for spectral data
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy={'>14': 150}, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y_binned)

# C. Spectral Mixup Augmentation (wavelength interpolation)
def spectral_mixup(spec1, spec2, alpha=0.5):
    return alpha * spec1 + (1 - alpha) * spec2

# D. Add Gaussian noise to high-Brix samples during training
X_high_brix_aug = X_high_brix + np.random.normal(0, 0.01, X_high_brix.shape)
```

**Implementation Priority:** 🔥 **IMMEDIATE** - Highest ROI

---

### **2. STRATIFIED ENSEMBLE WITH TARGET-SPECIFIC SUB-MODELS** ⭐⭐⭐⭐⭐
**Expected Gain:** 15-30% RMSE reduction for >14°Brix

**Concept:** Train separate expert models for different target ranges

**Architecture:**
```python
class StratifiedEnsemble:
    def __init__(self):
        self.low_expert = PLS_model(optimized_for_range='<12')
        self.mid_expert = ENSEMBLE_model(optimized_for_range='12-14')
        self.high_expert = XGB_model(optimized_for_range='>14')  # More samples needed
        self.router = LogisticRegression()  # Predict which expert to use
    
    def fit(self, X, y):
        # Train router to predict target bin
        self.router.fit(X, bin_labels)
        # Train each expert on its specialty
        self.low_expert.fit(X[y < 12], y[y < 12])
        self.mid_expert.fit(X[(y >= 12) & (y <= 14)], y[(y >= 12) & (y <= 14)])
        self.high_expert.fit(X[y > 14], y[y > 14])
    
    def predict(self, X):
        bin_probs = self.router.predict_proba(X)
        predictions = (
            bin_probs[:, 0] * self.low_expert.predict(X) +
            bin_probs[:, 1] * self.mid_expert.predict(X) +
            bin_probs[:, 2] * self.high_expert.predict(X)
        )
        return predictions
```

**Key Benefit:** Each model specializes in its range, dramatically improving high-Brix predictions

---

### **3. ADVANCED PREPROCESSING: EMSC + EPO** ⭐⭐⭐⭐
**Expected Gain:** 10-20% RMSE reduction

**Problem:** Current preprocessing (SNV + Savitzky-Golay) doesn't handle:
- Temperature variations
- Light scattering effects (especially in high-sugar fruits)
- Instrument drift

**Solutions:**

**A. Extended Multiplicative Signal Correction (EMSC):**
```python
from scipy.linalg import lstsq

def emsc_correction(spectra, reference_spectrum):
    """Removes multiplicative and additive effects"""
    n_samples, n_wavelengths = spectra.shape
    corrected = np.zeros_like(spectra)
    
    # Design matrix: [1, wl, wl^2, reference_spectrum]
    wl = np.arange(n_wavelengths)
    X_emsc = np.column_stack([
        np.ones(n_wavelengths),
        wl,
        wl**2,
        reference_spectrum
    ])
    
    for i in range(n_samples):
        coefs, _, _, _ = lstsq(X_emsc, spectra[i])
        corrected[i] = (spectra[i] - coefs[0] - coefs[1]*wl - coefs[2]*wl**2) / coefs[3]
    
    return corrected
```

**B. External Parameter Orthogonalization (EPO):**
```python
def epo_correction(X_train, X_test, temperature_train, temperature_test):
    """Remove temperature effects from spectra"""
    from sklearn.cross_decomposition import PLSRegression
    
    # Model temperature effects
    pls_temp = PLSRegression(n_components=2)
    pls_temp.fit(X_train, temperature_train)
    
    # Project out temperature subspace
    temp_subspace = pls_temp.x_loadings_
    projection = np.eye(X_train.shape[1]) - temp_subspace @ temp_subspace.T
    
    X_train_corrected = X_train @ projection
    X_test_corrected = X_test @ projection
    
    return X_train_corrected, X_test_corrected
```

**Implementation:** Medium complexity, high reward for temperature-variable datasets

---

### **4. OPTIMIZED WAVELENGTH SELECTION** ⭐⭐⭐⭐
**Expected Gain:** 8-15% RMSE reduction

**Current:** Using full 920-1680nm window (potentially includes noisy/irrelevant regions)

**Strategies:**

**A. Genetic Algorithm for Feature Selection:**
```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from deap import base, creator, tools, algorithms

def ga_wavelength_selection(X, y, n_wavelengths=50):
    """Select optimal wavelength subset using GA"""
    
    def fitness_function(individual):
        selected_wl = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected_wl) < 10:  # Minimum wavelengths
            return (float('inf'),)
        
        X_selected = X[:, selected_wl]
        model = PLSRegression(n_components=10)
        scores = cross_val_score(model, X_selected, y, cv=5, 
                                scoring='neg_mean_squared_error')
        return (-np.mean(scores),)
    
    # Run GA (implementation details omitted for brevity)
    # Returns optimal wavelength indices
```

**B. Interval PLS (iPLS) - Systematic Region Selection:**
```python
def interval_pls(X, y, n_intervals=20):
    """Test different spectral intervals"""
    n_wl = X.shape[1]
    interval_size = n_wl // n_intervals
    best_rmse = float('inf')
    best_interval = None
    
    for i in range(n_intervals):
        start = i * interval_size
        end = (i + 1) * interval_size
        X_interval = X[:, start:end]
        
        model = PLSRegression(n_components=10)
        scores = cross_val_score(model, X_interval, y, cv=5,
                                scoring='neg_root_mean_squared_error')
        rmse = -np.mean(scores)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_interval = (start, end)
    
    return best_interval
```

**C. Variable Importance in Projection (VIP) Scores:**
```python
def vip_scores(pls_model):
    """Calculate VIP scores for wavelength selection"""
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    
    p, h = w.shape
    vips = np.zeros((p,))
    
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    
    return vips
```

**Target:** Select 50-100 most informative wavelengths (currently using ~300+)

---

### **5. STACKED GENERALIZATION (Meta-Learning)** ⭐⭐⭐⭐
**Expected Gain:** 10-18% RMSE reduction

**Current Limitation:** Simple inverse-RMSE weighting doesn't learn optimal combinations

**Upgrade to Stacked Ensemble:**
```python
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge

class StackedEnsemble:
    def __init__(self):
        # Level 0: Base models
        self.base_models = {
            'PLS': PLSRegression(n_components=10),
            'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
            'XGB': XGBRegressor(n_estimators=100),
            'MPHNN': MPHNNWrapper(...)
        }
        # Level 1: Meta-learner
        self.meta_model = Ridge(alpha=1.0)
    
    def fit(self, X, y):
        # Train base models and get out-of-fold predictions
        meta_features = np.zeros((len(y), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            model.fit(X, y)
            # Out-of-fold predictions to avoid overfitting
            meta_features[:, i] = cross_val_predict(model, X, y, cv=5)
        
        # Train meta-model on base predictions
        self.meta_model.fit(meta_features, y)
    
    def predict(self, X):
        # Get base model predictions
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models.values()
        ])
        # Meta-model makes final prediction
        return self.meta_model.predict(meta_features)
```

**Key Advantage:** Meta-learner optimally combines base predictions (vs simple averaging)

---

## 🔬 Advanced Techniques (Higher Risk, Higher Reward)

### **6. ATTENTION-BASED DEEP LEARNING** ⭐⭐⭐
**Expected Gain:** 15-25% (if successful)

```python
import torch
import torch.nn as nn

class SpectralTransformer(nn.Module):
    def __init__(self, n_wavelengths=300, d_model=128, nhead=8):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(n_wavelengths, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=4
        )
        self.fc = nn.Linear(d_model * n_wavelengths, 1)
    
    def forward(self, x):
        # x: (batch, n_wavelengths)
        x = x.unsqueeze(-1)  # (batch, n_wavelengths, 1)
        x = self.embedding(x)  # (batch, n_wavelengths, d_model)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = x.flatten(1)
        return self.fc(x)
```

**Risk:** Needs more training data, prone to overfitting with small datasets

---

### **7. SELF-SUPERVISED PRE-TRAINING** ⭐⭐⭐
**Expected Gain:** 10-20% (especially with limited labels)

```python
# Pre-train on unlabeled spectral data using contrastive learning
def contrastive_pretraining(spectra_unlabeled):
    """Learn spectral representations without labels"""
    from torch.nn import CosineSimilarity
    
    class ContrastiveEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(300, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        
        def forward(self, x):
            return self.encoder(x)
    
    # Create augmented pairs (noise, smoothing, etc.)
    # Maximize similarity between augmented versions
    # Then fine-tune on labeled data
```

**Benefit:** Can leverage large amounts of unlabeled NIR spectra

---

### **8. CALIBRATION TRANSFER / DOMAIN ADAPTATION** ⭐⭐⭐⭐
**Expected Gain:** 20-35% for LOBO specifically

**Current Issue:** LOBO performance degrades due to batch effects

**Solution: Direct Standardization (DS) or Piecewise Direct Standardization (PDS):**

```python
def piecewise_direct_standardization(X_master, X_slave, window_size=5):
    """Transfer calibration between batches"""
    n_wl = X_master.shape[1]
    F = np.zeros((n_wl, n_wl))  # Transfer matrix
    
    for i in range(n_wl):
        # Window around wavelength i
        start = max(0, i - window_size)
        end = min(n_wl, i + window_size + 1)
        
        X_window = X_slave[:, start:end]
        y_target = X_master[:, i]
        
        # Fit linear model
        coef = np.linalg.lstsq(X_window, y_target, rcond=None)[0]
        F[i, start:end] = coef
    
    return F

# Apply transfer
X_slave_corrected = X_slave @ F.T
```

---

## 📊 Implementation Roadmap (Priority Order)

### **Phase 1: Quick Wins (1-2 weeks)**
1. ✅ **High-Brix Data Augmentation** (SMOTE/Synthetic Oversampling)
2. ✅ **VIP-based Wavelength Selection** (reduce from 300 to 80 best wavelengths)
3. ✅ **Stacked Ensemble** (replace inverse-RMSE with meta-learner)

**Expected Combined Gain:** 30-40% RMSE reduction for >14°Brix, 15-20% overall

### **Phase 2: Medium-Term (2-4 weeks)**
4. ✅ **EMSC Preprocessing** (handle scattering/temperature better)
5. ✅ **Stratified Ensemble** (target-specific models)
6. ✅ **PDS Calibration Transfer** (improve LOBO performance)

**Expected Combined Gain:** Additional 10-15% RMSE reduction

### **Phase 3: Advanced (4-8 weeks)**
7. ✅ **Spectral Transformer** (attention-based deep learning)
8. ✅ **Self-Supervised Pre-training** (if more data available)

**Expected Combined Gain:** Additional 10-20% (high variance)

---

## 💰 Cost-Benefit Analysis

| Strategy | Implementation Effort | Expected Gain | ROI |
|----------|---------------------|---------------|-----|
| **High-Brix Augmentation** | Low (2 days) | 20-30% | ⭐⭐⭐⭐⭐ |
| **Stacked Ensemble** | Low (2 days) | 10-15% | ⭐⭐⭐⭐⭐ |
| **Wavelength Selection (VIP)** | Low (1 day) | 8-12% | ⭐⭐⭐⭐⭐ |
| **EMSC Preprocessing** | Medium (3 days) | 10-15% | ⭐⭐⭐⭐ |
| **Stratified Ensemble** | Medium (4 days) | 15-25% | ⭐⭐⭐⭐ |
| **PDS Calibration Transfer** | Medium (3 days) | 15-20% (LOBO) | ⭐⭐⭐⭐ |
| **Spectral Transformer** | High (2-3 weeks) | 15-25% | ⭐⭐⭐ |
| **Collect More High-Brix Data** | Very High (months + $$$) | 30-50% | ⭐⭐⭐ |

---

## 🎯 Recommended Action Plan

**IMMEDIATE (Week 1):**
```bash
# 1. Implement SMOTE for high-Brix samples
# 2. Add VIP-based wavelength selection
# 3. Replace simple ensemble with stacked meta-learner
```

**Expected Result:** RMSE 1.1-1.2 (current: 1.385), >14°Brix RMSE < 2.0 (current: 2.7)

**SHORT-TERM (Weeks 2-4):**
```bash
# 4. Implement EMSC preprocessing
# 5. Add PDS for calibration transfer
# 6. Create stratified sub-models
```

**Expected Result:** Overall RMSE < 1.0, >14°Brix RMSE < 1.5

---

## 📈 Realistic Targets

**Conservative (Phase 1 only):**
- CV RMSE: 1.385 → **1.15** (16% improvement)
- LOBO RMSE: 1.353 → **1.15** (15% improvement)
- >14°Brix RMSE: 2.704 → **1.8** (33% improvement)

**Optimistic (Phases 1+2):**
- CV RMSE: 1.385 → **0.95** (31% improvement)
- LOBO RMSE: 1.353 → **1.0** (26% improvement)
- >14°Brix RMSE: 2.704 → **1.3** (52% improvement)

**Aggressive (All phases + more data):**
- CV RMSE: 1.385 → **0.75** (46% improvement)
- LOBO RMSE: 1.353 → **0.85** (37% improvement)
- >14°Brix RMSE: 2.704 → **1.0** (63% improvement)

---

## 🚦 Bottom Line

**TOP 3 PRIORITIES FOR MAXIMUM IMPACT:**

1. **🔥 Fix High-Brix Performance** (SMOTE augmentation + stratified models)
2. **🔥 Upgrade to Stacked Ensemble** (meta-learner instead of simple averaging)
3. **🔥 Intelligent Wavelength Selection** (VIP scores or GA)

These three alone could yield **30-40% improvement** with minimal effort!

