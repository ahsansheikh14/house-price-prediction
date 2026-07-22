# House Price Predictor — Linear Regression from Scratch

A from-scratch implementation of linear regression (no scikit-learn for training) built while learning Week 1–2 of Andrew Ng's Machine Learning Specialization. Uses the California Housing dataset.

## What this project covers

- Single-feature linear regression (`MedInc → MedHouseVal`)
- Multi-feature linear regression using vectorized NumPy operations (`np.dot`)
- Feature scaling (z-score normalization) to fix a real gradient descent overflow bug
- Learning curve visualization to confirm convergence
- Polynomial feature engineering (`MedInc²`) as an experiment
- Validation against scikit-learn's `LinearRegression`
- A working prediction function on new, unseen input

## Project structure

```
house-price-prediction/
├── linear_regression.ipynb   # full notebook, all steps in order
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Key results

| Model | Final Cost (MSE/2) |
|---|---|
| Single feature (MedInc only) | 0.3508 |
| Multi-feature (4 features, scaled) | converged cleanly, matched sklearn |
| Polynomial (MedInc + MedInc²) | 0.3545 |

**Weights comparison — from-scratch vs sklearn (4-feature model):**

| Feature | My model | sklearn |
|---|---|---|
| MedInc | 0.8404 | 0.8409 |
| HouseAge | 0.2205 | 0.2206 |
| AveRooms | -0.0633 | -0.0637 |
| Population | 0.0282 | 0.0282 |
| Intercept (b) | 2.0685 | 2.0686 |

The close match confirms the from-scratch gradient descent implementation is mathematically correct.

## What I learned / what confused me

**The bug that taught me the most:** training the 4-feature model on raw (unscaled) data caused gradient descent to overflow into `nan` within a few iterations. I diagnosed it by comparing feature ranges — `Population` (up to ~35,000) vs `MedInc` (up to ~15) — and confirmed that the derivative term for `Population` was dominating the update step, causing it to overshoot the minimum repeatedly. Applying z-score normalization (`(x - mean) / std`) fixed this by putting every feature on a comparable scale.

**An honest, non-forced result:** I hypothesized that house prices might plateau at high incomes (a curved relationship), so I engineered `MedInc²` as an additional feature. The final cost barely changed (0.3508 → 0.3545, slightly worse), suggesting the `MedInc → price` relationship in this dataset is already close to linear, or that more iterations may be needed for the polynomial model to fully converge. I'm reporting this as-is rather than picking a different feature to force a "successful" result.

**Core concepts solidified through this project:**
- The distinction between `f(x)` (a single prediction) and `J(w,b)` (an aggregate error score used to compare different parameter choices)
- Why gradient descent subtracts `α * slope` — the sign of the derivative determines direction, `α` only determines step size
- Why vectorization (`np.dot`) is both faster and cleaner than manual loops
- Why feature scaling isn't optional once features have very different ranges

## Next steps

- Add regularization once covered in Week 3
- Extend to logistic regression for a classification variant
- Try feature engineering with other combinations (e.g. rooms per household)
