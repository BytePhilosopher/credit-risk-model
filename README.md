# Credit Risk Probability Model — Xente

An end-to-end credit risk model built on transaction-level data from the Xente
eCommerce platform. The project derives a proxy credit-risk target from
unlabeled behavioral data, trains and tracks interpretable and high-performance
models, and serves risk predictions through a containerized REST API.

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/                          # gitignored
│   ├── raw/                       # Raw data (Xente challenge dataset)
│   └── processed/                 # Processed data for training
├── notebooks/
│   └── eda.ipynb                  # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering
│   ├── train.py                   # Model training
│   ├── predict.py                 # Inference
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord ties the amount of regulatory capital a bank must
hold directly to the measured riskiness of its assets. Under its Internal
Ratings-Based (IRB) approach, banks are permitted to use their own models to
estimate the core risk parameters — **Probability of Default (PD)**, **Loss
Given Default (LGD)**, and **Exposure at Default (EAD)** — which then feed
directly into the capital calculation. Because these estimates determine how
much capital is set aside, regulators require that the models be **transparent,
auditable, and well-documented**.

This has three practical consequences for how we build the model:

- **Interpretability is a regulatory requirement, not a nice-to-have.**
  Supervisors and internal validators must be able to understand *why* a borrower
  receives a given score. A model whose logic cannot be explained cannot be
  approved, regardless of its accuracy.
- **Documentation is part of the deliverable.** Every modeling choice — the
  definition of the target, feature transformations, data exclusions, and
  performance monitoring — must be recorded so the model can be independently
  reviewed and reproduced.
- **Errors are expensive in both directions.** Underestimating risk leaves the
  bank under-capitalized and exposed; overestimating it locks up capital and
  forgoes profitable lending. A defensible, well-understood model is the only way
  to justify the trade-off to regulators and the business.

In short, Basel II pushes us toward models we can *explain and defend*, which is
why interpretable techniques (e.g., Logistic Regression with Weight of Evidence)
remain the backbone of regulated credit scoring.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

The Xente dataset records transactions and a narrow `FraudResult` flag, but it
contains **no "loan default" label** — there is no record of customers who
borrowed and failed to repay. Supervised learning, however, requires a target to
learn from. We therefore must **engineer a proxy target**: an observable,
behavior-based signal that stands in for credit risk. A common approach is to use
**RFM (Recency, Frequency, Monetary) analysis** to segment customers and label
disengaged, low-value customers as "high risk" (proxy for likely-to-default) and
engaged, high-value customers as "low risk."

This is necessary to make the problem learnable, but it introduces real
**business risks**:

- **Label–reality mismatch.** The proxy measures *disengagement*, not actual
  *default*. A customer can be inactive yet perfectly creditworthy (or active yet
  a bad credit risk). Optimizing for the proxy can systematically mislabel good
  customers as bad, and vice versa.
- **Financial loss from both error types.** If the proxy mislabels good borrowers
  as high-risk, the business **rejects profitable customers and loses revenue**.
  If it mislabels bad borrowers as low-risk, the business **issues credit that
  defaults and incurs losses**.
- **Bias and fairness exposure.** A proxy built on behavioral patterns can encode
  hidden biases (e.g., against newer or lower-spending demographics), creating
  fair-lending and reputational risk.
- **Regulatory and validation risk.** The proxy definition is a major assumption
  that supervisors will scrutinize. If it cannot be justified against business
  reality, the whole model is undermined.

For these reasons, the proxy must be **clearly documented, justified to
stakeholders, and revisited** as soon as real repayment outcomes become
available.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) and a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Dimension | Logistic Regression + WoE (simple, interpretable) | Gradient Boosting (complex, high-performance) |
|---|---|---|
| **Interpretability** | High — each feature has a clear, monotonic, explainable contribution to the score. | Low — an ensemble of many trees; explanations require post-hoc tools (SHAP) that are approximations. |
| **Regulatory acceptance** | Strong — the standard for scorecards; easy to validate, document, and defend to supervisors. | Difficult — "black-box" models face heavier scrutiny and may require extra justification to approve. |
| **Predictive performance** | Good, but can underfit complex non-linear interactions. | Typically higher accuracy/AUC by capturing non-linearities and interactions automatically. |
| **Robustness & maintenance** | Stable, monotonic, less prone to overfitting; WoE handles missing values and outliers gracefully. | More sensitive to overfitting; needs careful tuning and ongoing monitoring. |
| **Cost & operations** | Cheap to train, fast to serve, simple to monitor. | More compute, more tuning, more complex serving and monitoring. |

**The core trade-off is interpretability and regulatory defensibility versus raw
predictive power.** In a regulated financial context the bias is deliberately
toward the interpretable model: a slightly less accurate model that can be
explained, audited, and approved is more valuable than a marginally more accurate
one that cannot be defended to a regulator or to a customer who is denied credit.

A pragmatic strategy — and the one this project follows — is to **build both**:
use the WoE-based Logistic Regression as the transparent, deployable baseline and
benchmark, and use Gradient Boosting to quantify the performance ceiling and to
challenge the simple model, while documenting the comparison so the final choice
is evidence-based.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn src.api.main:app --reload

# Or with Docker
docker-compose up --build

# Run tests
pytest
```

## References

- [Credit Scoring Statistical Analysis (Sinica)](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [Alternative Credit Scoring (HKMA)](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Credit Scoring Approaches Guidelines (World Bank)](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [How to Develop a Credit Risk Model and Scorecard (Towards Data Science)](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [Credit Risk — Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
