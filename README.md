# 💳 Loan Default Prediction

A clean, beginner-friendly, end-to-end Machine Learning web application built with **FastAPI**, **Jinja2 Templates**, and **Scikit-Learn**. It predicts whether a borrower is likely to default on a loan.

## 🛠️ Tech Stack

* **Backend:** FastAPI, Uvicorn
* **Frontend:** HTML + Jinja2 templates
* **Machine Learning:** Scikit-Learn (Gradient Boosting Classifier), Joblib
* **Training environment:** Google Colab / Jupyter Notebook

## 📁 Simple Project Structure

```
loan_api/
├── templates/
│   └── index.html                      # Single frontend file using Jinja2
├── gradient_boosting_model.joblib      # Best Classifier (Gradient Boosting)
├── loan_default_risk.ipynb             # Training & experimentation notebook
├── cs-training.csv                     # Dataset (not on GitHub, download from Kaggle)
├── main.py                             # Single FastAPI backend with Jinja2 rendering
├── requirements.txt                    # Minimal dependencies
├── .gitignore
└── README.md
```

## 🔬 Machine Learning Pipeline & Results

1. **Dataset**
   * Give Me Some Credit (`cs-training.csv`) from Kaggle
   * Not included in the repo (excluded via `.gitignore`). Download it from Kaggle and place it in the project folder to retrain.
   * Test set: 30,000 borrowers (2,005 defaulters, about 6.7%)
2. **Problem Type**
   * Binary classification: will the borrower default (1) or not (0)? The classes are heavily imbalanced.
3. **Model Comparison**

   | Model | ROC-AUC | PR-AUC |
   |---|---|---|
   | Logistic Regression | 0.855 | 0.381 |
   | Random Forest | 0.866 | 0.397 |
   | **Gradient Boosting (selected)** | **0.868** | **0.402** |

4. **Preprocessing**
  * Missing `dependents` values were filled with 0, on the assumption that a blank means no dependents.
   * Missing `monthly_income` values were filled with the median income.
   * After cleaning, the dataset has no missing values.
5. **Decision Threshold Tuning**
   * Because defaulters are rare, the default 0.50 cutoff misses most of them (recall 17.6%), so the cutoff was tuned.

   | Cutoff | Flagged | Precision | Recall |
   |---|---|---|---|
   | 0.50 | 577 | 0.612 | 0.176 |
   | 0.30 | 1,591 | 0.469 | 0.372 |
   | **0.20** | **2,631** | **0.388** | **0.509** |
   | 0.16 | 3,242 | 0.351 | 0.568 |
   | 0.10 | 5,047 | 0.274 | 0.690 |
   | 0.05 | 9,380 | 0.181 | 0.845 |

6. **Final Evaluation: Gradient Boosting at cutoff 0.20** (test set)
   * Accuracy: 91.3% | ROC-AUC: 0.868 | PR-AUC: 0.402
   * Defaulters (class 1): Precision 38.8% | Recall 50.9% | F1-Score 44.0%
   * Non-defaulters (class 0): Precision 96.4% | Recall 94.2% | F1-Score 95.3%
   * Confusion matrix:

   | | Predicted: No default | Predicted: Default |
   |---|---|---|
   | **Actual: No default** | 26,384 | 1,611 |
   | **Actual: Default** | 985 | 1,020 |

   * Insight: at this cutoff the model catches about half of all defaulters. Accuracy alone is misleading here, since predicting "no default" for everyone would already score about 93%. ROC-AUC and PR-AUC are more reliable measures.

## 🚀 How to Run Locally

1. **Clone the repository**

```
git clone https://github.com/UgguNagalakshmi/loan-default-prediction.git
cd loan-default-prediction
```

2. **Create a virtual environment and install dependencies**

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The scikit-learn version should match the one used to save `gradient_boosting_model.joblib`. If you retrain and replace the model file, update the version in `requirements.txt` to match.

3. **Start the FastAPI app**

```
uvicorn main:app --reload --port 8080
```

4. **Open in browser**

* Web app: http://127.0.0.1:8080/
* Interactive API docs (Swagger UI): http://127.0.0.1:8080/docs

## 🧪 How to Use

1. Open the web app in your browser.
2. Enter the applicant's details in the form.
3. Submit the form to get the predicted default risk.

## 🔁 Retraining the Model

1. Download `cs-training.csv` from Kaggle and place it in the project folder.
2. Open `loan_default_risk.ipynb` in Jupyter or Google Colab and run all cells.
3. Save the new model as `gradient_boosting_model.joblib`, replacing the old file.
4. Run `pip freeze > requirements.txt` so the scikit-learn version matches.

## ⚠️ Notes & Limitations

* The model was trained in Google Colab (scikit-learn 1.6.1) and is currently loaded with scikit-learn 1.9.1, which shows an `InconsistentVersionWarning`. Retrain with the deployed version for production use.
* The results above use a 0.20 decision cutoff. Roughly half of actual defaulters are still missed, and about 6 in 10 flagged borrowers would not actually default.
* This project is for learning and demonstration only. It should **not** be used for real lending decisions.

## 👤 Author

GitHub: [@UgguNagalakshmi](https://github.com/UgguNagalakshmi)