# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It leverages data preprocessing, anomaly detection (Isolation Forest), supervised learning (XGBoost), and class imbalance handling (SMOTE). A Streamlit web app is included for both manual and batch predictions.

---

## 📂 Dataset

* **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Description**: Transactions made by European cardholders over two days in 2013.
* **Features**:

  * 28 anonymized PCA features (`V1` to `V28`)
  * `Time`, `Amount`
  * `Class` (Target: `0` = Non-Fraud, `1` = Fraud)
* **Note**: The dataset is highly imbalanced with a small number of fraud cases.

---

## 🚀 Features

* ✅ Data cleaning and scaling (`StandardScaler`)
* ✅ Class balancing using **SMOTE**
* ✅ Anomaly detection using **Isolation Forest**
* ✅ Supervised classification with **XGBoost**
* ✅ Model evaluation with:

  * Confusion Matrix
  * Classification Report
  * ROC Curve and AUC
* ✅ Streamlit web app for:

  * Manual input-based fraud prediction
  * Batch prediction via CSV upload

---

## 🏗️ Project Structure

```
├── app.py                    # Streamlit app interface
├── creditcard.csv           # Kaggle dataset (not included)
├── xgb_fraud_model.pkl      # Trained XGBoost model (pickle)
├── amount_time_scaler.pkl   # Scaler for 'Time' and 'Amount'
├── notebook.ipynb           # Full EDA, training, and evaluation
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** &#x20;
   Download `creditcard.csv` from Kaggle and place it in the root project directory. *(Optional if using pre-trained model only for prediction)*

4. **(Optional)** Open `notebook.ipynb` to explore or retrain the model.

---

## 💻 Running the Streamlit App

To launch the app locally:

```bash
streamlit run app.py
```

Then, open your browser and go to: [http://localhost:8501](http://localhost:8501)

### ✅ App Features

* Upload a CSV file for **batch fraud detection**
* Enter transaction details manually for **real-time predictions**

---

## 🔗 Live Demo

You can access the deployed app here:
👉 [https://credit-card-fraud-detection-zzaxdcvpczckq5bj6odyof.streamlit.app/](https://credit-card-fraud-detection-zzaxdcvpczckq5bj6odyof.streamlit.app/)

> Replace this link with the actual URL of your deployed app.

---

## ✍️ Sample Manual Input

| Feature | Example Value |
| ------- | ------------- |
| Time    | 50000         |
| V1      | -1.25         |
| V2      | 2.35          |
| ...     | ...           |
| Amount  | 150.75        |

> ⚠️ 'Time' and 'Amount' are automatically scaled before prediction.

---

## 📈 Future Enhancements

* [ ] Hyperparameter optimization
* [ ] Add more ML models (Random Forest, LightGBM, etc.)
* [ ] Model interpretability (SHAP, LIME)
* [ ] Cloud deployment (AWS/GCP/Azure)
* [ ] Real-time fraud detection pipeline
* [ ] Email/SMS alerts for flagged transactions

---

## 📄 License

This project is intended for **educational and demonstration purposes** only.
Dataset usage is governed by [Kaggle’s Terms of Service](https://www.kaggle.com/terms).

---

## 🙋‍♂️ Contributing

Feel free to open issues or submit pull requests!
You can also customize this README with badges, contributor lists, or app screenshots.

---

*Made with ❤️ for fraud detection enthusiasts.*
