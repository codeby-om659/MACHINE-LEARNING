import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ---------------------------------------------------------
# Step 1: Realistic Production Dataset Generation
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 1000

data = {
    'tenure_months': np.random.randint(1, 72, size=n_samples),
    'monthly_charges': np.random.uniform(20.0, 120.0, size=n_samples),
    'total_charges': np.random.uniform(100.0, 8000.0, size=n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], size=n_samples),
    'tech_support': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.4, 0.6])
}

df = pd.DataFrame(data)

# Churn logic with probabilistic noise
churn_prob = (
    (df['contract_type'] == 'Month-to-month') * 0.35 +
    (df['monthly_charges'] > 80) * 0.25 -
    (df['tenure_months'] / 72) * 0.40 +
    (df['tech_support'] == 'No') * 0.20 +
    np.random.normal(0, 0.1, size=n_samples)
)

df['churn'] = (churn_prob > 0.3).astype(int)

print("--- Dataset Sample ---")
print(df.head(), "\n")

# ---------------------------------------------------------
# Step 2: Feature & Target Split + Train-Test Split
# ---------------------------------------------------------
X = df.drop(columns=['churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Step 3: Production Pipeline Setup (Preprocessing + Model)
# ---------------------------------------------------------
numeric_features = ['tenure_months', 'monthly_charges', 'total_charges']
categorical_features = ['contract_type', 'payment_method', 'tech_support']

# Numerical Pipeline: Standard Scaling
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical Pipeline: One-Hot Encoding
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor: Column-wise Transformation Combine Karna
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# Full Pipeline with High-Performance Gradient Boosting Model
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, max_iter=150))
])

# ---------------------------------------------------------
# Step 4: Model Training
# ---------------------------------------------------------
full_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 5: Evaluation & Metrics
# ---------------------------------------------------------
y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("--- Model Evaluation ---")
print("ROC-AUC Score:", round(roc_auc_score(y_test, y_pred_proba), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# Step 6: Direct Inference / Prediction Pipeline
# ---------------------------------------------------------
new_customer = pd.DataFrame([{
    'tenure_months': 3,
    'monthly_charges': 95.5,
    'total_charges': 286.5,
    'contract_type': 'Month-to-month',
    'payment_method': 'Electronic check',
    'tech_support': 'No'
}])

churn_risk = full_pipeline.predict_proba(new_customer)[0][1]
is_churning = full_pipeline.predict(new_customer)[0]

print(f"New Customer Churn Probability: {churn_risk * 100:.2f}%")
print(f"Prediction: {'Churn Alert! (Leave Karega)' if is_churning == 1 else 'Retained (Rukega)'}")
