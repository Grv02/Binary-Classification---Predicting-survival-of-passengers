from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def train_baseline(X, y, test_size=0.2, random_state=42):
    """
    Trains a baseline logistic regression model and returns performance metrics.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_val)
    y_pred_prob = clf.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_prob)
    
    return clf, acc, auc
