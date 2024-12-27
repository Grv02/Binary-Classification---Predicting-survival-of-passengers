import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

def train_experimental(
        X, 
        y, 
        test_size=0.2, 
        random_state=42,
        param_dist=None,
        n_iter=5
    ):
    """
    Trains an experimental XGBoost model with hyperparameter tuning
    using RandomizedSearchCV.
    """
    if param_dist is None:
        # Example parameter grid
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=random_state)
    
    random_search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=random_state
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    y_pred = best_model.predict(X_val)
    y_pred_prob = best_model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_prob)
    
    return best_model, acc, auc
