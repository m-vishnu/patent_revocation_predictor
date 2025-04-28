import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import xgboost as xgb
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import warnings
warnings.filterwarnings('ignore')

def dump_pickle(obj_to_pickle, filepath):
    # Check if file exists
    if os.path.exists(filepath):
        print("File exists. Not saving.")
    else:
        # Save the object to a pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(obj_to_pickle, f)
        print(f"Object saved to {filepath}")


class WeightedClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wraps classifier to pass sample_weight if available"""

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y, sample_weight=None):
        if 'sample_weight' in self.classifier.fit.__code__.co_varnames:
            self.classifier.fit(X, y, sample_weight=sample_weight)
        else:
            self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_params(self, deep=True):
        return {'classifier': self.classifier}

    def set_params(self, **params):
        self.classifier = params['classifier']
        return self


Y_COL = 'Order status manual'
df = pd.read_pickle("data/preprocessed/openai/openai_pca_embeddings.pkl")

df = df[df[Y_COL].notna()]

X = df.drop(columns=[Y_COL])
y_raw = df[Y_COL]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)




# 1. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# 2. Calculate sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)


# 4. Define pipeline steps
feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))

pipe = Pipeline([
    # ('preprocessing', preprocessor),
    ('feature_selection', feature_selector),
    ('classifier', WeightedClassifierWrapper(RandomForestClassifier()))  # Placeholder
])



# 5. Define parameter grid
param_grid = [
    {
        'feature_selection__threshold': ['mean', 'median', 0.01],
        'classifier__classifier': [RandomForestClassifier(class_weight='balanced', random_state=42)],
        'classifier__classifier__n_estimators': [100, 200],
        'classifier__classifier__max_depth': [5, 10, None],
        'classifier__classifier__min_samples_split': [2, 5]
    },
    {
        'feature_selection__threshold': ['mean', 'median', 0.01],
        'classifier__classifier': [xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)],
        'classifier__classifier__n_estimators': [100, 150, 200],
        'classifier__classifier__max_depth': [3, 6],
        'classifier__classifier__learning_rate': [0.01,0.05, 0.1],
        'classifier__classifier__subsample': [0.7, 1.0],
        'classifier__classifier__colsample_bytree': [0.2, 0.4, 0.5, 0.7, 1.0]
    },
    {
        'feature_selection__threshold': ['mean', 'median', 0.01],
        'classifier__classifier': [LogisticRegression(class_weight='balanced', solver='saga', max_iter=5000, random_state=42)],
        'classifier__classifier__C': [0.01, 0.1, 1, 10],
        'classifier__classifier__penalty': ['l1', 'l2']
    }
]

# 6. Grid search with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',  # Good for multiclass imbalance
    verbose=1,
    n_jobs=2
)

# 7. Fit
grid_search.fit(X_train, y_train, **{'classifier__sample_weight': sample_weights})

# 8. Best results
print("Best Parameters:\n", grid_search.best_params_)
print("Best CV Score (F1 Macro):", grid_search.best_score_)



import pickle
#save the entire grid search object
with open('outputs/models/openai/openai_grid_search_obj_v2.pkl', 'wb') as f:
    pickle.dump(grid_search, f)

# save the best model
with open('outputs/models/openai/openai_best_model_v2.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)



# from sklearn.metrics import classification_report
#
# y_test_pred = grid_search.predict(X_test)
# print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))
#
# grid_search.predict(X_train)
# X_train_preds = grid_search.predict(X_train)
# X_test_preds = grid_search.predict(X_test)
#
# print(classification_report(y_train, X_train_preds, target_names=label_encoder.classes_))
# print(classification_report(y_test, X_test_preds, target_names=label_encoder.classes_))
#
#
# from sklearn.metrics import confusion_matrix
# label_encoder.classes_
# confusion_matrix(y_test, X_test_preds)
#

# Which features were important?
selected_cols = grid_search.best_estimator_.steps[0][1].get_feature_names_out()
len(grid_search.best_estimator_.steps[1][1].classifier.feature_importances_)

feature_importances = pd.DataFrame({selected_cols[i]:grid_search.best_estimator_.steps[1][1].classifier.feature_importances_[i] for i in range(len(selected_cols))}, index=['Feature Importance']).T.sort_values(ascending=False, by='Feature Importance')

feature_importances.to_csv("outputs/misc/openai/feature_importances_xgb_feature_selected_v2.csv", index=False)


dump_pickle(label_encoder, filepath='outputs/misc/openai/openai_label_encoder_v2.pkl')


dump_pickle(X_train, filepath='data/model_ready/openai/X_train_v2.pkl')
dump_pickle(X_test, filepath='data/model_ready/openai/X_test_v2.pkl')
dump_pickle(y_train, filepath='data/model_ready/openai/y_train_v2.pkl')
dump_pickle(y_test, filepath='data/model_ready/openai/y_test_v2.pkl')


