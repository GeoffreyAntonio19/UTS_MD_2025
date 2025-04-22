# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# ==== UTILITY FUNCTION ====
def gdrive_to_direct_link(gdrive_link):
    file_id = gdrive_link.split('/d/')[1].split('/')[0]
    return f'https://drive.google.com/uc?id={file_id}'

# ==== EDA CLASS ====
import io

class EDA:
    @staticmethod
    def run(data):
        # Use StringIO to capture the output of info()
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        
        # Now you can display this in Streamlit
        st.text(info_str)

        # Other analysis outputs can go here
        st.subheader("Descriptive Statistics")
        st.write(data.describe(include='all'))

        st.subheader("Missing Values Count")
        st.write(data.isnull().sum())

        if 'Canceled' in data.columns or 'Not_Canceled' in data.columns:
            st.subheader("Target Value Counts")
            st.write(data.iloc[:, -1].value_counts())

# ==== DATA LOADER CLASS ====
class DataLoader:
    def __init__(self, gdrive_link, target_column, drop_columns=[]):
        self.gdrive_link = gdrive_link
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.data = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        direct_link = gdrive_to_direct_link(self.gdrive_link)
        self.data = pd.read_csv(direct_link)
        st.success("✅ Data successfully loaded.")

    def preprocess(self):
        if self.drop_columns:
            self.data.drop(columns=self.drop_columns, inplace=True)
            st.info(f"Dropped Columns: {self.drop_columns}")

        EDA.run(self.data)

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        y_encoded = self.label_encoder.fit_transform(y)

        num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        self.X_train = self.preprocessor.fit_transform(X_train)
        self.X_test = self.preprocessor.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        st.success("✅ Preprocessing completed.")

# ==== TRAINER CLASS ====
class Trainer:
    def __init__(self, model=None):
        self.model = model or XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        st.success("✅ Model training completed.")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.metric(label="🎯 Accuracy", value=f"{acc:.4f}")
        return acc

# ==== HYPERPARAMETER TUNER CLASS ====
class HyperparameterTuner:
    def __init__(self, model, param_grid, cv=5, scoring='accuracy', verbose=1):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.best_estimator_ = None

    def tune(self, X_train, y_train):
        st.info("🔍 Tuning hyperparameters...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            verbose=self.verbose,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.best_estimator_ = grid_search.best_estimator_
        st.success("✅ Hyperparameter tuning complete.")
        st.json(grid_search.best_params_)
        return self.best_estimator_

# ==== MASTER PIPELINE ====
class MachineLearningPipeline:
    def __init__(self, gdrive_link, target_column, drop_columns=[]):
        self.data_loader = DataLoader(gdrive_link, target_column, drop_columns)
        self.trainer = Trainer()

    def run(self, tune_hyperparameters=False):
        self.data_loader.load_data()
        self.data_loader.preprocess()

        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
            base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            tuner = HyperparameterTuner(base_model, param_grid)
            best_model = tuner.tune(self.data_loader.X_train, self.data_loader.y_train)
            self.trainer = Trainer(model=best_model)

        self.trainer.train(self.data_loader.X_train, self.data_loader.y_train)
        return self.trainer.evaluate(self.data_loader.X_test, self.data_loader.y_test)

# ==== STREAMLIT UI ====
def main():
    st.title("🚀 Hotel Booking Cancellation Prediction App")

    gdrive_link = st.text_input("Enter Google Drive Link to Dataset", 
        value="https://drive.google.com/file/d/1qPLgQzEVtMt3jw695tYvWoBgBWpIRZyB/view?usp=sharing")

    run_tuning = st.checkbox("🔧 Perform Hyperparameter Tuning?", value=True)

    if st.button("Start Pipeline"):
        pipeline = MachineLearningPipeline(
            gdrive_link=gdrive_link,
            target_column="booking_status",
            drop_columns=['Booking_ID']
        )
        pipeline.run(tune_hyperparameters=run_tuning)

if __name__ == "__main__":
    main()
