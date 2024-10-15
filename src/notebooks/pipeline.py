import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from cfg.cfg import Config
import wandb
from loguru import logger

class Pipeline:
    def __init__(self):
        self.config = Config()
        self.model = RandomForestClassifier(verbose=1, n_jobs=-1)
        self.vectorizer = TfidfVectorizer(analyzer='char')
        wandb.init(project='obfuscation', reinit=True)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _load_data(self):
        self.data = pd.read_parquet(self.config.data_path)
        self.logger.info("Data loaded successfully.")

    def _vectorize(self):
        self.content_vector = self.vectorizer.fit_transform(self.data['content'])
        self.logger.info("Data vectorized successfully.")

    def _prepare_data(self):
        y = self.data['class']
        X_train, X_test, y_train, y_test = train_test_split(self.content_vector, y, test_size=self.config.test_size, random_state=self.config.random_state)
        self.logger.info("Data prepared for training and testing.")
        return X_train, X_test, y_train, y_test

    def _evaluate_model(self, y_test, y_pred, y_proba):
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])

        wandb.log({'precision': precision, 'recall': recall, 'roc_auc': roc_auc})
        self.logger.info(f'Model precision: {precision}')
        self.logger.info(f'Model recall: {recall}')
        self.logger.info(f'Model ROC AUC: {roc_auc}')

    def run(self):
        self._load_data()
        self._vectorize()
        X_train, X_test, y_train, y_test = self._prepare_data()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        self._evaluate_model(y_test, y_pred, y_proba)

if __name__ == '__main__':
    p = Pipeline()
    p.run()