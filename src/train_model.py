"""Train model and save checkpoint"""

import argparse
import logging
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/train_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

TRAIN_DATA = 'data/proc/train.csv'
TEST_DATA = 'data/proc/test.csv'
MODEL_SAVE_PATH = 'models/catboost_v1.joblib'


def main(args):
    col = ['author_type', 'floor', 'floors_count', 'rooms_count', 'total_meters', 'underground']
    
    df_train = pd.read_csv(TRAIN_DATA)
    x_train = df_train[col]
    y_train = df_train['price']

    df_val = pd.read_csv(TEST_DATA)
    x_val = df_val[col]
    y_val = df_val['price']

    model = CatBoostRegressor(
        cat_features=['author_type', 'underground'],
        )

    model.fit(x_train, y_train, verbose=50)
    dump(model, args.model)
    logger.info(f'Saved to {args.model}')

    r2 = model.score(x_train, y_train)
    y_pred = model.predict(x_val)
    mae = mean_absolute_error(y_pred, y_val)
    logger.info(f'R2 = {r2:.3f}     MAE = {mae:.0f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', 
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)
