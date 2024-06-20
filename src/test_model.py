import argparse
import logging
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error

MODEL_SAVE_PATH = 'models/catboost_v1.joblib'
TEST_DATA = 'data/proc/test.csv'

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/test_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')


def main(args):
    col = ['author_type', 'floor', 'floors_count', 'rooms_count', 'total_meters', 'underground']
    
    df_test = pd.read_csv(TEST_DATA)
    x_test = df_test[col]
    y_test = df_test['price']
    model = load(args.model)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_pred, y_test)
    logger.info(f'Test model {args.model} on {TEST_DATA}, MAE = {mae:.0f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)
