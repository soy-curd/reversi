# ====================
# パラメータ更新部
# ====================

# パッケージのインポート
import os
from dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle

# パラメータの準備
RN_EPOCHS = 100  # 学習回数

MODEL_PATH = os.environ.get('MODEL_PATH', './model')
DATA_PATH = os.environ.get('DATA_PATH', './data')

BEST_PATH = os.path.join(MODEL_PATH, 'best.h5')
LATEST_PATH = os.path.join(MODEL_PATH, 'latest.h5')


def load_data(all_history=False):

    # 学習データの読み込み
    history_paths = Path(DATA_PATH).glob('*.history')
    if not all_history:
        history_paths = [sorted(history_paths)[-1]]

    data = []
    for history_path in history_paths:
        with history_path.open(mode='rb') as f:
            data += pickle.load(f)

    return data


def train_network(all_history=False):
    '''
    デュアルネットワークの学習
    '''
    # 学習データの読み込み
    print("hoge", all_history)
    history = load_data(all_history=all_history)
    xs, y_policies, y_values = zip(*history)

    # 学習のための入力データのシェイプの変換
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # ベストプレイヤーのモデルの読み込み
    model = load_model(BEST_PATH)

    # モデルのコンパイル
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    # 学習率
    def step_decay(epoch):
        x = 0.001
        if epoch >= 50:
            x = 0.0005
        if epoch >= 80:
            x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    # 出力
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
        print('\rTrain {}/{}'.format(epoch + 1, RN_EPOCHS), end=''))

    # 学習の実行
    model.fit(xs, [y_policies, y_values], batch_size=128, epochs=RN_EPOCHS,
              verbose=0, callbacks=[lr_decay, print_callback])
    print('')

    # 最新プレイヤーのモデルの保存
    model.save(LATEST_PATH)

    # モデルの破棄
    K.clear_session()
    del model


# 動作確認
if __name__ == '__main__':
    train_network(all_history=True)
