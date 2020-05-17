# ====================
# セルフプレイ部
# ====================

# パッケージのインポート
from game import State
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
import os
import sys
import time
import random
from train_network import load_data

# パラメータの準備
# SP_GAME_COUNT = 500  # セルフプレイを行うゲーム数（本家は25000）
SP_GAME_COUNT = 10  # 500だと2回回せない説

SP_TEMPERATURE = 1.0  # ボルツマン分布の温度パラメータ

MODEL_PATH = os.environ.get('MODEL_PATH', './model')
DATA_PATH = os.environ.get('DATA_PATH', './data')
BEST_PATH = os.path.join(MODEL_PATH, 'best.h5')
LATEST_PATH = os.path.join(MODEL_PATH, 'latest.h5')


def first_player_value(ended_state):
    '''
    先手プレイヤーの価値
    '''

    # 1:先手勝利, -1:先手敗北, 0:引き分け
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0


def write_data(history, suffix=""):
    '''
    学習データの保存
    '''

    now = datetime.now()
    os.makedirs(DATA_PATH, exist_ok=True)  # フォルダがない時は生成
    path = os.path.join(DATA_PATH, '{:04}{:02}{:02}{:02}{:02}{:02}{}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.microsecond, suffix))
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


def load_state():

    state_paths = Path(DATA_PATH).glob('*.state')
    state_path = random.choice(list(state_paths))
    if not state_paths:
        print("no state")
        return None
    with state_path.open(mode='rb') as f:
        state = pickle.load(f)

    print("state is ", state_path)
    print(state)
    return state


def save_state(state):
    now = datetime.now()
    os.makedirs(DATA_PATH, exist_ok=True)  # フォルダがない時は生成
    path = os.path.join(DATA_PATH, 'state_{:04}{:02}{:02}{:02}{:02}{:02}.state'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.microsecond))
    with open(path, mode='wb') as f:
        pickle.dump(state, f)


def concat_hisotry():
    data = load_data(all_history=True)
    write_data(data, suffix="_concated")


def play(model, using_saved_state=False, saving_ontheway_state=False):
    '''
    1ゲームの実行
    '''

    # 学習データ
    history = []

    # 状態の生成
    if using_saved_state:
        state = load_state()
        if not state:
            state = State()
    else:
        state = State()

    starttime = time.time()
    print('')
    while True:
        # ゲーム終了時
        if state.is_done():
            endtime = time.time()
            print("first player is ", "lose" if state.is_lose() else "win")
            print("first player num:", state.piece_count(state.pieces))
            print('elapsed time',  endtime - starttime)
            print(state)
            break

        # 合法手の確率分布の取得

        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)

        # 学習データに状態と方策を追加
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None])

        # 行動の取得
        if len(history) % 10 == 0:
            print("state len: ", len(history))
            print(state)

        if saving_ontheway_state and len(history) == 25:
            save_state(state)
        action = np.random.choice(state.legal_actions(), p=scores)

        # 次の状態の取得
        state = state.next(action)

    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history


def self_play(x=None):
    '''
    セルフプレイ
    '''
    print("start: ", x)
    np.random.seed()  # マルチプロセス対応

    # 学習データ
    history = []

    # ベストプレイヤーのモデルの読み込み
    model = load_model(BEST_PATH)

    # 複数回のゲームの実行
    for i in range(SP_GAME_COUNT):
        # 1ゲームの実行
        h = play(model)
        history.extend(h)

        # 出力
        print('\rSelfPlay {}/{}'.format(i+1, SP_GAME_COUNT), end='')
    print('')

    # 学習データの保存
    write_data(history)

    # モデルの破棄
    K.clear_session()
    del model


def do_multi(f=self_play, process_num=2):
    import random
    import os
    import multiprocessing as mp
    from queue import Empty
    import math
    import time

    p = mp.Pool(process_num)

    y = range(process_num)

    results = p.map(f, y)


# 動作確認
if __name__ == '__main__':
    is_multi = len(sys.argv) > 1 and sys.argv[1] == 'multi'

    for n in range(100):

        print("iterate: ", n)
        if is_multi:
            process_num = 8
            print("process: ", process_num)
            do_multi(self_play, process_num=process_num)
        else:
            self_play()
