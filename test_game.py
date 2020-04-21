
# パッケージのインポート
from game import State, random_action
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
from shutil import copy
import numpy as np
import math

# パラメータの準備
EN_GAME_COUNT = 100  # 1評価あたりのゲーム数（本家は400）
EN_GAME_COUNT = 10  # 1評価あたりのゲーム数（本家は400）

EN_TEMPERATURE = 1.0


def argmax(collection, key=None):
    return collection.index(max(collection))


def playout(state):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    return -playout(state.next(random_action(state)))  # 再帰的に探索


# モンテカルロ
PLAYOUT_NUM = 100


def mcs_action(state):
    legal_actions = state.legal_actions()
    values = [0] * len(legal_actions)
    for i, action in enumerate(legal_actions):
        for _ in range(PLAYOUT_NUM):
            values[i] += -playout(state.next(action))

    return legal_actions[argmax(values)]


def mcts_actions(state):
    # モンテカルロ木探索のノードの定義
    class Node:
        # ノードの初期化
        def __init__(self, state):
            self.state = state  # 状態
            self.w = 0  # 累計価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード群

        # 局面の価値の計算
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                if self.n == 10:
                    self.expand()
                return value

            # 子ノードが存在する時
            else:

                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))

        # アーク評価値が最大の子ノードを取得
        def next_child_node(self):

            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            t = 0
            for c in self.child_nodes:
                t += c.n

            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(
                    -child_node.w / child_node.n +
                    (2*math.log(t) / child_node.n) ** 0.5
                )

            return self.child_nodes[argmax(ucb1_values)]

    # 現在の局面のノードの作成
    root_node = Node(state)
    root_node.expand()
    PV_EVALUATE_COUNT = 100

    # 複数回の評価の実行
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]


def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action


def first_player_point(ended_state):
    # 1:先手勝利, 0:先手敗北, 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5


def play(next_actions):
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player(
        ) else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)


# ネットワークの評価
def evaluate_network():
    # # 最新プレイヤーのモデルの読み込み
    # model0 = load_model('./model/best.h5')

    # # PV MCTSで行動選択を行う関数の生成
    # next_action0 = pv_mcts_action(model0, EN_TEMPERATURE)
    next_action0 = mcts_actions
    next_action1 = random_action
    next_actions = (next_action0, next_action1)

    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EN_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 出力
        print('\rEvaluate {}/{}'.format(i + 1, EN_GAME_COUNT), end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)


# 動作確認
if __name__ == '__main__':
    evaluate_network()
