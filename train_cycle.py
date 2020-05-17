# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('{}  {} s'.format(method.__name__, (te - ts)))
        return result
    return timed


# デュアルネットワークの作成
dual_network()

for i in range(10):
    print('Train', i, '====================')
    # セルフプレイ部
    timeit(self_play)()

    # # パラメータ更新部
    timeit(train_network)(all_history=True)

    # 新パラメータ評価部
    timeit(evaluate_network)()
