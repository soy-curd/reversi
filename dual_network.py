# ====================
# デュアルネットワークの作成
# ====================

# パッケージのインポート
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

BOARD_SIZE = 8
MODEL_PATH = os.environ.get('MODEL_PATH', './model')
# パラメータの準備
DN_FILTERS = 128  # 畳み込み層のカーネル数（本家は256）
DN_RESIDUAL_NUM = 16  # 残差ブロックの数（本家は19）
DN_INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, 2)  # 入力シェイプ
DN_OUTPUT_SIZE = BOARD_SIZE * BOARD_SIZE + 1  # 行動数(配置先(6*6)+パス(1))


def conv(filters):
    '''
    畳み込み層の作成
    '''

    return Conv2D(filters, 3, padding='same', use_bias=False,
                  kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))


def residual_block():
    '''
    残差ブロックの作成
    '''

    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x
    return f


def dual_network():
    '''
    デュアルネットワークの作成
    '''

    # モデル作成済みの場合は無処理
    if os.path.exists(os.path.join(MODEL_PATH, 'best.h5')):
        return

    # 入力層
    input = Input(shape=DN_INPUT_SHAPE)

    # 畳み込み層
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 残差ブロック x 16
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    # プーリング層
    x = GlobalAveragePooling2D()(x)

    # ポリシー出力
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation='softmax', name='pi')(x)

    # バリュー出力
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)

    # モデルの作成
    model = Model(inputs=input, outputs=[p, v])

    # モデルの保存
    os.makedirs(MODEL_PATH, exist_ok=True)  # フォルダがない時は生成
    model.save(os.path.join(MODEL_PATH, 'best.h5'))  # ベストプレイヤーのモデル

    # モデルの破棄
    K.clear_session()
    del model


# 動作確認
if __name__ == '__main__':
    dual_network()
