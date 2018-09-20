"""
python3
バンディット問題を強化学習するプログラム

スロットマシンは２つ存在していて、それぞれの確率は[0.4, 0.6]
当たると報酬が１もらえる

方策はgreedyとε-greedyとε減衰の３つ(このコードにはgreedyしか実装されていない)

残りの二つを実装するのが課題　(このコードから編集するなら)
"""

import numpy as np
import matplotlib.pyplot as plt
import random


# バンディットタスク
class Bandit():
    def __init__(self):
        # バンディットの設定
        self.probability = np.asarray([[0.4, 0.6]])
        # スタート地点
        self.start = 0
        # ゴール地点
        self.goal = len(self.probability)

    # 報酬を評価
    def get_reward(self, current_state, action):
        # 受け取るアクションは0か1の2値
        # アタリなら１を返す
        if random.random() <= self.probability[current_state, action]:
            return 1
        # 外れなら0を返す
        else:
            return 0

    # 状態の数を返す
    def get_num_state(self):
        return len(self.probability)

    # 行動の数を返す
    def get_num_action(self):
        return len(self.probability[0])

    # スタート地点の場所を返す(初期化用)
    def get_start(self):
        return self.start

    # ゴール地点の場所を返す
    def get_goal_state(self):
        return self.goal

    # 行動を受け取り、次状態を返す
    def get_next_state(self, current_state, current_action):
        return current_state + 1


# Q学習のクラス
class Q_learning():
    # 学習率、割引率、状態数、行動数を定義する
    def __init__(self, learning_rate=0.1, discount_rate=0.9, num_state=None, num_action=None):
        self.learning_rate = learning_rate  # 学習率
        self.discount_rate = discount_rate  # 割引率
        self.num_state = num_state  # 状態数
        self.num_action = num_action  # 行動数
        # Qテーブルを初期化
        self.Q = np.zeros((self.num_state+1, self.num_action))

    # Q値の更新
    # 現状態、選択した行動、得た報酬、次状態を受け取って更新する
    def update_Q(self, current_state, current_action, reward, next_state):
        # TD誤差の計算
        TD_error = (reward
                    + self.discount_rate
                    * max(self.Q[next_state])
                    - self.Q[current_state, current_action])
        # Q値の更新
        self.Q[current_state, current_action] += self.learning_rate * TD_error

    # Q値の初期化
    def init_params(self):
        self.Q = np.zeros((self.num_state+1, self.num_action))

    # Q値を返す
    def get_Q(self):
        return self.Q


# 方策クラス
class Greedy():  # greedy方策
    # 行動価値を受け取って行動番号を返す
    def serect_action(self, value, current_state):
        return np.argmax(value[current_state])


# エージェントクラス
class Agent():
    def __init__(self, value_func="Q_learning", policy="greedy", learning_rate=0.1, discount_rate=0.9, n_state=None, n_action=None):
        # 価値更新方法の選択
        if value_func == "Q_learning":
            self.value_func = Q_learning(num_state=n_state, num_action=n_action)

        # 方策の選択
        if policy == "greedy":
            self.policy = Greedy()

    # パラメータ更新(基本呼び出し)
    def update(self, current_state, current_action, reward, next_state):
        self.value_func.update_Q(current_state, current_action, reward, next_state)

    # 行動選択(基本呼び出し)
    def serect_action(self, current_state):
        return self.policy.serect_action(self.value_func.get_Q(), current_state)

    # 行動価値の表示
    def print_value(self):
        print(self.value_func.get_Q())

    # 所持パラメータの初期化
    def init_params(self):
        self.value_func.init_params()


# メイン関数
def main():
    # ハイパーパラメータ等の設定
    task = Bandit()  # タスク定義

    SIMULATION_TIMES = 1  # シミュレーション回数
    EPISODE_TIMES = 100  # エピソード回数

    agent = Agent(n_state=task.get_num_state(), n_action=task.get_num_action())  # エージェントの設定

    sumreward_graph = np.zeros(EPISODE_TIMES)  # グラフ記述用の報酬記録

    # トレーニング開始
    print("トレーニング開始")
    for simu in range(SIMULATION_TIMES):
        agent.init_params()  # エージェントのパラメータを初期化
        for epi in range(EPISODE_TIMES):
            current_state = task.get_start()  # 現在地をスタート地点に初期化

            while True:
                # 行動選択
                action = agent.serect_action(current_state)
                # 報酬を観測
                reward = task.get_reward(current_state, action)
                sumreward_graph[epi] += reward
                # 次状態を観測
                next_state = task.get_next_state(current_state, action)
                # Q価の更新
                agent.update(current_state, action, reward, next_state)
                # 次状態が終端状態であれば終了
                if next_state == task.get_goal_state():
                    break

    print("Q値の表示")
    agent.print_value()

    print("グラフ表示")
    plt.plot(sumreward_graph / SIMULATION_TIMES, label="greedy")  # グラフ書き込み
    plt.legend()  # 凡例を付ける
    plt.title("reward")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("sum reward")  # y軸のラベルを付ける
    plt.show()  # グラフを表示

main()
