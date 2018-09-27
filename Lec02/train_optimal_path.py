"""
python3
崖ありグリッドワールドでの最適経路問題を強化学習するプログラム

今度はQ-learningとsarsaを実装してその学習の違いを見る

このコードから編集するなら、グリッドワールドに崖(マイナス報酬部分のマス)の環境を実装、sarsaを実装
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import sys


# 環境のベースクラス
class Environment(object):
    def __init__(self):
        self.current_state = 0
        self.next_state = 0
        self.reward = 0
        self.num_state = 0
        self.num_action = 0
        self.start_state = 0
        self.goal_state = 0

    def get_current_state(self):
        return self.current_state

    def get_next_state(self):
        return self.next_state

    def get_reward(self):
        return self.reward

    def get_num_state(self):
        return self.num_state

    def get_num_action(self):
        return self.num_action

    def get_start(self):
        return self.start_state

    def get_goal_state(self):
        return self.goal_state


# グリッドワールドの作成
class GlidWorld(Environment):
    # 簡単なステージを作る
    def __init__(self, row, col, start, goal):
        super().__init__()
        self.row = row
        self.col = col
        self.start_state = start
        self.goal_state = goal
        self.num_state = self.row * self.col
        self.num_action = 5

    # 座標に変換
    def coord_to_state(self, row, col):
        return ((row * self.col) + col)

    # 座標からx軸を算出
    def state_to_row(self, state):
        return ((int)(state / self.col))

    # 座標からy軸を算出
    def state_to_col(self, state):
        return (state % self.col)

    # 次の座標を算出
    def get_next_state(self, state, action):
        UPPER = 0
        LOWER = 1
        LEFT = 2
        RIGHT = 3
        STOP = 4

        row = self.state_to_row(state)
        col = self.state_to_col(state)

        if action == UPPER:
            if (row) > 0:
                row -= 1
        elif action == LOWER:
            if (row) < (self.row-1):
                row += 1
        elif action == RIGHT:
            if (col) < (self.col-1):
                col += 1
        elif action == LEFT:
            if (col) > 0:
                col -= 1
        elif action == STOP:
            pass

        self.next_state = self.coord_to_state(row, col)

        return self.next_state

    # 報酬判定
    def get_reward(self, state):
        if state == self.goal_state:
            return 1
        else:
            return 0


# Q学習のクラス
class Q_learning():
    # 学習率、割引率、状態数、行動数を定義する
    def __init__(self, learning_rate=0.1, discount_rate=0.9, num_state=None, num_action=None):
        self.learning_rate = learning_rate  # 学習率
        self.discount_rate = discount_rate  # 割引率
        self.num_state = num_state  # 状態数
        self.num_action = num_action  # 行動数
        # Qテーブルを初期化
        self.Q = np.zeros((self.num_state, self.num_action))

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
        self.Q = np.zeros((self.num_state, self.num_action))

    # Q値を返す
    def get_Q(self):
        return self.Q


# 方策クラス
class Greedy(object):  # greedy方策
    # 行動価値を受け取って行動番号を返す
    def select_action(self, value, current_state):
        idx = np.where(value[current_state] == max(value[current_state]))
        return random.choice(idx[0])
    
    def init_params(self):
        pass

    def update_params(self):
        pass


class EpsGreedy(Greedy):
    def __init__(self, eps):
        self.eps = eps

    def select_action(self, value, current_state):
        if random.random() < self.eps:
            return random.choice(range(len(value[current_state])))

        else:
            return super().select_action(value, current_state)


class EpsDecGreedy(EpsGreedy):
    def __init__(self, eps, eps_min, eps_decrease):
        super().__init__(eps)
        self.eps_init = eps
        self.eps_min = eps_min
        self.eps_decrease = eps_decrease

    def init_params(self):
        self.eps = self.eps_init

    def update_params(self):
        self.eps -= self.eps_decrease


# エージェントクラス
class Agent():
    def __init__(self, value_func="Q_learning", policy="greedy", learning_rate=0.1, discount_rate=0.9, eps=None, eps_min=None, eps_decrease=None, n_state=None, n_action=None):
        # 価値更新方法の選択
        if value_func == "Q_learning":
            self.value_func = Q_learning(num_state=n_state, num_action=n_action)
        
        else:
            print("error:価値関数候補が見つかりませんでした")
            sys.exit()

        # 方策の選択
        if policy == "greedy":
            self.policy = Greedy()
        
        elif policy == "eps_greedy":
            self.policy = EpsGreedy(eps=eps)

        elif policy == "eps_dec_greedy":
            self.policy = EpsDecGreedy(eps=eps, eps_min=eps_min, eps_decrease=eps_decrease)

        else:
            print("error:方策候補が見つかりませんでした")
            sys.exit()

    # パラメータ更新(基本呼び出し)
    def update(self, current_state, current_action, reward, next_state):
        self.value_func.update_Q(current_state, current_action, reward, next_state)
        self.policy.update_params()

    # 行動選択(基本呼び出し)
    def select_action(self, current_state):
        return self.policy.select_action(self.value_func.get_Q(), current_state)

    # 行動価値の表示
    def print_value(self):
        print(self.value_func.get_Q())

    # 所持パラメータの初期化
    def init_params(self):
        self.value_func.init_params()
        self.policy.init_params()


# メイン関数
def main():
    # ハイパーパラメータ等の設定
    task = GlidWorld(row=7, col=7, start=0, goal=48)  # タスク定義
    SIMULATION_TIMES = 100  # シミュレーション回数
    EPISODE_TIMES = 1000  # エピソード回数

    # エージェントの設定
    agent = {}
    agent[0] = Agent(policy="greedy", n_state=task.get_num_state(), n_action=task.get_num_action())
    agent[1] = Agent(policy="eps_greedy", eps=0.1, n_state=task.get_num_state(), n_action=task.get_num_action())
    agent[2] = Agent(policy="eps_dec_greedy", eps=1.0, eps_min=0.0, eps_decrease=0.001, n_state=task.get_num_state(), n_action=task.get_num_action())

    # グラフ記述用の記録
    reward_graph = np.zeros((len(agent), EPISODE_TIMES))
    step_graph = np.zeros((len(agent), EPISODE_TIMES))

    # トレーニング開始
    print("トレーニング開始")
    for simu in range(SIMULATION_TIMES):
        print("simu:{}" .format(simu))
        for n_agent in range(len(agent)):
            agent[n_agent].init_params()  # エージェントのパラメータを初期化

        for epi in range(EPISODE_TIMES):
            for n_agent in range(len(agent)):
                current_state = task.get_start()  # 現在地をスタート地点に初期化
                step = 0

                while True:
                    # 行動選択
                    action = agent[n_agent].select_action(current_state)
                    # 次状態を観測
                    next_state = task.get_next_state(current_state, action)
                    # 報酬を観測
                    reward = task.get_reward(next_state)
                    reward_graph[n_agent, epi] += reward
                    # Q価の更新
                    agent[n_agent].update(current_state, action, reward, next_state)
                    step += 1
                    current_state = next_state
                    # 次状態が終端状態であれば終了
                    if next_state == task.get_goal_state() or step == 100:
                        step_graph[n_agent, epi] = step
                        break

    # print("Q値の表示")
    # for n_agent in range(len(agent)):
        # agent[n_agent].print_value()

    print("グラフ表示")
    # グラフ書き込み
    plt.plot(reward_graph[0] / SIMULATION_TIMES, label="greedy")
    plt.plot(reward_graph[1] / SIMULATION_TIMES, label="eps_greedy")
    plt.plot(reward_graph[2] / SIMULATION_TIMES, label="eps_dec_greedy")
    plt.legend()  # 凡例を付ける
    plt.title("reward")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("sum reward")  # y軸のラベルを付ける
    plt.show()  # グラフを表示

    plt.figure()

    print("ステップ数表示")
    plt.plot(step_graph[0] / SIMULATION_TIMES, label="greedy")
    plt.plot(step_graph[1] / SIMULATION_TIMES, label="eps_greedy")
    plt.plot(step_graph[2] / SIMULATION_TIMES, label="eps_dec_greedy")
    plt.legend()  # 凡例を付ける
    plt.title("step")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("sum reward")  # y軸のラベルを付ける
    plt.show()  # グラフを表示

    plt.figure()


main()
