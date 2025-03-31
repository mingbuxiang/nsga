# nsga3_rl.py
from nsga3 import NSGAIII, Individual
import random
import pandas as pd

class NSGAIII_RL(NSGAIII):
    def __init__(self, env, config):
        super().__init__(env, config)
        # 增加 RL 所需参数
        self.actions = ['mutation_1', 'mutation_2', 'crossover_1', 'crossover_2']
        self.learning_rate = 0.1
        self.gamma = 0.9
        self.epsilon = 0.9  # ε 贪婪策略
        # 初始化 Q 表为空，行索引用状态字符串
        self.q_table = pd.DataFrame(columns=self.actions, dtype=float)

    def choose_action(self, state_key: str) -> str:
        """根据当前状态选择操作"""
        if state_key not in self.q_table.index:
            self.q_table.loc[state_key] = [0] * len(self.actions)
        if random.random() < self.epsilon:
            # 选择 Q 值最大的操作（当有多个最大时随机选取一个）
            state_actions = self.q_table.loc[state_key]
            best_actions = state_actions[state_actions == state_actions.max()].index.tolist()
            return random.choice(best_actions)
        else:
            return random.choice(self.actions)

    def update_q_table(self, state_key: str, action: str, reward: float, next_state_key: str):
        """更新 Q 表"""
        if next_state_key not in self.q_table.index:
            self.q_table.loc[next_state_key] = [0] * len(self.actions)
        q_predict = self.q_table.loc[state_key, action]
        q_target = reward + self.gamma * self.q_table.loc[next_state_key].max()
        self.q_table.loc[state_key, action] += self.learning_rate * (q_target - q_predict)

    def apply_action(self, action: str, parent1_state, parent2_state):
        """根据动作调用相应的遗传操作"""
        if action == 'mutation_1':
            return self._mutation(parent1_state, param=0.3)
        elif action == 'mutation_2':
            return self._mutation(parent2_state, param=0.1)
        elif action == 'crossover_1':
            return self._crossover(parent1_state, parent2_state, param=0.3)
        else:  # action == 'crossover_2'
            return self._crossover(parent1_state, parent2_state, param=0.1)

    def calculate_reward(self, offspring: Individual, parent1: Individual, parent2: Individual) -> float:
        """
        奖励函数
        """
        off_cost = offspring.objectives[0]
        parent_avg_cost = (parent1.objectives[0] + parent2.objectives[0]) / 2.0
        return 1.0 if off_cost < parent_avg_cost else -1.0

    def optimize(self):
        """
        重写 optimize
        """
        population = []
        for _ in range(self.config.pop_size):
            state = self.env.create_random_state()
            state = self.env.repair_solution(state)
            objectives = self.env.calculate_objectives(state)
            population.append(Individual(state=state, objectives=objectives))

        all_costs = []
        all_emissions = []

        # 2. 迭代演化过程
        for generation in range(self.config.n_generations):
            offspring = []
            while len(offspring) < self.config.pop_size:
                # 选择两个父代
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                state_key = str(generation % 100)
                # RL 选择操作
                action = self.choose_action(state_key)
                # 应用 RL 选出的操作生成子代
                child_state = self.apply_action(action, parent1.state, parent2.state)

                # 修复并评估新解
                child_state = self.env.repair_solution(child_state)
                objectives = self.env.calculate_objectives(child_state)
                child = Individual(state=child_state, objectives=objectives)
                offspring.append(child)

                # 计算奖励，并更新 Q 表
                reward = self.calculate_reward(child, parent1, parent2)
                next_state_key = str((generation + 1) % 100)
                self.update_q_table(state_key, action, reward, next_state_key)

            combined = population + offspring
            fronts = self._fast_non_dominated_sort(combined)
            new_population = []
            front_idx = 0
            while len(new_population) + len(fronts[front_idx]) <= self.config.pop_size:
                new_population.extend(fronts[front_idx])
                front_idx += 1
                if front_idx >= len(fronts):
                    break

            if len(new_population) < self.config.pop_size and front_idx < len(fronts):
                last_front = fronts[front_idx]
                self._associate_with_reference_points(last_front)
                last_front.sort(key=lambda x: x.distance)
                new_population.extend(last_front[:self.config.pop_size - len(new_population)])
            
            population = new_population

            min_cost = min(ind.objectives[0] for ind in population)
            min_emission = min(ind.objectives[1] for ind in population)
            all_costs.append(min_cost)
            all_emissions.append(min_emission)

            print(f"Generation {generation}: Min Cost = {min_cost:.2f}, Min Emission = {min_emission:.2f}")

        return population, all_costs, all_emissions
