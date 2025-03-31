# nsga3.py

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from environment import PowerSystemEnv, State
from config import NSGAConfig
import random

@dataclass
class Individual:
    state: State
    objectives: Optional[Tuple[float, float]] = None
    rank: int = 0
    distance: float = 0.0

class NSGAIII:
    def __init__(self, env: PowerSystemEnv, config: NSGAConfig):
        self.env = env
        self.config = config
        self.reference_points = self._generate_reference_points()

    def _generate_reference_points(self) -> np.ndarray:
        """Generates reference points (uniformly distributed on a unit hyperplane)."""
        p = self.config.n_divisions
        H = 1 / p  # Distance between reference points
        ref_points = []

        def generate_recursive(dim, current_point, remaining_sum):
            if dim == 1:
                ref_points.append(current_point + [remaining_sum])
            else:
                for i in range(int(remaining_sum / H) + 1):
                    generate_recursive(dim - 1, current_point + [i * H], remaining_sum - i * H)

        generate_recursive(2, [], 1)  # 2 objectives
        return np.array(ref_points)

    def _associate_with_reference_points(self, population: List[Individual]):
        """把每个解关联到最近的参考点"""
        for ind in population:
            # 计算到每个参考点的欧氏距离
            objectives = np.array(ind.objectives)
            distances = []
            for ref_point in self.reference_points:
                d = np.linalg.norm(objectives - ref_point)
                distances.append(d)
            ind.distance = min(distances)

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """快速非支配排序"""
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []

            for q in population:
                if all(p.objectives[i] <= q.objectives[i] for i in range(2)) and \
                        any(p.objectives[i] < q.objectives[i] for i in range(2)):
                    p.dominated_solutions.append(q)
                elif all(q.objectives[i] <= p.objectives[i] for i in range(2)) and \
                        any(q.objectives[i] < p.objectives[i] for i in range(2)):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:  # 当fronts不为空
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _tournament_selection(self, population: List[Individual], k: int = 2) -> Individual:
        candidates = random.sample(population, k)
        best = min(candidates, key=lambda x: (x.rank, x.distance))
        return best

    def _crossover(self, parent1: State, parent2: State, param=0.1) -> State:
        child = State(**{k: v.copy() if isinstance(v, np.ndarray) else v
                         for k, v in parent1.__dict__.items()})

        for attr in ['P_G1', 'P_G2', 'P_G3', 'theta', 'P_SC', 'P_SD']:
            if random.random() < self.config.crossover_prob:
                beta = random.random()
                child.__dict__[attr] = beta * parent1.__dict__[attr] + \
                                       (1 - beta) * parent2.__dict__[attr]

        for attr in ['u_G1', 'u_G2', 'u_G3', 'u_SC', 'u_SD']:
            if random.random() < self.config.crossover_prob:
                mask = np.random.random(len(parent1.__dict__[attr])) < param
                child.__dict__[attr] = np.where(mask,
                                                parent1.__dict__[attr],
                                                parent2.__dict__[attr])
        return child

    def _mutation(self, state: State, param=0.1) -> State:
        mutated = State(**{k: v.copy() if isinstance(v, np.ndarray) else v
                           for k, v in state.__dict__.items()})

        if random.random() < self.config.mutation_prob:
            for attr in ['P_G1', 'P_G2', 'P_G3', 'theta', 'P_SC', 'P_SD']:
                noise = np.random.normal(0, param, len(state.__dict__[attr]))
                mutated.__dict__[attr] += noise

        if random.random() < self.config.mutation_prob:
            for attr in ['u_G1', 'u_G2', 'u_G3', 'u_SC', 'u_SD']:
                flip_mask = np.random.random(len(state.__dict__[attr])) < param
                mutated.__dict__[attr] = np.where(flip_mask,
                                                  1 - state.__dict__[attr],
                                                  state.__dict__[attr])

        return mutated

    def optimize(self) -> Tuple[List[Individual], List[float], List[float]]:
        """Main optimization loop of NSGA-III."""
        # 1. Initialization
        population = []
        for _ in range(self.config.pop_size):
            state = self.env.create_random_state()
            state = self.env.repair_solution(state)
            objectives = self.env.calculate_objectives(state)
            population.append(Individual(state=state, objectives=objectives))

        all_costs = []
        all_emissions = []

        # 2. Main loop
        for generation in range(self.config.n_generations):
            offspring = []
            while len(offspring) < self.config.pop_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover and Mutation
                child_state = self._crossover(parent1.state, parent2.state, param=0.1)
                child_state = self._mutation(child_state,param=0.1)

                # Repair and evaluate
                child_state = self.env.repair_solution(child_state)
                objectives = self.env.calculate_objectives(child_state)

                offspring.append(Individual(state=child_state, objectives=objectives))

            # Combine parent and offspring populations
            combined = population + offspring

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined)

            # Environmental selection (select individuals for the next generation)
            new_population = []
            front_idx = 0
            while len(new_population) + len(fronts[front_idx]) <= self.config.pop_size:
                new_population.extend(fronts[front_idx])
                front_idx += 1
                if front_idx >= len(fronts):
                    break

            # If the population size is not filled, add individuals from the last front
            if len(new_population) < self.config.pop_size and front_idx < len(fronts):
                last_front = fronts[front_idx]
                self._associate_with_reference_points(last_front)
                last_front.sort(key=lambda x: x.distance)  # Sort by distance to reference points
                new_population.extend(last_front[:self.config.pop_size - len(new_population)])

            population = new_population

            # Store min cost and emission for plotting
            min_cost = min(ind.objectives[0] for ind in population)
            min_emission = min(ind.objectives[1] for ind in population)
            all_costs.append(min_cost)
            all_emissions.append(min_emission)

            print(f"Generation {generation}: Min Cost = {min_cost:.2f}, Min Emission = {min_emission:.2f}")

        return population, all_costs, all_emissions