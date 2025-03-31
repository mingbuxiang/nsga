# environment.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from config import SystemConfig, GeneratorConfig, StorageConfig

@dataclass
class State:
    P_G1: np.ndarray
    P_G2: np.ndarray
    P_G3: np.ndarray
    u_G1: np.ndarray
    u_G2: np.ndarray
    u_G3: np.ndarray
    theta: np.ndarray
    P_SC: np.ndarray
    P_SD: np.ndarray
    u_SC: np.ndarray
    u_SD: np.ndarray
    E_ES: np.ndarray
    P_Wcut: np.ndarray
    P_PVcut: np.ndarray
    P_Lcut: np.ndarray


class PowerSystemEnv:
    def __init__(self,
                 system_config: SystemConfig,
                 generator_config: GeneratorConfig,
                 storage_config: StorageConfig,
                 load_forecast: np.ndarray,
                 wind_forecast: np.ndarray,
                 pv_forecast: np.ndarray):
        self.sys_config = system_config
        self.gen_config = generator_config
        self.storage_config = storage_config
        self.load_forecast = load_forecast
        self.wind_forecast = wind_forecast
        self.pv_forecast = pv_forecast
        self.N_T = system_config.N_T

    def create_random_state(self) -> State:
        """
        优化随机初始解的生成
        """
        # 创建一个交替的01序列
        alternating_sequence = np.array([i % 2 for i in range(self.N_T)])

        u_G1 = alternating_sequence.copy()
        u_G2 = alternating_sequence.copy()
        u_G3 = alternating_sequence.copy()
        base_load = np.mean([self.gen_config.min_output, self.gen_config.max_output])
        variation = (self.gen_config.max_output - self.gen_config.min_output) * 0.3
        P_G1 = base_load + variation * np.sin(np.linspace(0, 2 * np.pi, self.N_T))
        P_G1 = np.clip(P_G1, self.gen_config.min_output, self.gen_config.max_output)
        P_G1 = P_G1 * u_G1
        P_G2 = P_G1 * u_G2
        P_G3 = P_G1 * u_G3
        theta = np.random.uniform(0, 1, self.N_T)
        u_SC = np.random.randint(0, 2, self.N_T)
        u_SD = 1 - u_SC

        P_SC = np.random.uniform(
            self.storage_config.min_charge,
            self.storage_config.max_charge,
            self.N_T
        ) * u_SC

        P_SD = np.random.uniform(
            self.storage_config.min_discharge,
            self.storage_config.max_discharge,
            self.N_T
        ) * u_SD

        E_ES = np.zeros(self.N_T + 1)
        E_ES[0] = self.storage_config.initial_capacity

        P_Wcut = np.zeros(self.N_T)
        P_PVcut = np.zeros(self.N_T)
        P_Lcut = np.zeros(self.N_T)

        return State(P_G1, P_G2, P_G3, u_G1, u_G2, u_G3, theta, P_SC, P_SD, u_SC, u_SD, E_ES,
                     P_Wcut, P_PVcut, P_Lcut)

    def adjust_power_balance(self, state: State) -> State:
        """
        调整解以满足功率平衡约束，主要通过调整弃风、弃光和失负荷来实现
        """
        for t in range(self.N_T):
            # 计算功率不平衡量
            power_mismatch = (state.P_G1[t] + state.P_G2[t] + state.P_G3[t] +
                              self.wind_forecast[t] - state.P_Wcut[t] +
                              self.pv_forecast[t] - state.P_PVcut[t] -
                              state.P_SC[t] + state.P_SD[t] -
                              self.load_forecast[t] + state.P_Lcut[t] -
                              state.P_G1[t] * self.sys_config.E_S * state.theta[
                                  t] * self.sys_config.eta_b * self.sys_config.sigma_s +
                              state.P_G2[t] * self.sys_config.E_S * state.theta[
                                  t] * self.sys_config.eta_b * self.sys_config.sigma_s +
                              state.P_G3[t] * self.sys_config.E_S * state.theta[
                                  t] * self.sys_config.eta_b * self.sys_config.sigma_s)

            if power_mismatch > 0:  # 发电过剩
                extra_wind_cut = min(power_mismatch,
                                     self.wind_forecast[t] - state.P_Wcut[t])
                state.P_Wcut[t] += extra_wind_cut
                power_mismatch -= extra_wind_cut

                if power_mismatch > 0:
                    extra_pv_cut = min(power_mismatch,
                                       self.pv_forecast[t] - state.P_PVcut[t])
                    state.P_PVcut[t] += extra_pv_cut

            elif power_mismatch < 0:  # 发电不足

                state.P_Lcut[t] = min(-power_mismatch, self.load_forecast[t])

        return state

    def calculate_objectives(self, state: State) -> Tuple[float, float]:
        """
        计算给定状态的两个目标函数值：运行成本和碳排放量
        
        Returns:
            Tuple[float, float]: (运行成本, 碳排放量)
        """
        # 计算运行成本
        gen_cost1 = sum((self.gen_config.a * state.P_G1 ** 2 +
                         self.gen_config.b * state.P_G1 +
                         self.gen_config.c) * self.sys_config.P_coal)

        gen_cost2 = sum((self.gen_config.a * state.P_G2 ** 2 +
                         self.gen_config.b * state.P_G2 +
                         self.gen_config.c) * self.sys_config.P_coal)

        gen_cost3 = sum((self.gen_config.a * state.P_G3 ** 2 +
                         self.gen_config.b * state.P_G3 +
                         self.gen_config.c) * self.sys_config.P_coal)

        gen_cost = gen_cost1 + gen_cost2 + gen_cost3

        switch_cost1 = self.gen_config.switch_cost * sum(abs(state.u_G1[1:] - state.u_G1[:-1]))

        switch_cost2 = self.gen_config.switch_cost * sum(abs(state.u_G2[1:] - state.u_G2[:-1]))

        switch_cost3 = self.gen_config.switch_cost * sum(abs(state.u_G3[1:] - state.u_G3[:-1]))

        switch_cost = switch_cost1 + switch_cost2 + switch_cost3

        storage_cost = self.storage_config.charge_cost * sum(state.P_SC + state.P_SD)

        penalty_cost = (self.sys_config.m_Wcut * sum(state.P_Wcut) +
                        self.sys_config.m_PVcut * sum(state.P_PVcut) +
                        self.sys_config.m_Lcut * sum(state.P_Lcut))

        carbon_trading_cost1 = self.sys_config.sigma_e * sum(
            self.sys_config.E_S * state.P_G1 - self.sys_config.lambda_c * state.P_G1)

        carbon_trading_cost2 = self.sys_config.sigma_e * sum(
            self.sys_config.E_S * state.P_G2 - self.sys_config.lambda_c * state.P_G2)

        carbon_trading_cost3 = self.sys_config.sigma_e * sum(
            self.sys_config.E_S * state.P_G3 - self.sys_config.lambda_c * state.P_G3)

        carbon_trading_cost = carbon_trading_cost1 + carbon_trading_cost2 + carbon_trading_cost3

        carbon_capture1 = self.sys_config.sigma_s * self.sys_config.rho * sum(
            state.P_G1 * self.sys_config.E_S * state.theta * self.sys_config.eta_b)

        carbon_capture2 = self.sys_config.sigma_s * self.sys_config.rho * sum(
            state.P_G2 * self.sys_config.E_S * state.theta * self.sys_config.eta_b)

        carbon_capture3 = self.sys_config.sigma_s * self.sys_config.rho * sum(
            state.P_G3 * self.sys_config.E_S * state.theta * self.sys_config.eta_b)

        carbon_capture = carbon_capture1 + carbon_capture2 + carbon_capture3

        carbon_fen1 = self.sys_config.sigma_e * self.sys_config.K_S * sum(
            state.P_G1 * self.sys_config.E_S * state.theta * self.sys_config.eta_b)

        carbon_fen2 = self.sys_config.sigma_e * self.sys_config.K_S * sum(
            state.P_G2 * self.sys_config.E_S * state.theta * self.sys_config.eta_b)

        carbon_fen3 = self.sys_config.sigma_e * self.sys_config.K_S * sum(
            state.P_G3 * self.sys_config.E_S * state.theta * self.sys_config.eta_b)

        carbon_fen = carbon_fen1 + carbon_fen2 + carbon_fen3

        total_cost = (gen_cost + switch_cost + storage_cost + penalty_cost + carbon_capture
        + carbon_trading_cost + carbon_fen)

        # 计算碳排放量
        carbon_emission1 = sum(self.sys_config.E_S * state.P_G1 -
                               state.P_G1 * self.sys_config.E_S * state.theta * self.sys_config.eta_b) 

        carbon_emission2 = sum(self.sys_config.E_S * state.P_G2 -
                               state.P_G2 * self.sys_config.E_S * state.theta * self.sys_config.eta_b) 

        carbon_emission3 = sum(self.sys_config.E_S * state.P_G3 -
                               state.P_G3 * self.sys_config.E_S * state.theta * self.sys_config.eta_b) 

        carbon_emission = (carbon_emission1 + carbon_emission2 + carbon_emission3)

        return total_cost, carbon_emission

    def repair_solution(self, state: State) -> State:
        """
        修复不可行解

        """
        new_state = State(**{k: v.copy() if isinstance(v, np.ndarray) else v
                             for k, v in state.__dict__.items()})

        for t in range(self.N_T - self.gen_config.T_on_min + 1):
            if (new_state.u_G1[t] - (0 if t == 0 else new_state.u_G1[t - 1])) == 1:
                if sum(new_state.u_G1[t:t + self.gen_config.T_on_min]) < self.gen_config.T_on_min:
                    new_state.u_G1[t:t + self.gen_config.T_on_min] = 1

        for t in range(self.N_T - self.gen_config.T_on_min + 1):
            if (new_state.u_G2[t] - (0 if t == 0 else new_state.u_G2[t - 1])) == 1:
                if sum(new_state.u_G2[t:t + self.gen_config.T_on_min]) < self.gen_config.T_on_min:
                    new_state.u_G2[t:t + self.gen_config.T_on_min] = 1

        for t in range(self.N_T - self.gen_config.T_on_min + 1):
            if (new_state.u_G3[t] - (0 if t == 0 else new_state.u_G3[t - 1])) == 1:
                if sum(new_state.u_G3[t:t + self.gen_config.T_on_min]) < self.gen_config.T_on_min:
                    new_state.u_G3[t:t + self.gen_config.T_on_min] = 1

        for t in range(self.N_T - self.gen_config.T_off_min + 1):
            if ((0 if t == 0 else new_state.u_G1[t - 1]) - new_state.u_G1[t]) == 1:
                if sum(1 - new_state.u_G1[t:t + self.gen_config.T_off_min]) < self.gen_config.T_off_min:
                    new_state.u_G1[t:t + self.gen_config.T_off_min] = 0

        for t in range(self.N_T - self.gen_config.T_off_min + 1):
            if ((0 if t == 0 else new_state.u_G2[t - 1]) - new_state.u_G2[t]) == 1:
                if sum(1 - new_state.u_G2[t:t + self.gen_config.T_off_min]) < self.gen_config.T_off_min:
                    new_state.u_G2[t:t + self.gen_config.T_off_min] = 0

        for t in range(self.N_T - self.gen_config.T_off_min + 1):
            if ((0 if t == 0 else new_state.u_G3[t - 1]) - new_state.u_G3[t]) == 1:
                if sum(1 - new_state.u_G3[t:t + self.gen_config.T_off_min]) < self.gen_config.T_off_min:
                    new_state.u_G3[t:t + self.gen_config.T_off_min] = 0

        for t in range(self.N_T):
            if new_state.u_G1[t] > 0:
                new_state.P_G1[t] = np.clip(
                    new_state.P_G1[t],
                    self.gen_config.min_output,
                    self.gen_config.max_output
                )
            else:
                new_state.P_G1[t] = 0

        for t in range(self.N_T):
            if new_state.u_G2[t] > 0:
                new_state.P_G2[t] = np.clip(
                    new_state.P_G2[t],
                    self.gen_config.min_output,
                    self.gen_config.max_output
                )
            else:
                new_state.P_G2[t] = 0

        for t in range(self.N_T):
            if new_state.u_G3[t] > 0:
                new_state.P_G3[t] = np.clip(
                    new_state.P_G3[t],
                    self.gen_config.min_output,
                    self.gen_config.max_output
                )
            else:
                new_state.P_G3[t] = 0

        new_state.theta = np.clip(new_state.theta, 0, 1)
        
        new_state.u_SC = new_state.u_SC.astype(bool)
        new_state.u_SD = new_state.u_SD.astype(bool)
        new_state.u_SD = new_state.u_SD & (~new_state.u_SC) 

        for t in range(self.N_T):
            new_state.P_SC[t] = np.clip(
                new_state.P_SC[t],
                self.storage_config.min_charge * new_state.u_SC[t],
                self.storage_config.max_charge * new_state.u_SC[t]
            )
            new_state.P_SD[t] = np.clip(
                new_state.P_SD[t],
                self.storage_config.min_discharge * new_state.u_SD[t],
                self.storage_config.max_discharge * new_state.u_SD[t]
            )
            new_state.E_ES[t + 1] = (new_state.E_ES[t] +
                                     self.storage_config.charge_efficiency * new_state.P_SC[t] -
                                     new_state.P_SD[t] / self.storage_config.discharge_efficiency)
            new_state.E_ES[t + 1] = np.clip(
                new_state.E_ES[t + 1],
                self.storage_config.min_capacity,
                self.storage_config.max_capacity
            )

            new_state.E_ES[t + 1] = max(new_state.E_ES[t + 1], 0)

        new_state.E_ES[0] = new_state.E_ES[-1]
        new_state = self.adjust_power_balance(new_state)

        return new_state