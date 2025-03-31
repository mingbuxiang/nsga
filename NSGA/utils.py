# utils.py

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from environment import State, PowerSystemEnv
from nsga3 import Individual, NSGAIII  # 从 nsga3 模块导入 Individual
from typing import List, Tuple
from config import get_default_configs
system_config, generator_config, storage_config, nsga_config, load_forecast, wind_forecast, pv_forecast = get_default_configs()

def cost_result(population: List[Individual]):
    # 选择帕累托前沿上的一个解
    individual = population[50]

    gen_cost1 = sum((generator_config.a * individual.state.P_G1 ** 2 +
                         generator_config.b * individual.state.P_G1 +
                         generator_config.c) * system_config.P_coal)

    gen_cost2 = sum((generator_config.a * individual.state.P_G2 ** 2 +
                        generator_config.b * individual.state.P_G2 +
                        generator_config.c) * system_config.P_coal)

    gen_cost3 = sum((generator_config.a * individual.state.P_G3 ** 2 +
                        generator_config.b * individual.state.P_G3 +
                        generator_config.c) * system_config.P_coal)

    gen_cost = gen_cost1 + gen_cost2 + gen_cost3

    switch_cost1 = generator_config.switch_cost * sum(abs(individual.state.u_G1[1:] - individual.state.u_G1[:-1]))

    switch_cost2 = generator_config.switch_cost * sum(abs(individual.state.u_G2[1:] - individual.state.u_G2[:-1]))

    switch_cost3 = generator_config.switch_cost * sum(abs(individual.state.u_G3[1:] - individual.state.u_G3[:-1]))

    switch_cost = switch_cost1 + switch_cost2 + switch_cost3

    storage_cost = storage_config.charge_cost * sum(individual.state.P_SC + individual.state.P_SD)

    penalty_cost = (system_config.m_Wcut * sum(individual.state.P_Wcut) +
                    system_config.m_PVcut * sum(individual.state.P_PVcut) +
                    system_config.m_Lcut * sum(individual.state.P_Lcut))

    carbon_trading_cost1 = system_config.sigma_e * sum(
        system_config.E_S * individual.state.P_G1 - system_config.lambda_c * individual.state.P_G1)

    carbon_trading_cost2 = system_config.sigma_e * sum(
        system_config.E_S * individual.state.P_G2 - system_config.lambda_c * individual.state.P_G2)

    carbon_trading_cost3 = system_config.sigma_e * sum(
        system_config.E_S * individual.state.P_G3 - system_config.lambda_c * individual.state.P_G3)

    carbon_trading_cost = carbon_trading_cost1 + carbon_trading_cost2 + carbon_trading_cost3

    carbon_capture1 = system_config.sigma_s * system_config.rho * sum(
        individual.state.P_G1 * system_config.E_S * individual.state.theta * system_config.eta_b)

    carbon_capture2 = system_config.sigma_s * system_config.rho * sum(
        individual.state.P_G2 * system_config.E_S * individual.state.theta * system_config.eta_b)

    carbon_capture3 = system_config.sigma_s * system_config.rho * sum(
        individual.state.P_G3 * system_config.E_S * individual.state.theta * system_config.eta_b)

    carbon_capture = carbon_capture1 + carbon_capture2 + carbon_capture3

    carbon_fen1 = system_config.sigma_e * system_config.K_S * sum(
        individual.state.P_G1 * system_config.E_S * individual.state.theta * system_config.eta_b)

    carbon_fen2 = system_config.sigma_e * system_config.K_S * sum(
        individual.state.P_G2 * system_config.E_S * individual.state.theta * system_config.eta_b)

    carbon_fen3 = system_config.sigma_e * system_config.K_S * sum(
        individual.state.P_G3 * system_config.E_S * individual.state.theta * system_config.eta_b)

    carbon_fen = carbon_fen1 + carbon_fen2 + carbon_fen3
    total_cost = (gen_cost + switch_cost + storage_cost + penalty_cost + carbon_capture 
              + carbon_trading_cost + carbon_fen)


    print(f"发电成本: {gen_cost}")
    print(f"开关成本: {switch_cost}")
    print(f"充放电成本: {storage_cost}")
    print(f"弃能源成本: {penalty_cost}")
    print(f"碳捕捉成本: {carbon_capture}")
    print(f"碳交易成本: {carbon_trading_cost}")
    print(f"碳封存成本: {carbon_fen}")
    print(f"总成本: {total_cost}")



def calculate_and_save_costs(population: List[Individual], output_file='result.csv'):

    def calculate_individual_costs(individual: Individual):
        """Calculates the costs for a single individual."""

        # --- 发电成本 ---
        gen_cost1 = (generator_config.a * individual.state.P_G1 ** 2 +
                     generator_config.b * individual.state.P_G1 +
                     generator_config.c) * system_config.P_coal
        gen_cost2 = (generator_config.a * individual.state.P_G2 ** 2 +
                     generator_config.b * individual.state.P_G2 +
                     generator_config.c) * system_config.P_coal
        gen_cost3 = (generator_config.a * individual.state.P_G3 ** 2 +
                     generator_config.b * individual.state.P_G3 +
                     generator_config.c) * system_config.P_coal
        gen_cost = sum(gen_cost1) + sum(gen_cost2) + sum(gen_cost3)

        # --- 开关成本 ---
        switch_cost1 = generator_config.switch_cost * \
            sum(abs(individual.state.u_G1[1:] - individual.state.u_G1[:-1]))
        switch_cost2 = generator_config.switch_cost * \
            sum(abs(individual.state.u_G2[1:] - individual.state.u_G2[:-1]))
        switch_cost3 = generator_config.switch_cost * \
            sum(abs(individual.state.u_G3[1:] - individual.state.u_G3[:-1]))
        switch_cost = switch_cost1 + switch_cost2 + switch_cost3

        # --- 充放电成本 ---
        storage_cost = storage_config.charge_cost * \
            sum(individual.state.P_SC + individual.state.P_SD)

        # --- 弃能源成本 ---
        penalty_cost = (system_config.m_Wcut * sum(individual.state.P_Wcut) +
                        system_config.m_PVcut * sum(individual.state.P_PVcut) +
                        system_config.m_Lcut * sum(individual.state.P_Lcut))

        # --- 碳交易成本 ---
        carbon_trading_cost = sum(system_config.sigma_e * (
            system_config.E_S * individual.state.P_G1 - system_config.lambda_c * individual.state.P_G1 +
            system_config.E_S * individual.state.P_G2 - system_config.lambda_c * individual.state.P_G2 +
            system_config.E_S * individual.state.P_G3 - system_config.lambda_c * individual.state.P_G3
        ))

        # --- 碳捕捉成本 ---
        carbon_capture = sum(system_config.sigma_s * system_config.rho * (
            individual.state.P_G1 * system_config.E_S * individual.state.theta * system_config.eta_b +
            individual.state.P_G2 * system_config.E_S * individual.state.theta * system_config.eta_b +
            individual.state.P_G3 * system_config.E_S * individual.state.theta * system_config.eta_b
        ))

        # --- 碳封存成本 ---
        carbon_fen = sum(system_config.sigma_e * system_config.K_S * (
            individual.state.P_G1 * system_config.E_S * individual.state.theta * system_config.eta_b +
            individual.state.P_G2 * system_config.E_S * individual.state.theta * system_config.eta_b +
            individual.state.P_G3 * system_config.E_S * individual.state.theta * system_config.eta_b
        ))

        # --- 总成本 ---
        total_cost = (gen_cost + switch_cost + storage_cost + penalty_cost +
                      carbon_capture + carbon_trading_cost + carbon_fen)

        # --- 碳排放 ---
        carbon_emission1 = sum(system_config.E_S * individual.state.P_G1 -
                        individual.state.P_G1 * system_config.E_S * individual.state.theta * system_config.eta_b)
        carbon_emission2 = sum(system_config.E_S * individual.state.P_G2 -
                                individual.state.P_G2 * system_config.E_S * individual.state.theta * system_config.eta_b)
        carbon_emission3 = sum(system_config.E_S * individual.state.P_G3 -
                                individual.state.P_G3 * system_config.E_S * individual.state.theta * system_config.eta_b)
        carbon_emission = carbon_emission1 + carbon_emission2 + carbon_emission3

        return {
            "发电成本": gen_cost,
            "开关成本": switch_cost,
            "充放电成本": storage_cost,
            "弃能源成本": penalty_cost,
            "碳捕捉成本": carbon_capture,
            "碳交易成本": carbon_trading_cost,
            "碳封存成本": carbon_fen,
            "总成本": total_cost,
            "碳排放": carbon_emission  # 返回碳排放
        }

    # Calculate costs for all individuals
    costs_list = [calculate_individual_costs(ind) for ind in population]

    df = pd.DataFrame(costs_list)
    emissions = [item['碳排放'] for item in costs_list]
    df['碳排放'] = emissions

    # Save to CSV
    df.to_csv(output_file, index=False)



def filter_data_for_pareto_front(population: List[Individual]) -> Tuple[np.ndarray, np.ndarray]:
    """获取帕累托前沿的数据点（非支配解）"""
    costs = [ind.objectives[0] for ind in population]
    emissions = [ind.objectives[1] for ind in population]
    
    # 使用支配关系筛选帕累托前沿
    pareto_points = []
    for point in zip(costs, emissions):
        if not any(dominates(p, point) for p in zip(costs, emissions) if p != point):
            pareto_points.append(point)
    
    # 按成本排序
    pareto_points = sorted(pareto_points, key=lambda x: x[0])
    return np.array([p[0] for p in pareto_points]), np.array([p[1] for p in pareto_points])

def dominates(p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
    """判断p1是否支配p2"""
    return (p1[0] <= p2[0] and p1[1] <= p2[1]) and (p1[0] < p2[0] or p1[1] < p2[1])


def plot_pareto_front(population: List[Individual],
                      save_path: str = None,
                      show: bool = True) -> None:
    """
    绘制帕累托前沿
    """
    # 获取所有解和帕累托前沿
    all_costs = [ind.objectives[0] for ind in population]
    all_emissions = [ind.objectives[1] for ind in population]
    pareto_costs, pareto_emissions = filter_data_for_pareto_front(population)

    os.makedirs('data', exist_ok=True)
    sorted_pareto = sorted(zip(pareto_costs, pareto_emissions), key=lambda x: x[0])

    # 保存为CSV文件
    df = pd.DataFrame(sorted_pareto, columns=['Operation_Cost', 'Carbon_Emissions'])
    df.to_csv('data/pareto_front_nsga3.csv', index=False)

    # 绘图设置
    plt.figure(figsize=(10, 6))
    plt.scatter(all_costs, all_emissions, c='blue', alpha=0.6, label='All Solutions')
    plt.plot(pareto_costs, pareto_emissions, 'r--', linewidth=2, label='Pareto Front')

    plt.xlabel('Operation Cost (¥)')
    plt.ylabel('Carbon Emissions (ton)')
    plt.title('Pareto Front of Operation Cost vs Carbon Emissions')
    plt.legend()
    plt.grid(True)

    # 保存和显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_result(state: State, 
                env: PowerSystemEnv, 
                save_path: str = None, 
                show: bool = True) -> None:
    """
    绘制解的运行曲线
    """
    time = range(24)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # 发电机组出力
    axes[0].plot(time, state.P_G1, label='Generator 1')
    axes[0].plot(time, state.P_G2, label='Generator 2')
    axes[0].plot(time, state.P_G3, label='Generator 3')
    axes[0].set_xlabel('Time (h)')
    axes[0].set_ylabel('Power Output (MW)')
    axes[0].set_title('Generator Power Output')
    axes[0].legend()
    axes[0].grid(True)
    
    # 可再生能源和负荷
    axes[1].plot(time, env.wind_forecast, label='Wind Power')
    axes[1].plot(time, env.pv_forecast, label='PV Power')
    axes[1].plot(time, env.load_forecast, label='Load')
    axes[1].plot(time, state.P_Wcut, '--', label='Wind Curtailment')
    axes[1].plot(time, state.P_PVcut, '--', label='PV Curtailment')
    axes[1].set_xlabel('Time (h)')
    axes[1].set_ylabel('Power (MW)')
    axes[1].set_title('Renewable Energy and Load')
    axes[1].legend()
    axes[1].grid(True)
    
    # 储能系统
    axes[2].plot(time, state.P_SC, label='Charging Power')
    axes[2].plot(time, state.P_SD, label='Discharging Power')
    axes[2].plot(time, state.E_ES[:-1], label='Energy Storage Level')
    axes[2].set_xlabel('Time (h)')
    axes[2].set_ylabel('Power (MW) / Energy (MWh)')
    axes[2].set_title('Energy Storage System')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
