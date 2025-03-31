# main.py
from config import get_default_configs
from environment import PowerSystemEnv
from nsga3 import NSGAIII
#from nsga3_rl import NSGAIII_RL as NSGAIII

from utils import plot_pareto_front, plot_result, cost_result, calculate_and_save_costs


def main():
    # Load configurations
    system_config, generator_config, storage_config, nsga_config, load_forecast, wind_forecast, pv_forecast = get_default_configs()

    # Create environment
    env = PowerSystemEnv(system_config, generator_config, storage_config,
                        load_forecast, wind_forecast, pv_forecast)

    # Create and run NSGA-III
    nsga = NSGAIII(env, nsga_config)
    population, _, _ = nsga.optimize()

    # Plot results
    plot_pareto_front(population, save_path='pareto_front_nsga3.png')
    plot_result(population[50].state, env, save_path='operation_profiles_nsga3.png')

    # Calculate and save costs
    cost_result(population)
    calculate_and_save_costs(population, 'result_nsga3.csv')


if __name__ == "__main__":
    main()