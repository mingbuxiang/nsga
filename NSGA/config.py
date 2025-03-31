# config.py
from dataclasses import dataclass
import pandas as pd
from typing import Tuple
import numpy as np

@dataclass
class SystemConfig:
    N_T: int = 24
    sigma_e: float = 100
    E_S: float = 1
    lambda_c: float = 0.07
    eta_b: float = 0.9
    K_S: float = 4
    sigma_s: float = 0.27
    P_coal: float = 870
    rho: float = 400
    m_Wcut: float = 40
    m_PVcut: float = 40
    m_Lcut: float = 40

@dataclass
class GeneratorConfig:
    max_output: float = 600
    min_output: float = 325
    ramp_rate: float = 300
    a: float = 0.000013
    b: float = 0.2322
    c: float = 16
    switch_cost: float = 5
    T_on_min: int = 6
    T_off_min: int = 6

@dataclass
class StorageConfig:
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    max_capacity: float = 200
    min_capacity: float = 50
    max_charge: float = 50
    min_charge: float = 50
    max_discharge: float = 50
    min_discharge: float = 50
    initial_capacity: float = 80
    charge_cost: float = 20

@dataclass
class NSGAConfig:
    pop_size: int = 100
    n_generations: int = 100
    mutation_prob: float = 0.8
    crossover_prob: float = 0.8
    n_divisions: int = 24


def load_forecasts(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads forecasts from a CSV file."""
    forecasts_df = pd.read_csv(filename, index_col='Hour')
    load_forecast = forecasts_df['Load Forecast'].values
    wind_forecast = forecasts_df['Wind Forecast'].values
    pv_forecast = forecasts_df['PV Forecast'].values
    return load_forecast, wind_forecast, pv_forecast


def get_default_configs(data_filepath='data/forecasts.csv'):
    """Creates and returns default configuration instances."""
    system_config = SystemConfig()
    generator_config = GeneratorConfig()
    storage_config = StorageConfig()
    nsga_config = NSGAConfig()
    load_forecast, wind_forecast, pv_forecast = load_forecasts(data_filepath)
    return system_config, generator_config, storage_config, nsga_config, load_forecast, wind_forecast, pv_forecast