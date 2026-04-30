import skrf as rf
import numpy as np


def load_s16p(file_path: str) -> rf.Network:
    network = rf.Network(file_path)
    if network.number_of_ports != 16:
        raise ValueError(f"需要 16-port S16P 檔案，但這個檔案是 {network.number_of_ports}-port")
    return network


def load_s4p(file_path: str) -> rf.Network:
    network = rf.Network(file_path)
    if network.number_of_ports != 4:
        raise ValueError(f"需要 4-port S4P 檔案，但這個檔案是 {network.number_of_ports}-port")
    return network


def get_frequency_ghz(network: rf.Network) -> np.ndarray:
    return network.f / 1e9


def get_smatrix(network: rf.Network) -> np.ndarray:
    """回傳 S-parameter 矩陣，shape: [freq, 4, 4]"""
    return network.s
