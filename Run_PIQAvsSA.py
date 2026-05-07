from nqueens import GenerateBoard, PrintReadableBoard, ClassicalEnergy, SwapMove, QuantumEnergy, QuantumSwapMove, RunSimulatedAnnealing, RunPIQA
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import os

def compare_annealing_steps(
        problem_name: str,
        opt_energy: float,
        r: int = 5,
        save_data: bool = True
):
    """
    Compares SA and QA performance for a given 'problem' at various
    numbers of annealing steps. Repeats each run 'r' times, records mean & std.
    """

    plt.rcParams.update({
        'figure.figsize': (8, 6)
        # 'font.size': 8,
        # 'axes.labelsize': 10,
        # 'text.usetex': False
    })

    # Sets of Monte Carlo steps to loop over
    annealing_steps_sa = [10, 50, 75, 100, 150, 300, 500, 1000, 2000, 5000] # exclude 10,000
    annealing_steps_qa = [10, 50, 75, 100, 150, 300, 500]

    # Problem size (NxN chessboard)
    N = 8
    A, B, C = 2, 2, 2
    MC_moves_per_step = 8

    # Temperature schedule bounds (for SA)
    T_0, T_f = 100, 0.001

    # Quantum schedule bounds (for QA)
    G_0, G_f = 40, 0.001

    # Other parameters (for QA)
    P = 10 # Trotter number
    T = 0.1  # "Classical" temperature used in acceptance


    means_sa = []
    stds_sa = []
    means_qa = []
    stds_qa = []


    # -- Simulated Annealing --
    for num in annealing_steps_sa:
        print(f"[SA] annealing_steps = {num}")

        energies_sa = []
        for _ in range(r):
            _, _, final_energy = RunSimulatedAnnealing(N, MC_moves_per_step, num, T_0, T_f, A, B, C)
            energies_sa.append(final_energy)

        means_sa.append(np.mean(energies_sa))
        stds_sa.append(np.std(energies_sa) / np.sqrt(r))

    # -- Quantum Annealing --
    for num in annealing_steps_qa:
        print(f"[QA] annealing_steps = {num}")

        energies_qa = []
        for _ in range(r):
            _, final_energy, _ = RunPIQA(N, P, 1.6/P, MC_moves_per_step, num, G_0, G_f, A, B, C)
            energies_qa.append(final_energy)

        means_qa.append(np.mean(energies_qa))
        stds_qa.append(np.std(energies_qa) / np.sqrt(r))


    plt.figure()
    plt.errorbar(annealing_steps_sa, means_sa, yerr=stds_sa, fmt='o-', label="SA", capsize=3)
    plt.errorbar(annealing_steps_qa, means_qa, yerr=stds_qa, fmt='s-', label="QA", capsize=3)
    plt.xlabel("Number of Monte Carlo Steps")
    plt.ylabel("Residual Energy")
    plt.title(f"Comparison of SA vs. QA for {problem_name}")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Save Data
    if save_data:
        data_dict = {
            "annealing_steps_sa": annealing_steps_sa,
            "means_sa": means_sa,
            "stds_sa": stds_sa,
            "annealing_steps_qa": annealing_steps_qa,
            "means_qa": means_qa,
            "stds_qa": stds_qa,
            "opt_energy": opt_energy
        }
        np.savez(f"{problem_name}_annealing_steps_data_final.npz", **data_dict)
        print(f"Data saved to {problem_name}_annealing_steps_data_final.npz")



if __name__ == "__main__":
    compare_annealing_steps(
        problem_name="N_queens_8",
        opt_energy=0.001,
        r=5,
        save_data=True
    )