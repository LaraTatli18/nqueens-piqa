from nqueens import GenerateBoard, ClassicalEnergy, QuantumEnergy, QuantumSwapMove, RunPIQA
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib
matplotlib.use('TkAgg')

def run_single_piqa(args):
    N, P, MC_moves_per_step, annealing_steps, G_0, G_f, A, B, C = args
    _, final_energy, _ = RunPIQA(N, P, 1.6/P, MC_moves_per_step, annealing_steps, G_0, G_f, A, B, C)
    return final_energy

def compare_trotter_steps(
        problem_name: str,
        opt_energy: float,
        r: int = 40,
        save_data: bool = True):

    plt.rcParams.update({'figure.figsize': (8, 6)})

    # Parameters
    trotter_numbers = [24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    N = 8
    A, B, C = 2, 2, 2
    annealing_steps = 100
    MC_moves_per_step = 8
    G_0, G_f = 60, 0.001

    mean_energy = []
    std_energy = []

    # Create a multiprocessing pool
    pool = mp.Pool(processes=8)

    for P in trotter_numbers:
        print(f"Starting runs for P = {P}")

        # Prepare argument tuples for each run
        args_list = [(N, P, MC_moves_per_step, annealing_steps, G_0, G_f, A, B, C)] * r

        # Run in parallel
        ground_energies = pool.map(run_single_piqa, args_list)

        mean_energy.append(np.mean(ground_energies))
        std_energy.append(np.std(ground_energies) / np.sqrt(r))

    pool.close()
    pool.join()

    plt.errorbar(trotter_numbers, mean_energy, yerr=std_energy, fmt='s-', capsize=3)
    plt.xlabel("Trotter Number $P$")
    plt.ylabel("Residual Energy")
    plt.tight_layout()
    plt.show()

    # Save Data
    if save_data:
        data_dict = {
            "trotter_numbers": trotter_numbers,
            "mean_energy": mean_energy,
            "stds_energy": std_energy,
            "opt_energy": opt_energy
        }
        np.savez(f"{problem_name}_trotter_numbers_data.npz", **data_dict)
        print(f"Data saved to {problem_name}_trotter_numbers_data.npz")

if __name__ == "__main__":
    compare_trotter_steps(
        problem_name="N_queens_8",
        opt_energy=0.001,
        r=40,
        save_data=True
    )