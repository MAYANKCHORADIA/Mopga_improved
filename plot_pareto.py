import sys
import numpy as np
import matplotlib.pyplot as plt

# Import benchmark functions and problem loaders from the experiment master
from experiment_master import load_problem, run_nsga2_benchmark, run_mopga_benchmark, PROBLEM_NAMES

def plot_pareto(problem_name="zdt1", show=True):
    print(f"\n{'='*50}")
    print(f"[*] Processing problem {problem_name.upper()}...")
    print(f"{'='*50}")
    problem, pf = load_problem(problem_name)
    
    if pf is None:
        print(f"[!] Could not load True Pareto Front for {problem_name}.")
        return

    print("[*] Running NSGA-II ...")
    nsga2_F = run_nsga2_benchmark(problem)
    
    print("[*] Running Improved MOPGA ...")
    mopga_F = run_mopga_benchmark(problem)

    n_obj = problem.n_obj
    
    print(f"[*] Rendering plot for {problem_name.upper()}...")
    
    if n_obj == 2:
        plt.figure(figsize=(10, 7))
        # True PF
        sort_idx = np.argsort(pf[:, 0])
        plt.plot(pf[sort_idx, 0], pf[sort_idx, 1], label="True Pareto Front", color='black', linewidth=2, zorder=1)
        
        # NSGA-II
        plt.scatter(nsga2_F[:, 0], nsga2_F[:, 1], label="NSGA-II", color='#1f77b4', alpha=0.6, marker='s', s=40, zorder=2)
        
        # MOPGA Improved
        plt.scatter(mopga_F[:, 0], mopga_F[:, 1], label="MOPGA Improved", color='#d62728', alpha=0.9, marker='x', s=60, zorder=3)
        
        plt.title(f"Pareto Front Comparison: {problem_name.upper()}")
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        
    elif n_obj == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # True PF
        ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], label="True Pareto Front", color='black', alpha=0.15, s=15, zorder=1)
        
        # NSGA-II
        ax.scatter(nsga2_F[:, 0], nsga2_F[:, 1], nsga2_F[:, 2], label="NSGA-II", color='#1f77b4', alpha=0.6, marker='s', s=40, zorder=2)
        
        # MOPGA Improved
        ax.scatter(mopga_F[:, 0], mopga_F[:, 1], mopga_F[:, 2], label="MOPGA Improved", color='#d62728', alpha=0.9, marker='x', s=70, zorder=3)
        
        ax.set_title(f"Pareto Front Comparison: {problem_name.upper()}")
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        plt.legend(frameon=True, shadow=True)
        # Tweak viewing angle for better 3D visualization
        ax.view_init(elev=30, azim=45)
    else:
        print(f"[!] Plotting for {n_obj} objectives is not implemented.")
        return
        
    out_file = f"{problem_name}_pareto_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"[+] Saved plot to {out_file}\n")
    
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "all":
            print(f"[*] Starting batch generation for all {len(PROBLEM_NAMES)} problems...")
            for p in PROBLEM_NAMES:
                plot_pareto(p, show=False)
            print("[+] BATCH GENERATION COMPLETE!")
        else:
            plot_pareto(sys.argv[1].lower(), show=True)
    else:
        plot_pareto("uf8", show=True)
