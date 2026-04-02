"""Worker subprocess: runs one chain length on one GPU."""
import sys, os, pickle, random, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouse_model_python.config import SimulationConfig, SIGMA, L0, SEED
from rouse_model_python.simulation import RouseSimulation
from rouse_model_python.io_utils import write_all_tsvs, ensure_dir

N = int(sys.argv[1])
results_dir = sys.argv[2]
deliverable_dir = sys.argv[3]
params = json.loads(sys.argv[4])

seed_base = SEED
random.seed(seed_base + N)
np.random.seed(seed_base + N)
torch.manual_seed(seed_base + N)
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed(seed_base + N)

device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

cfg = SimulationConfig(
    N=N,
    n_chains=params["n_chains"],
    eq_sweeps=params["eq_sweeps"],
    prod_sweeps=params["prod_sweeps"],
    box_size=params["box"],
    seed=seed_base + N,
    device=device,
    target_phi=0.035,
    sample_interval=max(1, min(20, N // 10)),
    use_batched_mode=torch.cuda.device_count() > 0,
)

print(f"[W-N{N}] device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}", flush=True)

sim = RouseSimulation(cfg)
results = sim.run()

write_all_tsvs(results, deliverable_dir, N, 0.035)

serializable = {
    "N": results["N"],
    "n_chains": results["n_chains"],
    "final_R2": results["final_R2"].cpu().numpy(),
    "final_Rg2": results["final_Rg2"].cpu().numpy(),
    "g1": results["g1"],
    "gcm": results["gcm"],
    "gr": results["gr"],
    "sweep_data": results["sweep_data"],
}
pkl_path = os.path.join(results_dir, f"results_N{N}.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(serializable, f)

mean_R2 = float(results["final_R2"].mean())
mean_Rg2 = float(results["final_Rg2"].mean())
ratio = mean_R2 / mean_Rg2 if mean_Rg2 > 0 else 0
print(f"[W-N{N}] Done: <R2>={mean_R2:.2f}, <Rg2>={mean_Rg2:.2f}, R2/Rg2={ratio:.4f}", flush=True)
