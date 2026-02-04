import os
import shutil
import warnings
import time
import pandas as pd
import torch
import multiprocessing as mp
from queue import Empty
import autogluon.common.space as ag
from autogluon.tabular import TabularPredictor

# === Import Custom Modules ===
# Assumes previous files are saved as TAB_model.py and TAB_dataload.py
from TAB_model import RationalTABModel  
from TAB_dataload import get_target_tasks, get_openml_data, RESULTS_FILE 

warnings.filterwarnings('ignore')

# ==========================================
# 1. Experiment Configuration
# ==========================================
# [Config 1] Specify GPU IDs to use
# Anonymized: Automatically detect all available GPUs. 
# Users can manually set this list (e.g., [0, 1, 2]) if needed.
try:
    device_count = torch.cuda.device_count()
    TARGET_GPUS = list(range(device_count))
except:
    TARGET_GPUS = []

# [Config 2] How many tasks to run concurrently per GPU?
JOBS_PER_GPU = 3 

TIME_LIMIT_PER_DATASET = 3600  # 1 Hour
HPO_KWARGS = {'num_trials': 60, 'scheduler': 'local', 'searcher': 'random'}

# Hyperparameter Search Space
SEARCH_SPACE = {
    RationalTABModel: {
        'epochs': 50,
        'lr': ag.Real(1e-4, 5e-2, default=1e-2, log=True),
        'num_heads': ag.Categorical(2, 4),
        'hidden_dim1': ag.Categorical(32, 64),
        'batch_size': ag.Categorical(128, 256),
        'degree_P': ag.Categorical(*list(range(2, 7))),
        'degree_Q': ag.Categorical(*list(range(2, 7))),
    }
}

# ==========================================
# 2. GPU Worker Logic
# ==========================================
def gpu_worker(gpu_id, worker_id, task_queue, result_queue):
    # Bind process to specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"🔧 [GPU {gpu_id}-W{worker_id}] Worker Online")
    
    while True:
        try:
            task_id = task_queue.get(timeout=5)
        except Empty:
            break

        # 1. Download Data
        try:
            name, df, label_col, task_type = get_openml_data(task_id)
            if df is None: continue
        except Exception as e:
            print(f"❌ [GPU {gpu_id}] Data Error: {e}")
            continue

        print(f"🔥 [GPU {gpu_id}-{worker_id}] Training: {name} (ID:{task_id} | {task_type})")
        
        # Unique model save path (Includes Worker ID to prevent conflicts)
        save_path = f"AutogluonModels/GPU{gpu_id}_W{worker_id}_{task_id}"
        if os.path.exists(save_path): shutil.rmtree(save_path)

        ag_problem_type = 'regression' if task_type == 'regression' else None
        eval_metric = 'root_mean_squared_error' if task_type == 'regression' else 'accuracy'

        try:
            start_time = time.time()
            predictor = TabularPredictor(
                label=label_col,
                problem_type=ag_problem_type,
                eval_metric=eval_metric,
                path=save_path,
                verbosity=0
            ).fit(
                train_data=df,
                hyperparameters=SEARCH_SPACE,
                hyperparameter_tune_kwargs=HPO_KWARGS,
                num_gpus=1,
                time_limit=TIME_LIMIT_PER_DATASET,
                fit_weighted_ensemble=False,
                presets='medium'
            )
            tune_time = time.time() - start_time
            
            # Check Results
            leaderboard = predictor.leaderboard(silent=True)
            if leaderboard is None or leaderboard.empty:
                raise RuntimeError("All trials failed")

            best_model = predictor.model_best
            score_val = leaderboard.loc[leaderboard['model'] == best_model, 'score_val'].values[0]
            
            model_info = predictor.info()['model_info'].get(best_model, {})
            hparams = model_info.get('hyperparameters', {})
            if 'RationalTABModel' in hparams:
                 hparams = hparams['RationalTABModel']
            
            result_entry = {
                "Dataset": name,
                "TaskID": task_id,
                "Type": task_type,
                "Metric": eval_metric,
                "Score": score_val,
                "Time": round(tune_time, 1),
                "Best_P": hparams.get('degree_P', '-'),
                "Best_Q": hparams.get('degree_Q', '-'),
                "Best_LR": hparams.get('lr', '-')
            }
            print(f"✅ [GPU {gpu_id}-{worker_id}] {name}: {score_val:.4f}")
            result_queue.put(result_entry)

        except Exception as e:
            print(f"❌ [GPU {gpu_id}-{worker_id}] {name} Failed: {str(e)}")
            result_queue.put({"Dataset": name, "TaskID": task_id, "Error": str(e)})
        
        finally:
            if os.path.exists(save_path): shutil.rmtree(save_path)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    try:
        sys_gpus = torch.cuda.device_count()
    except: sys_gpus = 0
    
    if sys_gpus == 0:
        print("❌ No GPU detected. Cannot run benchmark.")
        return

    # Filter out valid GPUs from the configuration
    valid_gpus = [g for g in TARGET_GPUS if g < sys_gpus]
    
    # Calculate total concurrency
    total_workers = len(valid_gpus) * JOBS_PER_GPU

    print(f"🚀 Starting Parallel Benchmark")
    print(f"   - Available Physical GPUs: {valid_gpus}")
    print(f"   - Jobs per GPU: {JOBS_PER_GPU}")
    print(f"   - Total Workers: {total_workers}")

    if not valid_gpus:
        print("❌ No valid GPUs found (Check TARGET_GPUS configuration)")
        return

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # 1. Fill Task Queue
    tasks = get_target_tasks(limit=100) 
    for t in tasks:
        task_queue.put(t)

    # 2. Start Worker Processes (Double Loop)
    processes = []
    for gpu_id in valid_gpus:
        for job_idx in range(JOBS_PER_GPU):
            worker_id = job_idx 
            p = mp.Process(target=gpu_worker, args=(gpu_id, worker_id, task_queue, result_queue))
            p.start()
            processes.append(p)

    # 3. Monitor Results
    print(f"👀 {len(processes)} workers are running full speed...")
    
    finished = 0
    total = len(tasks)
    
    while finished < total:
        if not any(p.is_alive() for p in processes) and result_queue.empty():
            print("⚠️ All child processes have exited.")
            break
        try:
            res = result_queue.get(timeout=2)
            df_res = pd.DataFrame([res])
            header = not os.path.exists(RESULTS_FILE)
            df_res.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
            
            finished += 1
            status = f"{res.get('Score', 'Fail')}"
            print(f"📊 Progress: {finished}/{total} | {res['Dataset']} -> {status}")
            
        except Empty:
            continue

    for p in processes:
        p.join()
    print("\n🏆 All tasks completed.")

if __name__ == "__main__":
    main()