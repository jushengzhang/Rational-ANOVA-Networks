import openml
import pandas as pd
import numpy as np
import os
import warnings

# ==========================================
# Global Configuration
# ==========================================
RESULTS_FILE = "tabular_benchmark_results.csv"
SUITE_ID = 457  # OpenML Suite ID (e.g., TabArena)

def get_target_tasks(limit=10):
    """
    Retrieves the task list for the specified OpenML suite and automatically 
    filters out task IDs that already exist in the RESULTS_FILE.
    """
    print(f">>> Connecting to OpenML Suite {SUITE_ID}...")
    try:
        suite = openml.study.get_suite(suite_id=SUITE_ID)
        tasks = suite.tasks
        
        # Filter out completed tasks
        if os.path.exists(RESULTS_FILE):
            try:
                done_df = pd.read_csv(RESULTS_FILE)
                if 'TaskID' in done_df.columns:
                    done_ids = done_df['TaskID'].unique().tolist()
                    tasks = [t for t in tasks if t not in done_ids]
            except Exception as e:
                print(f"Failed to read existing results file. Re-running all tasks: {e}")
        
        final_tasks = tasks[:limit]
        print(f"✅ Scheduled {len(final_tasks)} new tasks (Total remaining: {len(tasks)})")
        return final_tasks
    except Exception as e:
        print(f"❌ Failed to retrieve task list: {e}")
        return []

def get_openml_data(task_id):
    """
    Downloads data for a specific Task ID.
    
    Returns:
        (dataset_name, dataframe, label_column, task_type)
    """
    try:
        # 1. Get task metadata (download_data=False for speed)
        task = openml.tasks.get_task(task_id, download_data=False)
        
        # === Core Logic: Determine Task Type based on Official ID ===
        # OpenML API might return an integer (2) or an Enum object (TaskType.SUPERVISED_REGRESSION).
        # The most robust method is to convert to string and uppercase.
        type_str = str(task.task_type_id).upper()
        
        # Logic Determination:
        # - If it contains 'REGRESSION' -> regression
        # - If it equals '2' -> regression (OpenML Standard: 1=Classification, 2=Regression)
        # - Otherwise -> classification
        if 'REGRESSION' in type_str or type_str == '2':
            task_type = 'regression'
        else:
            task_type = 'classification'
            
        # 2. Get actual dataset
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, 
            dataset_format="dataframe"
        )
        
        # [Debug] Verify the task type detection
        # print(f"   [Debug] Task {task_id} ({dataset.name}): Raw ID={type_str} -> {task_type}")
        
        return dataset.name, pd.concat([X, y], axis=1), dataset.default_target_attribute, task_type
        
    except Exception as e:
        print(f"OpenML Data Download Error (Task {task_id}): {e}")
        return None, None, None, None