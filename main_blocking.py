import os
import sys
import json
from plot import Plotter
from svm import PassiveSVM
from logistic_regression import PassiveLogisticRegression
from a2 import A2Algorithm
from rpe import RPEAlgorithm

# Configuration
# Use the blocked data if available, or pass as argument
DATA_FILE = "processed_data/entity_pairs_blocked.csv"
RESULTS_DIR = "results_blocking"
START_SIZE = 0
STEP_SIZE = 1000
END_SIZE = 10000

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run 'python3 init_handle.py --blocking' first.")
        return

    plotter = Plotter()
    
    # 1. Run Algorithms
    print("Starting Experiments (Blocking Mode)...")
    
    # RPE
    print("\n--- Running RPE ---")
    rpe = RPEAlgorithm(DATA_FILE, os.path.join(RESULTS_DIR, "RPE"))
    rpe.load_data()
    rpe_results = rpe.run(START_SIZE, STEP_SIZE, END_SIZE, save_train_sets=True)
    
    # A2
    print("\n--- Running A2 ---")
    a2 = A2Algorithm(DATA_FILE, os.path.join(RESULTS_DIR, "A2"))
    a2.load_data()
    a2_results = a2.run(START_SIZE, STEP_SIZE, END_SIZE, save_train_sets=True)
    
    # SVM
    print("\n--- Running SVM ---")
    svm = PassiveSVM(DATA_FILE, os.path.join(RESULTS_DIR, "SVM"))
    svm.load_data()
    svm_results = svm.run(START_SIZE, STEP_SIZE, END_SIZE, save_train_sets=True)
    
    # Logistic Regression
    print("\n--- Running Logistic Regression ---")
    lr = PassiveLogisticRegression(DATA_FILE, os.path.join(RESULTS_DIR, "LogisticRegression"))
    lr.load_data()
    lr_results = lr.run(START_SIZE, STEP_SIZE, END_SIZE, save_train_sets=True)
    
    # 2. Plot Comparisons
    print("\nGenerating Comparison Plots...")
    
    # Collect data
    all_results = {
        'RPE': rpe_results,
        'A2': a2_results,
        'SVM': svm_results,
        'LogisticRegression': lr_results
    }
    
    # F1 Comparison
    plotter.plot_multi({k: {'x': v['x'], 'y': v['f1']} for k, v in all_results.items()}, 
                       'F1', title="F1 Score Comparison (Blocking)", save_path=os.path.join(RESULTS_DIR, "comparison_f1.png"))
    
    # Recall Comparison
    plotter.plot_multi({k: {'x': v['x'], 'y': v['recall']} for k, v in all_results.items()}, 
                       'Recall', title="Recall Comparison (Blocking)", save_path=os.path.join(RESULTS_DIR, "comparison_recall.png"))
                       
    # Accuracy Comparison
    plotter.plot_multi({k: {'x': v['x'], 'y': v['accuracy']} for k, v in all_results.items()}, 
                       'Accuracy', title="Accuracy Comparison (Blocking)", save_path=os.path.join(RESULTS_DIR, "comparison_accuracy.png"))

    # Precision Comparison
    plotter.plot_multi({k: {'x': v['x'], 'y': v['precision']} for k, v in all_results.items()}, 
                       'Precision', title="Precision Comparison (Blocking)", save_path=os.path.join(RESULTS_DIR, "comparison_precision.png"))

    # Time Comparison - ONLY A2 and RPE
    time_results = {
        'RPE': rpe_results,
        'A2': a2_results
    }
    plotter.plot_multi({k: {'x': v['x'], 'y': v['time']} for k, v in time_results.items()}, 
                       'Time (s)', title="Computation Time per Step (Blocking)", save_path=os.path.join(RESULTS_DIR, "comparison_time.png"))
    
    print(f"Experiments Completed. Results saved in '{RESULTS_DIR}' directory.")

if __name__ == "__main__":
    main()
