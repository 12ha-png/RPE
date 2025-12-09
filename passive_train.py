import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
import os
import json
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = "processed_data/entity_pairs_blocked.csv" # Use blocked data as it's cleaner
OUTPUT_DIR = "results_passive_ratios"
RATIOS = [0.3, 0.4, 0.5]

def train_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not os.path.exists(DATA_FILE):
        # Fallback to unblocked if blocked not found
        fallback = "processed_data/entity_pairs.csv"
        if os.path.exists(fallback):
            print(f"Blocked data not found, using {fallback}")
            data_path = fallback
        else:
            print("Error: No data found.")
            return
    else:
        data_path = DATA_FILE

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    features = ['name_cos', 'name_jac', 'name_lev', 'desc_cos', 'desc_jac']
    X = df[features].values
    y = df['label'].values
    
    results = {}
    
    # For passive learning at X%, we just split the data
    # Note: "30% sampling" usually means 30% training data.
    
    for ratio in RATIOS:
        print(f"\n--- Ratio: {ratio} ---")
        # Random split
        indices = np.random.permutation(len(df))
        split_point = int(len(df) * ratio)
        train_idx = indices[:split_point]
        test_idx = indices[split_point:] # Or test on ALL? Usually test on remainder or test on separate set.
        # Prompt says "Passive learning ... sample reaches 30% ... performance".
        # "Performance" usually implies testing on the REST or a holdout.
        # However, in previous experiments we tested on ALL.
        # To be consistent with "Evaluation of passive learning performance", let's test on the REMAINING 70%/60%/50% (Standard Cross Val style)
        # OR test on the full dataset if the goal is to label the full dataset.
        # Given the context of "Labeling Cost", it implies we labeled 30%, and want to know how good the model is on the UNLABELED part (or all).
        # Let's test on the remaining data (unlabeled pool).
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        # SVM
        svm_model = LinearSVC(class_weight='balanced', dual=False, max_iter=10000)
        svm_res = train_evaluate("SVM", svm_model, X_train, y_train, X_test, y_test)
        
        # LR
        lr_model = LogisticRegression(class_weight='balanced')
        lr_res = train_evaluate("LogisticRegression", lr_model, X_train, y_train, X_test, y_test)
        
        results[ratio] = {
            'SVM': svm_res,
            'LogisticRegression': lr_res
        }
        
        print(f"Ratio {ratio}:")
        print(f"  SVM F1: {svm_res['f1']:.4f}")
        print(f"  LR  F1: {lr_res['f1']:.4f}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "passive_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Plotting
    # Bar chart for F1 at different ratios
    ratios_str = [f"{int(r*100)}%" for r in RATIOS]
    svm_f1 = [results[r]['SVM']['f1'] for r in RATIOS]
    lr_f1 = [results[r]['LogisticRegression']['f1'] for r in RATIOS]
    
    x = np.arange(len(ratios_str))
    width = 0.25  # Thinner bars
    
    # Use better colors and style
    plt.style.use('seaborn-v0_8-whitegrid') # or 'ggplot'
    
    # F1 Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, svm_f1, width, label='SVM', color='#2ca02c', alpha=0.8)
    rects2 = ax.bar(x + width/2, lr_f1, width, label='LogisticRegression', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Training Data Ratio', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Passive Learning Performance (F1 Score)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(ratios_str, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    
    # Adjust Y-axis to make differences visible, but keep 0 if values are small
    # If values are close (e.g. 0.16 vs 0.17), zooming in helps.
    all_f1 = svm_f1 + lr_f1
    if all_f1:
        min_val = min(all_f1)
        max_val = max(all_f1)
        margin = (max_val - min_val) * 0.5
        # ax.set_ylim(max(0, min_val - margin), max_val + margin) 
        # Or just standard 0-1 if values are high, but here they are ~0.16
        ax.set_ylim(0, max_val * 1.2) # Give some headroom

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "passive_performance_f1.png"), dpi=300)
    plt.close()

    # Accuracy Plot
    svm_acc = [results[r]['SVM']['accuracy'] for r in RATIOS]
    lr_acc = [results[r]['LogisticRegression']['accuracy'] for r in RATIOS]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, svm_acc, width, label='SVM', color='#2ca02c', alpha=0.8)
    rects2 = ax.bar(x + width/2, lr_acc, width, label='LogisticRegression', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Training Data Ratio', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_title('prediction', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(ratios_str, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    autolabel(rects1)
    autolabel(rects2)

    # Set y-axis to 0.9 - 0.95 as requested
    ax.set_ylim(0.90, 0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "passive_performance_accuracy.png"), dpi=300)
    plt.close()
    
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
