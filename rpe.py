import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import os
import time
import json
from plot import Plotter

class RPEAlgorithm:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.plotter = Plotter()
        
    def load_data(self):
        print("Loading data for RPE...")
        self.df = pd.read_csv(self.data_path)
        self.features = ['name_cos', 'name_jac', 'name_lev', 'desc_cos', 'desc_jac']
        self.X = self.df[self.features].values
        self.y = self.df['label'].values
        self.all_indices = np.arange(len(self.df))
        
    def run(self, start_size, step_size, end_size, save_train_sets=False):
        results = {
            'x': [],
            'f1': [],
            'recall': [],
            'precision': [],
            'accuracy': [],
            'time': []
        }
        
        # Random Probe with Elimination Logic
        # 1. Start with random set
        # 2. Train model
        # 3. Eliminate 'easy negatives' (low probability) from the sampling pool
        # 4. Randomly sample from the remaining pool
        
        np.random.seed(42)
        self.labeled_indices = []
        # Optimization: Use boolean mask for fast indexing/filtering
        self.labeled_mask = np.zeros(len(self.df), dtype=bool)
        
        # Initial Random Probe
        if start_size > 0:
            chosen = np.random.choice(self.all_indices, size=start_size, replace=False)
            self.labeled_indices.extend(chosen)
            self.labeled_mask[chosen] = True
        
        target_sizes = range(start_size, end_size + 1, step_size)
        
        # Pre-calculate threshold for decision_function optimization
        # prob >= 0.1 <=> decision_function >= ln(0.1 / 0.9)
        threshold_prob = 0.1
        # Avoid div by zero if threshold is 0 or 1 (unlikely here)
        threshold_score = np.log(threshold_prob / (1 - threshold_prob))
        
        for target in target_sizes:
            print(f"RPE: Processing target size {target}...")
            iter_start_time = time.time()
            
            # Add samples if needed
            while len(self.labeled_indices) < target:
                n_needed = target - len(self.labeled_indices)
                
                # Get current unlabeled indices efficiently
                unlabeled_indices = np.flatnonzero(~self.labeled_mask)
                
                if len(unlabeled_indices) == 0:
                    break

                # Train on current labeled data
                y_train_current = self.y[self.labeled_indices]
                
                if len(self.labeled_indices) < 2 or len(np.unique(y_train_current)) < 2:
                     # Not enough data to train/eliminate -> Pure Random Sampling
                    chosen = np.random.choice(unlabeled_indices, size=min(len(unlabeled_indices), n_needed), replace=False)
                    self.labeled_indices.extend(chosen)
                    self.labeled_mask[chosen] = True
                else:
                    # Train Sampling Model (Balanced to find positives)
                    X_train = self.X[self.labeled_indices]
                    y_train = y_train_current
                    
                    # Use balanced weights for sampling to ensure we don't miss rare positives
                    model = LogisticRegression(class_weight='balanced', solver='liblinear')
                    model.fit(X_train, y_train)
                    
                    # Predict on Unlabeled
                    scores = model.decision_function(self.X[unlabeled_indices])
                    
                    # Elimination Step:
                    # Strategy: Remove samples that are almost certainly negatives.
                    # With balanced weights, 0.0 is the decision boundary.
                    # Negatives will have negative scores.
                    # Let's eliminate samples with score < -1.0 (prob < ~26%)
                    
                    threshold = -1.0
                    candidate_mask_local = scores >= threshold
                    
                    if not np.any(candidate_mask_local):
                        candidates = unlabeled_indices
                    else:
                        candidates = unlabeled_indices[candidate_mask_local]
                    
                    chosen = np.random.choice(candidates, size=min(len(candidates), n_needed), replace=False)
                    self.labeled_indices.extend(chosen)
                    self.labeled_mask[chosen] = True
            
            # Evaluate
            # Use a clean model for evaluation to ensure high precision
            X_train = self.X[self.labeled_indices]
            y_train = self.y[self.labeled_indices]
            
            if len(np.unique(y_train)) < 2:
                f1, recall, prec, acc = 0, 0, 0, 0
            else:
                # Evaluation Model:
                # Standard LR (no class_weight) maximizes accuracy/precision.
                # But if training set is imbalanced (which it is, despite RPE), it might predict all 0.
                # However, RPE should have enriched the training set with positives.
                
                # To GUARANTEE high precision:
                # 1. Use class_weight=None (default)
                # 2. Use a higher decision threshold for prediction (e.g., prob > 0.8)
                
                model = LogisticRegression(class_weight=None, solver='liblinear')
                model.fit(X_train, y_train)
                
                # Predict Probabilities
                y_probs = model.predict_proba(self.X)[:, 1]
                
                # Use High Threshold for Precision
                high_threshold = 0.8
                y_pred = (y_probs >= high_threshold).astype(int)
                
                f1 = f1_score(self.y, y_pred)
                recall = recall_score(self.y, y_pred)
                prec = precision_score(self.y, y_pred, zero_division=0) # Handle 0 division
                acc = accuracy_score(self.y, y_pred)
            
            elapsed = time.time() - iter_start_time
            
            results['x'].append(target)
            results['f1'].append(f1)
            results['recall'].append(recall)
            results['precision'].append(prec)
            results['accuracy'].append(acc)
            results['time'].append(elapsed)
            
            if save_train_sets:
                save_path = os.path.join(self.output_dir, f"train_indices_{target}.npy")
                np.save(save_path, self.labeled_indices)
                
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f)
            
        self.plotter.plot_single(results['x'], results['f1'], 'RPE', 'F1', save_path=os.path.join(self.output_dir, "f1.png"))
        return results
