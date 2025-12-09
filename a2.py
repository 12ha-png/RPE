import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import os
import time
import json
from plot import Plotter

class A2Algorithm:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.plotter = Plotter()
        
    def load_data(self):
        print("Loading data for A2...")
        self.df = pd.read_csv(self.data_path)
        self.features = ['name_cos', 'name_jac', 'name_lev', 'desc_cos', 'desc_jac']
        self.X = self.df[self.features].values
        self.y = self.df['label'].values
        # Initialize unlabelled pool (indices)
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
        
        # Initial Seed: Randomly select 'start_size' (or small number if start=0)
        # If start_size is 0, we start with 0.
        # If start_size > 0, we pick random.
        
        np.random.seed(42)
        labeled_indices = []
        if start_size > 0:
             labeled_indices = list(np.random.choice(self.all_indices, size=start_size, replace=False))
        
        current_size = len(labeled_indices)
        
        # We need to loop until end_size
        # The loop logic in prompt: "Probe quantity range (start, step, end)"
        # So we should report at start, start+step, ...
        
        # If start_size is 0, we report 0 then add step.
        
        # Helper to train and eval
        def train_eval(indices):
            if not indices:
                return 0, 0, 0, 0
            
            X_train = self.X[indices]
            y_train = self.y[indices]
            
            if len(np.unique(y_train)) < 2:
                # Fallback if only one class
                return 0, 0, 0, 0
                
            model = LogisticRegression(class_weight='balanced')
            model.fit(X_train, y_train)
            y_pred = model.predict(self.X)
            
            return (f1_score(self.y, y_pred), 
                    recall_score(self.y, y_pred), 
                    precision_score(self.y, y_pred), 
                    accuracy_score(self.y, y_pred))

        # Main Loop
        # We iterate based on the requested sizes
        target_sizes = range(start_size, end_size + 1, step_size)
        
        # To support "Resume" or "Add to pool", we maintain `labeled_indices`.
        
        for target in target_sizes:
            print(f"A2: Processing target size {target}...")
            iter_start_time = time.time()
            
            # Add samples if needed
            while len(labeled_indices) < target:
                # Select batch to add
                n_needed = target - len(labeled_indices)
                # We can add in smaller batches or one go.
                # For efficiency, we add `n_needed` at once using the current model.
                
                # Train on current
                if len(labeled_indices) < 2 or len(np.unique(self.y[labeled_indices])) < 2:
                    # Random sampling if not enough data to train
                    remaining = list(set(self.all_indices) - set(labeled_indices))
                    if not remaining: break
                    chosen = np.random.choice(remaining, size=min(len(remaining), n_needed), replace=False)
                    labeled_indices.extend(chosen)
                else:
                    # Active Selection
                    X_train = self.X[labeled_indices]
                    y_train = self.y[labeled_indices]
                    model = LogisticRegression(class_weight='balanced')
                    model.fit(X_train, y_train)
                    
                    # Predict proba on unlabeled
                    unlabeled = list(set(self.all_indices) - set(labeled_indices))
                    if not unlabeled: break
                    
                    # Optimization: Don't predict on ALL 1M if not needed? 
                    # But we need to find the BEST.
                    # Prediction on 1M is fast with LR (vectorized).
                    probs = model.predict_proba(self.X[unlabeled])[:, 1]
                    
                    # Uncertainty: Closest to 0.5
                    uncertainty = np.abs(probs - 0.5)
                    
                    # Sort by uncertainty (ascending)
                    # Get indices of top n_needed
                    sorted_args = np.argsort(uncertainty)
                    top_args = sorted_args[:n_needed]
                    
                    chosen_indices = [unlabeled[i] for i in top_args]
                    labeled_indices.extend(chosen_indices)
            
            # Evaluate
            f1, recall, prec, acc = train_eval(labeled_indices)
            elapsed = time.time() - iter_start_time
            
            results['x'].append(target)
            results['f1'].append(f1)
            results['recall'].append(recall)
            results['precision'].append(prec)
            results['accuracy'].append(acc)
            results['time'].append(elapsed)
            
            if save_train_sets:
                save_path = os.path.join(self.output_dir, f"train_indices_{target}.npy")
                np.save(save_path, labeled_indices)
                
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f)
            
        self.plotter.plot_single(results['x'], results['f1'], 'A2', 'F1', save_path=os.path.join(self.output_dir, "f1.png"))
        return results
