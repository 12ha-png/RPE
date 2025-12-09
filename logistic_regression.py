import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import os
import time
import json
from plot import Plotter

class PassiveLogisticRegression:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.plotter = Plotter()
        
    def load_data(self):
        print("Loading data for LR...")
        self.df = pd.read_csv(self.data_path)
        # Features
        self.features = ['name_cos', 'name_jac', 'name_lev', 'desc_cos', 'desc_jac']
        self.X = self.df[self.features].values
        self.y = self.df['label'].values
        self.ids = self.df.index.values # Use index as ID
        
    def run(self, start_size, step_size, end_size, save_train_sets=False):
        results = {
            'x': [],
            'f1': [],
            'recall': [],
            'precision': [],
            'accuracy': [],
            'time': []
        }
        
        # Initialize random pool
        # Passive learning: Randomly select samples.
        # "Incremental": We select indices.
        # Shuffle all indices once
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(self.df))
        
        current_size = start_size
        
        while current_size <= end_size:
            if current_size == 0:
                # 0 cost -> No training? Or random guess? 
                # Usually 0 means no model. We skip or record 0.
                results['x'].append(0)
                results['f1'].append(0)
                results['recall'].append(0)
                results['precision'].append(0)
                results['accuracy'].append(0)
                results['time'].append(0)
                current_size += step_size
                continue
                
            if current_size > len(self.df):
                break
                
            print(f"LR: Training with size {current_size}...")
            start_time = time.time()
            
            train_indices = shuffled_indices[:current_size]
            # Test on ALL data (or remaining? Prompt says "test set" but "prediction is for real training set"? 
            # "Accuracy is for training set, Prediction is for real training set prediction situation" - this is confusing.
            # Usually: Train on subset, Test on ALL (or Holdout). 
            # Prompt: "Passive learning: adopt limited training set training... used for overall and compute precision..."
            # So Test on ALL.
            
            X_train = self.X[train_indices]
            y_train = self.y[train_indices]
            
            # Check if we have both classes
            if len(np.unique(y_train)) < 2:
                # Cannot train properly
                f1, recall, precision, acc = 0, 0, 0, 0
            else:
                model = LogisticRegression(class_weight='balanced') # Important for imbalanced data
                model.fit(X_train, y_train)
                
                y_pred = model.predict(self.X)
                
                f1 = f1_score(self.y, y_pred)
                recall = recall_score(self.y, y_pred)
                precision = precision_score(self.y, y_pred)
                acc = accuracy_score(self.y, y_pred)
            
            elapsed = time.time() - start_time
            
            results['x'].append(current_size)
            results['f1'].append(f1)
            results['recall'].append(recall)
            results['precision'].append(precision)
            results['accuracy'].append(acc)
            results['time'].append(elapsed)
            
            if save_train_sets:
                # Save indices
                save_path = os.path.join(self.output_dir, f"train_indices_{current_size}.npy")
                np.save(save_path, train_indices)
            
            current_size += step_size
            
        # Save results
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f)
            
        # Plot
        self.plotter.plot_single(results['x'], results['f1'], 'LogisticRegression', 'F1', save_path=os.path.join(self.output_dir, "f1.png"))
        
        return results

if __name__ == "__main__":
    # Test run
    pass
