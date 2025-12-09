import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import os
import time
import json
from plot import Plotter

class PassiveSVM:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.plotter = Plotter()
        
    def load_data(self):
        print("Loading data for SVM...")
        self.df = pd.read_csv(self.data_path)
        self.features = ['name_cos', 'name_jac', 'name_lev', 'desc_cos', 'desc_jac']
        self.X = self.df[self.features].values
        self.y = self.df['label'].values
        
    def run(self, start_size, step_size, end_size, save_train_sets=False):
        results = {
            'x': [],
            'f1': [],
            'recall': [],
            'precision': [],
            'accuracy': [],
            'time': []
        }
        
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(self.df))
        
        current_size = start_size
        
        while current_size <= end_size:
            if current_size == 0:
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
                
            print(f"SVM: Training with size {current_size}...")
            start_time = time.time()
            
            train_indices = shuffled_indices[:current_size]
            X_train = self.X[train_indices]
            y_train = self.y[train_indices]
            
            if len(np.unique(y_train)) < 2:
                f1, recall, precision, acc = 0, 0, 0, 0
            else:
                # Use Linear SVM for speed, or RBF if needed. Given features are similarities (0-1), Linear might be okay but RBF is standard.
                # However, for 1M points, SVM prediction is slow.
                # If we train on small set (up to 3000), prediction on 1M is O(N_test * N_sv).
                # With 3000 SVs, 1M * 3000 is 3*10^9 ops. Might be slow.
                # I'll use LinearSVC which is much faster for prediction (O(N_test * d)).
                from sklearn.svm import LinearSVC
                model = LinearSVC(class_weight='balanced', dual=False, max_iter=10000)
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
                save_path = os.path.join(self.output_dir, f"train_indices_{current_size}.npy")
                np.save(save_path, train_indices)
            
            current_size += step_size
            
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f)
            
        self.plotter.plot_single(results['x'], results['f1'], 'SVM', 'F1', save_path=os.path.join(self.output_dir, "f1.png"))
        
        return results
