import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self):
        self.colors = {
            'RPE': 'red',
            'A2': 'blue',
            'SVM': 'green',
            'LogisticRegression': 'orange'
        }
        self.markers = {
            'RPE': 'o',
            'A2': 's',
            'SVM': '^',
            'LogisticRegression': 'v'
        }
        
    def get_color(self, algo_name):
        return self.colors.get(algo_name, 'black')

    def plot_single(self, x, y, algo_name, metric_name, title=None, save_path=None, show=False):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, color=self.get_color(algo_name), marker=self.markers.get(algo_name, 'o'), label=algo_name)
        plt.title(title if title else f"{metric_name} over Cost")
        plt.xlabel("Labeled Cost / Probe Size")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()

    def plot_multi(self, data_dict, metric_name, title=None, save_path=None, show=False):
        """
        data_dict: {
            'AlgoName': {'x': [], 'y': []}
        }
        """
        plt.figure(figsize=(10, 6))
        
        for algo_name, data in data_dict.items():
            if 'x' in data and 'y' in data:
                plt.plot(data['x'], data['y'], 
                         color=self.get_color(algo_name), 
                         marker=self.markers.get(algo_name, 'o'), 
                         label=algo_name)
        
        plt.title(title if title else f"{metric_name} Comparison")
        plt.xlabel("Labeled Cost / Probe Size")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        if show:
            plt.show()
            
        plt.close()
