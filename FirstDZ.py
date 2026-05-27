# coding: cp1251
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        
    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    def _get_neighbors(self, X, point_idx):
        neighbors = []
        for i in range(len(X)):
            if self._euclidean_distance(X[point_idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def fit(self, X):
        if isinstance(X, list):
            X = np.array(X)
        
        n_samples = len(X)
        
        self.labels_ = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)
        
        cluster_id = 0
        
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
                
            visited[point_idx] = True
            neighbors = self._get_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                continue
                
            self._expand_cluster(X, point_idx, neighbors, cluster_id, visited)
            cluster_id += 1
        
        return self
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        self.labels_[point_idx] = cluster_id
        queue = deque(neighbors)
        
        while queue:
            current_idx = queue.popleft()
            
            if not visited[current_idx]:
                visited[current_idx] = True
                current_neighbors = self._get_neighbors(X, current_idx)
                
                if len(current_neighbors) >= self.min_samples:
                    for neighbor_idx in current_neighbors:
                        if neighbor_idx not in queue and self.labels_[neighbor_idx] != cluster_id:
                            queue.append(neighbor_idx)
            
            if self.labels_[current_idx] == -1:
                self.labels_[current_idx] = cluster_id
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

def visualize_dbscan(X, labels, eps, min_samples):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.scatter(X[:, 0], X[:, 1], c='blue', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_title('Исходные данные (без кластеризации)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Признак X1')
    ax1.set_ylabel('Признак X2')
    ax1.grid(True, alpha=0.3)
    
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            mask = (labels == label)
            ax2.scatter(X[mask, 0], X[mask, 1], c='gray', s=20, alpha=0.5, 
                       marker='x', label=f'Шум ({np.sum(mask)})')
        else:
            mask = (labels == label)
            ax2.scatter(X[mask, 0], X[mask, 1], c=[color], s=40, alpha=0.7,
                       edgecolors='black', linewidth=0.5, label=f'Кластер {label} ({np.sum(mask)})')
    
    ax2.set_title(f'Результат кластеризации DBSCAN\neps={eps}, min_samples={min_samples}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Признак X1')
    ax2.set_ylabel('Признак X2')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("СТАТИСТИКА КЛАСТЕРИЗАЦИИ")
    print("="*50)
    print(f"Всего точек: {len(X)}")
    print(f"Количество кластеров (без учёта шума): {len([l for l in unique_labels if l != -1])}")
    print(f"Количество шумовых точек: {np.sum(labels == -1)} ({np.sum(labels == -1)/len(X)*100:.1f}%)")
    
    for label in sorted(unique_labels):
        if label == -1:
            print(f"  Шум: {np.sum(labels == label)} точек")
        else:
            print(f"  Кластер {label}: {np.sum(labels == label)} точек")

def create_test_data():
    np.random.seed(42)
    
    cluster1 = np.random.randn(80, 2) * 0.3
    cluster2 = np.random.randn(80, 2) * 0.3 + np.array([3, 3])
    cluster3 = np.random.randn(60, 2) * 0.25 + np.array([0, 3])
    noise = np.random.uniform(-2, 5, (30, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3, noise])
    return X

def main():
    print("="*50)
    print("DBSCAN КЛАСТЕРИЗАЦИЯ - ДЕМОНСТРАЦИЯ РАБОТЫ")
    print("="*50)
    
    print("\n1. Генерация тестовых данных...")
    X = create_test_data()
    print(f"   Создано {len(X)} точек")
    
    eps = 0.4
    min_samples = 8
    
    print(f"\n2. Запуск DBSCAN с параметрами:")
    print(f"   eps (радиус окрестности) = {eps}")
    print(f"   min_samples = {min_samples}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    print("\n3. Кластеризация завершена!")
    
    print("\n4. Отображение визуализации...")
    visualize_dbscan(X, labels, eps, min_samples)
    
    print("\nПрограмма завершена!")

if __name__ == "__main__":
    main()
    input("\nНажмите Enter для выхода...")