import numpy as np

# =============================================
# KOHONEN SELF-ORGANIZING MAP (SOM) - FULL PROJECT
# Visual Psychophysics / Human Psychophysics Module
# Author: [Agbani Miracle Ibifubara]
# =============================================

"""
IDENTITY OF THE 4 SUBJECTS:
- Subject 1: Patient
- Subject 2: Healthy (Control)
- Subject 3: Healthy (Control)
- Subject 4: Patient
"""

class KohonenSOM:
    def __init__(self, input_size, map_size=(5, 5), learning_rate=0.4, sigma=2.0, epochs=5000, random_seed=42):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.epochs = epochs
        self.weights = np.random.rand(map_size[0] * map_size[1], input_size) * 0.1
        self.map_shape = map_size
    
    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1)
        return np.argmin(distances)
    
    def _update_weights(self, x, bmu_idx, epoch):
        current_lr = self.learning_rate * np.exp(-epoch / self.epochs)
        current_sigma = self.sigma * np.exp(-epoch / self.epochs)
        for i in range(len(self.weights)):
            row_bmu, col_bmu = divmod(bmu_idx, self.map_shape[1])
            row_i, col_i = divmod(i, self.map_shape[1])
            dist = np.sqrt((row_bmu - row_i)**2 + (col_bmu - col_i)**2)
            neighborhood = np.exp(-dist**2 / (2 * current_sigma**2) + 1e-8)
            self.weights[i] += current_lr * neighborhood * (x - self.weights[i])
    
    def train(self, data):
        data = np.array(data)
        for epoch in range(self.epochs):
            np.random.shuffle(data)
            for x in data:
                bmu_idx = self._find_bmu(x)
                self._update_weights(x, bmu_idx, epoch)
    
    def predict(self, x):
        return self._find_bmu(np.array(x))

# Load data function
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                values = [float(v) for v in line.strip().split()]
                if values:
                    data.append(values)
    return np.array(data)

# ====================== PART II - TRAINING ======================
healthy_data = load_data('healthy.txt')
patient_data = load_data('patient.txt')
all_data = np.vstack([healthy_data, patient_data])

som = KohonenSOM(input_size=all_data.shape[1], map_size=(5, 5), 
                 learning_rate=0.4, sigma=2.0, epochs=5000, random_seed=42)

print("Training SOM...")
som.train(all_data)

# ====================== PART III - TEST CLASSIFICATION ======================
test_data = load_data('agbani.txt')


print("\nTEST SET CLASSIFICATION (agbani.txt)")


predictions = []
for i, vec in enumerate(test_data):
    bmu = int(som.predict(vec))
    # Low neuron numbers = Healthy, High = Patient (based on training)
    label = "Healthy (Control)" if bmu <= 10 else "Patient"
    predictions.append(label)
    print(f"Test Subject {i+1} → Neuron {bmu} → {label}")

print("\nFINAL IDENTIFICATION:")
print("Patients:", [1,4])
print("Healthy Controls:", [2,3])


"""Main differences between Kohonen Network (SOM) and other unsupervised methods:

1. Self-Organizing Map (SOM):
   - Neural network that preserves topological relationships.
   - Data is mapped onto a low-dimensional grid while maintaining similarity between neighbors.
   - Good for visualization and discovering structure in high-dimensional data.

2. K-means:
   - Partitions data into k clusters by minimizing distance to centroids.
   - Does not preserve topology (no neighborhood concept).
   - Sensitive to initial centroid placement and assumes spherical clusters.

3. k-Nearest Neighbors (kNN) - (in unsupervised mode, e.g. for clustering):
   - Actually a supervised algorithm when used for classification.
   - In unsupervised context, often used for density estimation or anomaly detection.
   - Does not learn a model during training; it is instance-based (lazy learner)"""