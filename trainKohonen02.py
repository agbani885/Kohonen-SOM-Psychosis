import numpy as np

class KohonenSOM:
    def __init__(self, input_size, map_size=(5, 5), learning_rate=0.5, sigma=2.0, epochs=10000, random_seed=42):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.epochs = epochs
        self.weights = np.random.rand(map_size[0] * map_size[1], input_size) * 0.1  # Small initialization
        self.map_shape = map_size
    
    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1)
        return np.argmin(distances)
    
    def _update_weights(self, x, bmu_idx, epoch):
        current_lr = self.learning_rate * np.exp(-epoch / self.epochs)
        current_sigma = self.sigma * np.exp(-epoch / self.epochs)
        
        for i in range(len(self.weights)):
            # 2D grid distance
            row_bmu, col_bmu = divmod(bmu_idx, self.map_shape[1])
            row_i, col_i = divmod(i, self.map_shape[1])
            dist = np.sqrt((row_bmu - row_i)**2 + (col_bmu - col_i)**2)
            neighborhood = np.exp(-dist**2 / (2 * current_sigma**2))
            self.weights[i] += current_lr * neighborhood * (x - self.weights[i])
    
    def train(self, data):
        data = np.array(data)
        for epoch in range(self.epochs):
            np.random.shuffle(data)  # Important for better convergence
            for x in data:
                bmu_idx = self._find_bmu(x)
                self._update_weights(x, bmu_idx, epoch)
        print(f"Training completed on {len(data)} subjects with {data.shape[1]}-dimensional vectors.")
    
    def predict(self, x):
        return self._find_bmu(np.array(x))

# Load data
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                values = [float(v) for v in line.strip().split()]
                data.append(values)
    return np.array(data)

healthy_data = load_data('healthy.txt')
patient_data = load_data('patient.txt')

print(f"Healthy data shape: {healthy_data.shape}")
print(f"Patient data shape: {patient_data.shape}")

# Combine for unsupervised training
all_data = np.vstack([healthy_data, patient_data])

# Train SOM
input_dim = all_data.shape[1]
som = KohonenSOM(input_size=input_dim, map_size=(6, 6), learning_rate=0.6, sigma=3.0, epochs=8000, random_seed=42)

print("Starting training on real marker data...")
som.train(all_data)

# Analyze clusters
print("\n=== Cluster Analysis ===")
healthy_bmus = [int(som.predict(vec)) for vec in healthy_data]
patient_bmus = [int(som.predict(vec)) for vec in patient_data]

print(f"Healthy subjects BMUs: {set(healthy_bmus)}")
print(f"Patient subjects BMUs: {set(patient_bmus)}")
print("\nNumber of unique neurons used by Healthy:", len(set(healthy_bmus)))
print("Number of unique neurons used by Patients:", len(set(patient_bmus)))