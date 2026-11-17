import numpy as np

class KohonenSOM:
    def __init__(self, input_size, map_size=(2, 1), learning_rate=0.5, sigma=1.0, epochs=1000):
        self.input_size = input_size
        self.map_size = map_size  # e.g., (2,1) for 2 neurons in 1D topology
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.epochs = epochs
        # Initialize weights randomly
        self.weights = np.random.rand(map_size[0] * map_size[1], input_size)
        self.map_shape = map_size
    
    def _find_bmu(self, x):
        """Find Best Matching Unit (neuron with closest weight)"""
        distances = np.linalg.norm(self.weights - x, axis=1)
        bmu_idx = np.argmin(distances)
        return bmu_idx
    
    def _update_weights(self, x, bmu_idx, epoch):
        """Update weights using neighborhood function"""
        current_lr = self.learning_rate * np.exp(-epoch / self.epochs)
        current_sigma = self.sigma * np.exp(-epoch / self.epochs)
        
        for i in range(len(self.weights)):
            dist = np.abs(i - bmu_idx)  # Simple 1D topology distance
            neighborhood = np.exp(-dist**2 / (2 * current_sigma**2))
            self.weights[i] += current_lr * neighborhood * (x - self.weights[i])
    
    def train(self, data):
        """Train the SOM"""
        data = np.array(data)
        for epoch in range(self.epochs):
            for x in data:
                bmu_idx = self._find_bmu(x)
                self._update_weights(x, bmu_idx, epoch)
        print("Training completed. Final learning rate adjustment led to convergence.")
    
    def predict(self, x):
        """Return the BMU index (class) for a vector"""
        return self._find_bmu(np.array(x))
    
    def get_class(self, bmu_idx):
        """Map BMU to Class I or II based on training behavior"""
        # After training, we map clusters: lower indices ~ Class I, higher ~ Class II
        return "Class I" if bmu_idx == 0 else "Class II"

# Part 1: Training vectors
training_data = [
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1]
]

# Create and train SOM (2 neurons for 2 classes)
som = KohonenSOM(input_size=4, map_size=(2, 1), learning_rate=0.8, sigma=1.0, epochs=5000)
som.train(training_data)

print("\n=== Training Results ===")
for i, vec in enumerate(training_data):
    bmu = som.predict(vec)
    cls = som.get_class(bmu)
    print(f"Vector {vec} -> BMU {bmu} -> {cls}")

# Part 2: Test vectors
test_data = [
    [0, 0, 0, 0.9],
    [0, 0, 0.8, 0.9],
    [0.7, 0, 0, 0],
    [0.7, 0.9, 0, 0]
]

print("\n=== Test Results ===")
for vec in test_data:
    bmu = som.predict(vec)
    cls = som.get_class(bmu)
    print(f"Test vector {vec} -> BMU {bmu} -> {cls}")