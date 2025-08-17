import numpy as np
import tensorflow as tf
from create_compressed_model import create_resnet9_model # Reuse model creation logic

class ECGClient:
    """Simulates a client device in the federated network."""
    def __init__(self, client_id, X_data, y_data, model_template):
        self.client_id = client_id
        self.X_data = X_data
        self.y_data = y_data
        self.model = tf.keras.models.clone_model(model_template)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def set_weights(self, weights):
        """Sets the model weights from the global model."""
        self.model.set_weights(weights)

    def train(self):
        """Trains the model on local data."""
        self.model.fit(self.X_data, self.y_data, epochs=1, batch_size=32, verbose=0)

    def get_updates(self):
        """Gets model updates and simulates motif vector."""
        layer_updates = [self.model.layers[i].get_weights() for i in [1, 4, 8]] # Indices of first conv in blocks
        feature_extractor = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.get_layer('global_average_pooling1d').output)
        features = feature_extractor.predict(self.X_data[:100], verbose=0)
        motif_vector = np.mean(features, axis=0)
        epsilon = 0.5
        for layer_weights in layer_updates:
            for i in range(len(layer_weights)):
                noise = np.random.laplace(0, 1.0 / epsilon, layer_weights[i].shape)
                layer_weights[i] += noise
        return layer_updates, motif_vector, len(self.y_data)

class FederatedServer:
    """Simulates the central server orchestrating the federated learning."""
    def __init__(self, model_template):
        self.global_model = tf.keras.models.clone_model(model_template)

    def aggregate_updates(self, client_updates):
        """Aggregates client updates using MA-FedAvg."""
        all_weights = [update[0] for update in client_updates]
        all_motifs = np.array([update[1] for update in client_updates])
        all_counts = np.array([update[2] for update in client_updates])
        mean_motif = np.mean(all_motifs, axis=0)
        motif_sim = np.dot(all_motifs, mean_motif) / (np.linalg.norm(all_motifs, axis=1) * np.linalg.norm(mean_motif))
        final_weights = (all_counts / np.sum(all_counts)) * motif_sim
        final_weights /= np.sum(final_weights)

        aggregated_layer_weights = []
        for layer_idx in range(len(all_weights[0])):
            avg_tensors = [np.sum([w[tensor_idx] * final_weights[i] for i, w in enumerate(all_weights)], axis=0) for tensor_idx in range(len(all_weights[0][0]))]
            aggregated_layer_weights.append(avg_tensors)
        return aggregated_layer_weights

    def run_federated_rounds(self, clients, rounds=50):
        """Main federated training loop."""
        for r in range(rounds):
            print(f"--- Federated Round {r+1}/{rounds} ---")
            num_selected = max(1, int(len(clients) * 0.3))
            selected_indices = np.random.choice(len(clients), num_selected, replace=False)
            selected_clients = [clients[i] for i in selected_indices]
            client_updates = []
            for client in selected_clients:
                client.set_weights(self.global_model.get_weights())
                client.train()
                client_updates.append(client.get_updates())
            
            aggregated_weights = self.aggregate_updates(client_updates)
            current_global_weights = self.global_model.get_weights()
            for i, layer_idx in enumerate([1, 4, 8]):
                current_global_weights[layer_idx] = aggregated_weights[i]
            self.global_model.set_weights(current_global_weights)
            print(f"Round {r+1} complete. Global model updated.")

def main():
    X = np.load('data/ecg_segments.npy')
    y = np.load('data/ecg_labels.npy')
    X = np.expand_dims(X, axis=-1)
    num_clients = 10
    X_shards = np.array_split(X, num_clients)
    y_shards = np.array_split(y, num_clients)
    model_template = create_resnet9_model(X.shape[1:], len(np.unique(y)))
    clients = [ECGClient(i, X_shards[i], y_shards[i], model_template) for i in range(num_clients)]
    server = FederatedServer(model_template)
    server.run_federated_rounds(clients)
    server.global_model.save('feel_ecg_federated_model.h5')
    print("Final federated model saved as 'feel_ecg_federated_model.h5'")

if __name__ == '__main__':
    main()
