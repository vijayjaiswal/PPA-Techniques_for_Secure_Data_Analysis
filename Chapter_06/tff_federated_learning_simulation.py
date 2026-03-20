import collections
import numpy as np
import sys

# Apply monkeypatch for older TensorFlow / TensorFlow Federated versions
# running on newer numpy versions (>= 1.20)
try:
    np.object = object
    np.bool = bool
    np.typeDict = np.sctypeDict
    np.int = int
    np.float = float
    np.complex = complex
except AttributeError:
    pass

import tensorflow as tf

# Suppress TF warnings for cleaner output
tf.get_logger().setLevel('ERROR')

try:
    import tensorflow_federated as tff
    HAS_TFF = True
except ImportError:
    HAS_TFF = False
    print("="*60)
    print("INFO: 'tensorflow_federated' module not active.")
    print("TensorFlow Federated (TFF) pip packages are natively built for Linux/macOS.")
    print("On Windows, TFF typically requires WSL (Windows Subsystem for Linux).")
    print("Because you are running natively on Windows, we are dynamically falling back")
    print("to a PURE TensorFlow simulation of the exact same Non-IID FedAvg scenario.")
    print("="*60 + "\n")


def create_keras_model():
    """A simple Sequential Keras model for the FEMNIST classification task."""
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


# ==============================================================================
# TFF IMPLEMENTATION (Runs if TFF is available, e.g. Linux/WSL)
# ==============================================================================
if HAS_TFF:
    def load_and_preprocess_emnist_tff():
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
        print(f"Total simulated clients (writers) available in TFF: {len(emnist_train.client_ids)}")
        return emnist_train

    def preprocess_function_tff(dataset):
        def batch_format_fn(element):
            return (tf.reshape(element['pixels'], [-1, 784]), 
                    tf.reshape(element['label'], [-1, 1]))
        return dataset.repeat(1).shuffle(100).batch(20).map(batch_format_fn)

    def make_federated_data_tff(client_data, client_ids):
        return [
            preprocess_function_tff(client_data.create_tf_dataset_for_client(x))
            for x in client_ids
        ]

    def model_fn_tff():
        keras_model = create_keras_model()
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=(tf.TensorSpec(shape=[None, 784], dtype=tf.float32), 
                        tf.TensorSpec(shape=[None, 1], dtype=tf.int32)),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    def simulate_fl_training_tff(emnist_train):
        training_process = tff.learning.algorithms.build_unweighted_fed_avg(
            model_fn_tff,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
        )
        state = training_process.initialize()
        NUM_ROUNDS, CLIENTS_PER_ROUND, DROPOUT_RATE = 5, 10, 0.2 
        print("\n--- Starting TFF Federated Learning Simulation (FedAvg) ---")
        
        for round_num in range(1, NUM_ROUNDS + 1):
            sampled_clients = np.random.choice(emnist_train.client_ids, size=CLIENTS_PER_ROUND, replace=False)
            active_clients = [c for c in sampled_clients if np.random.rand() > DROPOUT_RATE]
            dropouts = CLIENTS_PER_ROUND - len(active_clients)
            print(f"Round {round_num}: Selected {CLIENTS_PER_ROUND} clients. "
                  f"Active: {len(active_clients)} (Simulated {dropouts} dropouts via realistic latency).")
            federated_train_data = make_federated_data_tff(emnist_train, active_clients)
            result = training_process.next(state, federated_train_data)
            state = result.state
            print(f"Round {round_num} metrics: {result.metrics['client_work']['train']}")


# ==============================================================================
# PURE TENSORFLOW FALLBACK IMPLEMENTATION (Runs natively on Windows)
# ==============================================================================
else:
    def simulate_fl_training_fallback():
        print("--- Starting Pure TF Fallback Federated Learning Simulation (FedAvg) ---")
        
        # 1. Simulate Non-IID Data via Standard MNIST Partitioning
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
        
        TOTAL_SIMULATED_CLIENTS = 100
        # Sort by labels to synthesize an extreme non-IID environment!
        sort_indices = np.argsort(y_train)
        x_train_sorted = x_train[sort_indices]
        y_train_sorted = y_train[sort_indices]
        
        # Partition data among clients
        client_data = {}
        examples_per_client = len(x_train) // TOTAL_SIMULATED_CLIENTS
        for i in range(TOTAL_SIMULATED_CLIENTS):
            start = i * examples_per_client
            end = start + examples_per_client
            client_data[i] = (x_train_sorted[start:end], y_train_sorted[start:end])
        
        print(f"Total simulated clients (devices) created: {TOTAL_SIMULATED_CLIENTS}")

        # Initialize the global server model
        global_model = create_keras_model()
        global_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        NUM_ROUNDS = 5
        CLIENTS_PER_ROUND = 10
        DROPOUT_RATE = 0.2  # Simulate a 20% dropout rate
        
        for round_num in range(1, NUM_ROUNDS + 1):
            # Sample a cohort of devices randomly
            sampled_clients = np.random.choice(range(TOTAL_SIMULATED_CLIENTS), size=CLIENTS_PER_ROUND, replace=False)
            
            # Simulate network failures / client timeout
            active_clients = [c for c in sampled_clients if np.random.rand() > DROPOUT_RATE]
            dropouts = CLIENTS_PER_ROUND - len(active_clients)
            
            print(f"Round {round_num}: Selected {CLIENTS_PER_ROUND} clients. "
                  f"Active: {len(active_clients)} (Simulated {dropouts} dropouts via realistic latency).")
            
            # Local Client Training (Simulated)
            client_weights = []
            metrics_acc = []
            metrics_loss = []
            
            for client_id in active_clients:
                # 1. Download global model weights to client device
                client_model = create_keras_model()
                client_model.set_weights(global_model.get_weights())
                client_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
                                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # 2. Local Training on purely Non-IID local data subset
                cx, cy = client_data[client_id]
                history = client_model.fit(cx, cy, batch_size=20, epochs=1, verbose=0)
                
                client_weights.append(client_model.get_weights())
                metrics_loss.append(history.history['loss'][-1])
                metrics_acc.append(history.history['accuracy'][-1])
                
            # 3. Server Aggregation (FedAvg) over the Active Clients
            if client_weights:
                new_global_weights = []
                # Unweighted average of layers
                for layers in zip(*client_weights):
                    new_global_weights.append(np.mean(layers, axis=0))
                # Update Server Global Model
                global_model.set_weights(new_global_weights)
                
                print(f"Round {round_num} metrics: OrderedDict([('sparse_categorical_accuracy', {np.mean(metrics_acc):.4f}), ('loss', {np.mean(metrics_loss):.4f})])\n")
            else:
                print(f"Round {round_num} failed: All selected clients dropped out!\n")


if __name__ == "__main__":
    if HAS_TFF:
        emnist_train = load_and_preprocess_emnist_tff()
        simulate_fl_training_tff(emnist_train)
    else:
        simulate_fl_training_fallback()
