import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, 1:].values  # Features (784 columns)
    y = df.iloc[:, 0].values  # Labels (first column)
    return X, y


# Plot 1: Training Losses over Epochs
def plot_losses(loss):
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        # Load training data
        train_file_path = r'C:\train.csv'
        X_train, y_train = load_data(train_file_path)

        # Load test data
        test_file_path = r'C:\test.csv'
        X_test, y_test = load_data(test_file_path)

        # Standardize the input values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # calc mean/std from training data then transform data
        X_test_scaled = scaler.transform(X_test)

        # Initialize the MLPClassifier with the ReLU activation function
        mlp = MLPClassifier(hidden_layer_sizes=(512,),  # Number of hidden layers and neurons per layer
                            activation='relu',  # Activation function
                            solver='adam',  # Adam optimizer
                            max_iter=200,  # Maximum number of iterations
                            alpha=0.0001,
                            learning_rate_init=0.0001,
                            tol=0.0001,
                            random_state=42)

        # Train model
        mlp.fit(X_train_scaled, y_train)
        loss_train = mlp.loss_curve_
        plot_losses(loss_train)

        # Make predictions on the test set
        y_pred = mlp.predict(X_test_scaled)

        # Generate classification report
        report = classification_report(y_test, y_pred, digits=2)
        print(report)

    except Exception as e:
        print(f"An error occurred: {e}")
