# logistic_regression.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data():
    """Load the Iris dataset and split it into train and test sets."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return the accuracy score."""
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
