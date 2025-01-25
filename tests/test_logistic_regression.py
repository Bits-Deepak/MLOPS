# tests/test_logistic_regression.py
import unittest
from logistic_regression import load_data, train_model, evaluate_model

class TestLogisticRegression(unittest.TestCase):
    def test_data_loading(self):
        """Test if the data loads correctly."""
        X_train, X_test, y_train, y_test = load_data()
        self.assertEqual(X_train.shape[0], 120)  # 80% of 150 samples for training
        self.assertEqual(X_test.shape[0], 30)   # 20% of 150 samples for testing

    def test_model_training(self):
        """Test if the model trains without errors."""
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)  # Ensure model is trained

    def test_model_evaluation(self):
        """Test if the model achieves reasonable accuracy."""
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        self.assertGreater(accuracy, 0.7)  # Accuracy should be above 70%

if __name__ == "__main__":
    unittest.main()
