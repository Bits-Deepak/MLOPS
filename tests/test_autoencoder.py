import unittest
import numpy as np
from Autoencoder import autoencoder, encoder

class TestAutoencoder(unittest.TestCase):
    def test_autoencoder_training(self):
        """Test if the autoencoder trains without errors for 1 epoch."""
        # Use a small subset of data for testing
        input_data = np.random.rand(2, 784)  # 2 samples, 784 features each
        autoencoder.fit(input_data, input_data, epochs=1, batch_size=32, verbose=0)

    def test_encoder_output_shape(self):
        """Test if the encoder produces the correct output shape."""
        input_data = np.random.rand(1, 784)  # One sample with 784 features
        encoded_output = encoder.predict(input_data)
        self.assertEqual(encoded_output.shape, (1, 32))  # Expect latent space of size 32

if __name__ == "__main__":
    unittest.main()
