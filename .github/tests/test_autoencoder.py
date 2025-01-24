import unittest
import numpy as np
from Autoencoder import encoder

class TestAutoencoder(unittest.TestCase):
    def test_encoder_output_shape(self):
        """Test if the encoder produces the correct output shape"""
        input_data = np.random.rand(1, 784)  # Random input with 784 features
        encoded_output = encoder.predict(input_data)
        self.assertEqual(encoded_output.shape, (1, 32))  # Expecting latent space of size 32

    def test_encoder_dtype(self):
        """Test if the encoder output is of type float"""
        input_data = np.random.rand(1, 784)
        encoded_output = encoder.predict(input_data)
        self.assertTrue(np.issubdtype(encoded_output.dtype, np.float))

if __name__ == "__main__":
    unittest.main()
