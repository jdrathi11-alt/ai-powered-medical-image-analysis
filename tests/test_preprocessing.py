import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - gives a useful local setup message
    np = None

@unittest.skipIf(np is None, "numpy is required for preprocessing tests")
class PreprocessingTests(unittest.TestCase):
    def test_preprocessing_returns_normalized_channel_last_batch(self):
        from src.preprocessing import preprocess_images

        images = np.zeros((2, 12, 10), dtype=np.uint8)
        images[0, 2:8, 2:8] = 255
        result = preprocess_images(images, image_size=16)

        self.assertEqual(result.shape, (2, 16, 16, 1))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)
        self.assertAlmostEqual(float(result[0].max()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
