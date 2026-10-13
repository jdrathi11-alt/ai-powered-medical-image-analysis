import unittest

from src.labels import CLASS_NAMES, decode_label, encode_label


class LabelTests(unittest.TestCase):
    def test_dataset_labels_are_stable(self):
        self.assertEqual(CLASS_NAMES, ("normal", "pneumonia"))
        self.assertEqual(decode_label(0), "normal")
        self.assertEqual(decode_label(1), "pneumonia")

    def test_label_names_round_trip(self):
        for expected_id, name in enumerate(CLASS_NAMES):
            self.assertEqual(encode_label(name), expected_id)
            self.assertEqual(decode_label(encode_label(name)), name)

    def test_unknown_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_label(2)
        with self.assertRaises(ValueError):
            encode_label("fracture")


if __name__ == "__main__":
    unittest.main()
