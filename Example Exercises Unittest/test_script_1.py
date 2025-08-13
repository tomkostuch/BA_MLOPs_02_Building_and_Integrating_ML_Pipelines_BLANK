
import unittest
from script_1 import add

class TestAddFunction(unittest.TestCase):
    def test_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_negative_numbers(self):
        self.assertEqual(add(-1, -1), -2)

    def test_zero_with_number(self):
        self.assertEqual(add(0, 7), 7)

if __name__ == '__main__':
    unittest.main()