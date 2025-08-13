import unittest

from script_2 import format_name

class TestFormatNameFunction(unittest.TestCase):
    def test_first_last_only(self):
        self.assertEqual(format_name("John", "Doe"), "Doe, John")

    def test_first_middle_last(self):
        self.assertEqual(format_name("Jane", "Smith", "Alice"), "Smith, Jane A.")

    def test_empty_names(self):
        self.assertEqual(format_name("", ""), "")
        self.assertEqual(format_name("John", ""), ", John") # Or define specific behavior
        self.assertEqual(format_name("", "Doe"), "Doe, ") # Or define specific behavior

    def test_middle_name_first_char_only(self):
        self.assertEqual(format_name("Alice", "Wonderland", "Marie"), "Wonderland, Alice M.")
        self.assertEqual(format_name("Bob", "Builder", "theGreat"), "Builder, Bob T.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)