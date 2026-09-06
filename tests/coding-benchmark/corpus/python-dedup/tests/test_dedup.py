import unittest

from dedup import dedup


class DedupTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(dedup([]), [])

    def test_no_duplicates(self):
        self.assertEqual(dedup(["a", "b", "c"]), ["a", "b", "c"])

    def test_adjacent_duplicates_removed(self):
        self.assertEqual(dedup(["x", "x", "y"]), ["x", "y"])

    def test_non_adjacent_duplicates_removed(self):
        self.assertEqual(dedup(["a", "b", "a", "c", "b"]), ["a", "b", "c"])

    def test_first_occurrence_order_kept(self):
        self.assertEqual(dedup([1, 2, 1, 3, 2, 1, 4]), [1, 2, 3, 4])

    def test_input_list_not_mutated(self):
        src = ["a", "b", "a"]
        dedup(src)
        self.assertEqual(src, ["a", "b", "a"])


if __name__ == "__main__":
    unittest.main()
