import unittest
from Aphopis_Armillien.main import Guass_8th_seed

class TestGuass8thSeed(unittest.TestCase):
    def test_seed_return(self):
        # Call the Guass_8th_seed function and verify it returns a value.
        result = Guass_8th_seed()
        self.assertIsNotNone(result)
        # You can add further assertions here based on the expected behavior of the function.
        # For example, if Guass_8th_seed() should return an integer:
        # self.assertIsInstance(result, int)

if __name__ == "__main__":
    unittest.main()