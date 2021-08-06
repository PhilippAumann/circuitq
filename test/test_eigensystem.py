import unittest
import numpy as np
import os

class TestCircuitQ(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.system("python create_test_data.py")

    def setUp(self):
        main_dir = os.path.abspath('..')
        data_path = os.path.join(main_dir, 'test/unittest_data.npy')
        benchmark_data_path = os.path.join(main_dir, 'test/unittest_data_benchmark.npy')
        self.test_data = np.load(data_path, allow_pickle=True)
        self.benchmark_data = np.load(benchmark_data_path, allow_pickle=True)

    def test_eigenenergies(self):
        print("Test eigenenergies of LC-Circuit, Transmon, Fluxonium and Flux Qubit")
        for n, evals in enumerate(self.test_data[0]):
            np.testing.assert_almost_equal(evals, self.benchmark_data[0][n], decimal=25)
            #self.assertAlmostEqual(evals, self.benchmark_data[0][n], delta=1e-25)

    def test_eigenstates(self):
        print("Test eigenstates of LC-Circuit, Transmon, Fluxonium and Flux Qubit")
        for n, estates in enumerate(self.test_data[1]):
            for l, k in enumerate(range(estates.shape[1])):
                if l > 10:
                    break
                np.testing.assert_almost_equal(np.real(estates[:,k]*np.conjugate(estates[:,k])),
                    np.real(self.benchmark_data[1][n][:,k]*np.conjugate(self.benchmark_data[1][n][:,k])),
                                                   decimal=3)
            # self.assertAlmostEqual(evals, self.benchmark_data[1][n], places=5)

if __name__ == '__main__':
    unittest.main()