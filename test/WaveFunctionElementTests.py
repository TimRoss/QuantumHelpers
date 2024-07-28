import numpy as np
import unittest
import TestHelpers

TestHelpers.addSourceDirectoryToPath()

import QuantumHelpers as qh


class QuantumElementTests(unittest.TestCase):
    def test_QEAddBraBra(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.BRA)
        y = qh.WaveFunctionElement(np.array([0, 1]), qh.WaveFunctionTokens.BRA)
        z = x + y
        TestHelpers.compareMatricies(self, z.data, np.array([1, 1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.BRA)

    def test_QEAddKetKet(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.KET)
        y = qh.WaveFunctionElement(np.array([0, 1]), qh.WaveFunctionTokens.KET)
        z = x + y
        TestHelpers.compareMatricies(self, z.data, np.array([1, 1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.KET)

    def test_QEAddOpOp(self):
        x = qh.WaveFunctionElement(
            np.array([[1, 0], [0, -1]]), qh.WaveFunctionTokens.OPERATOR
        )
        y = qh.WaveFunctionElement(
            np.array([[1, 0], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        z = x + y
        TestHelpers.compareMatricies(self, z.data, np.array([[2, 0], [0, 0]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEAddMismatch(self):
        x = qh.WaveFunctionElement(np.array([[1, 0]]), qh.WaveFunctionTokens.KET)
        y = qh.WaveFunctionElement(
            np.array([[1, 0], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        z = x + y
        self.assertEqual(z.type, qh.WaveFunctionTokens.ERROR)

    def test_QESubBraBra(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.BRA)
        y = qh.WaveFunctionElement(np.array([0, 1]), qh.WaveFunctionTokens.BRA)
        z = x - y
        TestHelpers.compareMatricies(self, z.data, np.array([1, -1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.BRA)

    def test_QESubKetKet(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.KET)
        y = qh.WaveFunctionElement(np.array([0, 1]), qh.WaveFunctionTokens.KET)
        z = x - y
        TestHelpers.compareMatricies(self, z.data, np.array([1, -1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.KET)

    def test_QESubOpOp(self):
        x = qh.WaveFunctionElement(
            np.array([[1, 0], [0, -1]]), qh.WaveFunctionTokens.OPERATOR
        )
        y = qh.WaveFunctionElement(
            np.array([[1, 0], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        z = x - y
        TestHelpers.compareMatricies(self, z.data, np.array([[0, 0], [0, -2]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QESubMismatch(self):
        x = qh.WaveFunctionElement(np.array([[1, 0]]), qh.WaveFunctionTokens.KET)
        y = qh.WaveFunctionElement(
            np.array([[1, 0], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        z = x - y
        self.assertEqual(z.type, qh.WaveFunctionTokens.ERROR)

    def test_QEMulOpFloat(self):
        x = qh.WaveFunctionElement(
            np.array([[1, 0], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        z = x * 5
        TestHelpers.compareMatricies(self, z.data, np.array([[5, 0], [0, 5]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEMulBraBra(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.BRA)
        y = qh.WaveFunctionElement(np.array([0, 1]), qh.WaveFunctionTokens.BRA)
        z = x * y
        TestHelpers.compareMatricies(self, z.data, np.array([0, 1, 0, 0]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.BRA)

    def test_QEMulBraKet(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.BRA)
        y = qh.WaveFunctionElement(
            np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), qh.WaveFunctionTokens.KET
        )
        z = x * y
        self.assertEqual(z.data, 1 / np.sqrt(2))
        self.assertEqual(z.type, qh.WaveFunctionTokens.SCALAR)

    def test_QEMulBraOp(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.BRA)
        y = qh.WaveFunctionElement(
            np.array([[1, 3], [2, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        z = x * y
        TestHelpers.compareMatricies(self, z.data, np.array([1, 3]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.BRA)

    def test_QEMulKetBra(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.KET)
        y = qh.WaveFunctionElement(np.array([0, 1]), qh.WaveFunctionTokens.BRA)
        z = x * y
        TestHelpers.compareMatricies(self, z.data, np.array([[0, 1], [0, 0]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEMulKetKet(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.KET)
        y = qh.WaveFunctionElement(np.array([0, 1]), qh.WaveFunctionTokens.KET)
        z = x * y
        TestHelpers.compareMatricies(self, z.data, np.array([0, 1, 0, 0]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.KET)

    def test_QEOpKet(self):
        x = qh.WaveFunctionElement(
            np.array([[0, 1], [1, 0]]), qh.WaveFunctionTokens.OPERATOR
        )
        y = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.KET)
        z = x * y
        TestHelpers.compareMatricies(self, z.data, np.array([0, 1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.KET)

    def test_QEMulOpOp(self):
        x = qh.WaveFunctionElement(
            np.array([[1, 2], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        y = qh.WaveFunctionElement(
            np.array([[1, 3], [2, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        z = x * y
        TestHelpers.compareMatricies(self, z.data, np.array([[5, 5], [2, 1]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEdivOpFloat(self):
        x = qh.WaveFunctionElement(
            np.array([[1, 0], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        y = np.sqrt(2)
        z = x / y
        TestHelpers.compareMatricies(
            self, z.data, np.array([[1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2)]])
        )
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEkronOpOp(self):
        x = qh.WaveFunctionElement(
            np.array([[1, 0], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        y = qh.WaveFunctionElement(
            np.array([[0, 1], [1, 0]]), qh.WaveFunctionTokens.OPERATOR
        )
        z = x & y
        TestHelpers.compareMatricies(
            self,
            z.data,
            np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        )
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEToString(self):
        x = qh.eval("<0|((1/Sr(2))(|0> + |1>))")
        y = qh.eval("Prob(<0|((1/Sr(2))(|0> + |1>)))")
        z = qh.eval("H|0>")
        w = qh.eval("<0|")

        self.assertEqual(str(x), "1/{s}2".format(s=TestHelpers.sqrtSymbol))
        self.assertEqual(str(y), "1/2")
        self.assertEqual(
            str(z), "[[1/{s}2]\n [1/{s}2]]".format(s=TestHelpers.sqrtSymbol)
        )
        self.assertEqual(str(w), "[1 0]")

    def test_HadamardInRotation(self):
        x = qh.eval("H")
        y = qh.eval("(-1j(Rz(π))*(Ry(3 * π/2)))")

        TestHelpers.compareMatricies(self, x.data, y.data, places=7)

    def test_QEDaggerKet(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.KET)
        x = x.dagger()
        TestHelpers.compareMatricies(self, x.data, np.array([1, 0]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.BRA)

    def test_QEDaggerKetImag(self):
        x = qh.WaveFunctionElement(np.array([1j, 0]), qh.WaveFunctionTokens.KET)
        x = x.dagger()
        TestHelpers.compareMatricies(self, x.data, np.array([-1j, 0]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.BRA)

    def test_QEDaggerBra(self):
        x = qh.WaveFunctionElement(np.array([1, 0]), qh.WaveFunctionTokens.BRA)
        x = x.dagger()
        TestHelpers.compareMatricies(self, x.data, np.array([1, 0]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.KET)

    def test_QEDaggerOp(self):
        x = qh.WaveFunctionElement(
            np.array([[1, 1], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        x = x.dagger()
        TestHelpers.compareMatricies(self, x.data, np.array([[1, 0], [1, 1]]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEDaggerOpImag(self):
        x = qh.WaveFunctionElement(
            np.array([[1, 1j], [0, 1]]), qh.WaveFunctionTokens.OPERATOR
        )
        x = x.dagger()
        TestHelpers.compareMatricies(self, x.data, np.array([[1, 0], [-1j, 1]]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.OPERATOR)
