import numpy as np
import unittest
import TestHelpers

TestHelpers.addSourceDirectoryToPath()

import QuantumHelpers as qh


class BuildWaveFunctionTests(unittest.TestCase):
    def test_BuildWaveFunctionSuperposition(self):
        testPsi = "{c}(|0> + |1>)".format(c=1 / np.sqrt(2))
        tokens = qh.tokenizeWaveFunctionString(testPsi)
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(
            self, rtnPsi.data, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        )

    def test_BuildWaveFunctionXGate(self):
        tokens = ["X", "|0>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([0, 1]))

    def test_BuildWaveFunctionCnotGate(self):
        tokens = ["Cnot", "|00>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([1, 0, 0, 0]))
        tokens = ["Cnot", "|10>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([0, 0, 0, 1]))

    def test_BuildWaveFunctionFlipFirst(self):
        tokens = ["(", "X", "I", ")", "|00>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([0, 0, 1, 0]))

    def test_BuildWaveFunctionFlipSecond(self):
        tokens = ["(", "I", "X", ")", "|00>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([0, 1, 0, 0]))

    def test_BuildWaveFunctionBellState(self):
        tokens = ["Cnot", "(", "H", "I", ")", "|00>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(
            self, rtnPsi.data, np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
        )

    def test_BuildWaveFunctionSqrtSymbol(self):
        tokens = ["(", "1", "/", "√", "(", "2", ")", ")", "(", "|0>", "+", "|1>", ")"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(
            self, rtnPsi.data, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        )

    def test_BuildWaveFunctionSqrtWord(self):
        tokens = ["(", "1", "/", "Sr", "(", "2", ")", ")", "(", "|0>", "+", "|1>", ")"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(
            self, rtnPsi.data, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        )

    def test_evalKetBra(self):
        tokens = ["|0>", "<0|"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([[1, 0], [0, 0]]))

    def test_OperatorSubtract(self):
        tokens = ["I", "-", "I"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([[0, 0], [0, 0]]))

    def test_OperatorAdd(self):
        tokens = ["I", "+", "I"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([[2, 0], [0, 2]]))

    def test_buildWaveFunctionNestedParams(self):
        testPsiString = "((IX)|10>)|0>"
        tokens = qh.tokenizeWaveFunctionString(testPsiString)
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, qh.buildKet("|110>").data)

    def test_buildWaveFunctionExponential(self):
        testPsiString = "(H * (Exp(-1j * (Pi/2) * Z)) * H) |0>"
        tokens = qh.tokenizeWaveFunctionString(testPsiString)
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(
            self, rtnPsi.data, qh.eval("-1j * |1>").data, places=5
        )

    def test_buildWaveFunctionComplexScalar(self):
        tokens = [str(np.pi * 1j), "|0>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        TestHelpers.compareMatricies(self, rtnPsi.data, np.array([np.pi * 1j, 0]))

    def test_buildWaveFunctionProb(self):
        testPsiString = "Prob(<0|((1/Sr(2))(|0> + |1>)))"
        tokens = qh.tokenizeWaveFunctionString(testPsiString)
        rtnProb = qh.buildWaveFunction(tokens)
        self.assertEqual(qh.WaveFunctionTokens.SCALAR, rtnProb.type)
        self.assertAlmostEqual(0.5, rtnProb.data)
