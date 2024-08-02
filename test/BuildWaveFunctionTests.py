import numpy as np
import unittest
import TestHelpers

TestHelpers.addSourceDirectoryToPath()

import QuantumHelpers as qh


class BuildWaveFunctionTests(unittest.TestCase):
    def test_BuildWaveFunctionSuperposition(self):
        test_psi = "{c}(|0> + |1>)".format(c=1 / np.sqrt(2))
        tokens = qh.tokenize_wave_function_string(test_psi)
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(
            self, rtn_psi.data, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        )

    def test_BuildWaveFunctionXGate(self):
        tokens = ["X", "|0>"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([0, 1]))

    def test_BuildWaveFunctionCnotGate(self):
        tokens = ["Cnot", "|00>"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([1, 0, 0, 0]))
        tokens = ["Cnot", "|10>"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([0, 0, 0, 1]))

    def test_BuildWaveFunctionFlipFirst(self):
        tokens = ["(", "X", "I", ")", "|00>"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([0, 0, 1, 0]))

    def test_BuildWaveFunctionFlipSecond(self):
        tokens = ["(", "I", "X", ")", "|00>"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([0, 1, 0, 0]))

    def test_BuildWaveFunctionBellState(self):
        tokens = ["Cnot", "*", "H", "I", "|00>"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(
            self, rtn_psi.data, np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
        )

    def test_BuildWaveFunctionSqrtSymbol(self):
        tokens = ["(", "1", "/", "√", "(", "2", ")", ")", "(", "|0>", "+", "|1>", ")"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(
            self, rtn_psi.data, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        )

    def test_BuildWaveFunctionSqrtWord(self):
        tokens = ["(", "1", "/", "Sr", "(", "2", ")", ")", "(", "|0>", "+", "|1>", ")"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(
            self, rtn_psi.data, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        )

    def test_evalKetBra(self):
        tokens = ["|0>", "<0|"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([[1, 0], [0, 0]]))

    def test_OperatorSubtract(self):
        tokens = ["I", "-", "I"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([[0, 0], [0, 0]]))

    def test_OperatorAdd(self):
        tokens = ["I", "+", "I"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([[2, 0], [0, 2]]))

    def test_buildWaveFunctionNestedParams(self):
        test_psi_string = "((IX)|10>)|0>"
        tokens = qh.tokenize_wave_function_string(test_psi_string)
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, qh.build_ket("|110>").data)

    def test_buildWaveFunctionExponential(self):
        test_psi_string = "(H * (Exp(-1j * (Pi/2) * Z)) * H) |0>"
        tokens = qh.tokenize_wave_function_string(test_psi_string)
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(
            self, rtn_psi.data, qh.eval("-1j * |1>").data, places=5
        )

    def test_buildWaveFunctionComplexScalar(self):
        tokens = [str(np.pi * 1j), "|0>"]
        rtn_psi = qh.build_wave_function(tokens)
        TestHelpers.compareMatricies(self, rtn_psi.data, np.array([np.pi * 1j, 0]))

    def test_buildWaveFunctionProb(self):
        test_psi_string = "Prob(<0|((1/Sr(2))(|0> + |1>)))"
        tokens = qh.tokenize_wave_function_string(test_psi_string)
        rtn_prob = qh.build_wave_function(tokens)
        self.assertEqual(qh.WaveFunctionTokens.SCALAR, rtn_prob.type)
        self.assertAlmostEqual(0.5, rtn_prob.data)

    def test_buildControlMatrix(self):
        test_psi = qh.eval("Ctrl(1,2,X,2)")
        cnot = qh.eval("Cnot")
        TestHelpers.compareMatricies(self, test_psi.data , cnot.data)

    def test_kronMatriciesFirst(self):
        # Verify that operators will be kroned together instead of evaluating wavefunction from right to left
        try:
            qh.eval("HH |00>")
        except:
            self.fail("Operators were probably not kroned first")

    def test_insertingElements(self):
        x_1 = qh.eval("XI")
        psi = qh.eval("? |00>", x_1)
        TestHelpers.compareMatricies(self, psi.data, qh.eval("XI |00>").data)

    def test_insertingElements2(self):
        x = qh.eval("X")
        psi = qh.eval("?? |00>", x, x)
        TestHelpers.compareMatricies(self, psi.data, qh.eval("XX |00>").data)
    
    def test_validKet(self):
        x = qh.eval("|0>")
        self.assertTrue(x.valid)

    def test_invalidKets(self):
        bad_type = qh.WaveFunctionElement([1,0], qh.WaveFunctionTokens.KET)
        self.assertFalse(bad_type.valid)
        bad_length = qh.WaveFunctionElement(np.array([1,0,0]), qh.WaveFunctionTokens.KET)
        self.assertFalse(bad_length.valid)
        bad_dimensions = qh.WaveFunctionElement(np.array([[1,0],[1,0]]), qh.WaveFunctionTokens.KET)
        self.assertFalse(bad_dimensions.valid)

    def test_validOperator(self):
        x = qh.eval("H")
        self.assertTrue(x.valid)

    def test_invalidOperators(self):
        bad_type = qh.WaveFunctionElement([1,0], qh.WaveFunctionTokens.OPERATOR)
        self.assertFalse(bad_type.valid)
        bad_length = qh.WaveFunctionElement(np.array([[1,0,0],[1,0,0]]), qh.WaveFunctionTokens.OPERATOR)
        self.assertFalse(bad_length.valid)
        non_square = qh.WaveFunctionElement(np.array([[1,0],[1,0],[1,0],[1,0]]), qh.WaveFunctionTokens.OPERATOR)
        self.assertFalse(non_square.valid)
        bad_dimensions = qh.WaveFunctionElement(np.array([1,0]), qh.WaveFunctionTokens.OPERATOR)
        self.assertFalse(bad_dimensions.valid)
        
        
