import sys
import os
import numpy as np
import unittest

# Add src directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_dir)

import QuantumHelpers as qh


class TestQuantumHelpers(unittest.TestCase):
    sqrtSymbol = "\u221A"
    def test_findFraction(self):
        self.assertEqual((1,1), qh.findFraction(1))

        self.assertEqual((1,2), qh.findFraction(1/2))

        self.assertEqual((1,3), qh.findFraction(1/3))
        self.assertEqual((2,3), qh.findFraction(2/3))

        self.assertEqual((1,4), qh.findFraction(1/4))
        self.assertEqual((1,2), qh.findFraction(2/4))
        self.assertEqual((3,4), qh.findFraction(3/4))

        self.assertEqual((1,5), qh.findFraction(1/5))
        self.assertEqual((2,5), qh.findFraction(2/5))
        self.assertEqual((3,5), qh.findFraction(3/5))
        self.assertEqual((4,5), qh.findFraction(4/5))
        self.assertEqual((1,1), qh.findFraction(5/5))

        self.assertEqual((1,6), qh.findFraction(1/6))
        self.assertEqual((1,3), qh.findFraction(2/6))
        self.assertEqual((1,2), qh.findFraction(3/6))
        self.assertEqual((2,3), qh.findFraction(4/6))
        self.assertEqual((5,6), qh.findFraction(5/6))
        self.assertEqual((1,1), qh.findFraction(6/6))

        self.assertEqual((1,7), qh.findFraction(1/7))
        self.assertEqual((2,7), qh.findFraction(2/7))
        self.assertEqual((3,7), qh.findFraction(3/7))
        self.assertEqual((4,7), qh.findFraction(4/7))
        self.assertEqual((5,7), qh.findFraction(5/7))
        self.assertEqual((6,7), qh.findFraction(6/7))
        self.assertEqual((1,1), qh.findFraction(7/7))

        self.assertEqual((1,8), qh.findFraction(1/8))
        self.assertEqual((1,4), qh.findFraction(2/8))
        self.assertEqual((3,8), qh.findFraction(3/8))
        self.assertEqual((1,2), qh.findFraction(4/8))
        self.assertEqual((5,8), qh.findFraction(5/8))
        self.assertEqual((3,4), qh.findFraction(6/8))
        self.assertEqual((7,8), qh.findFraction(7/8))
        self.assertEqual((1,1), qh.findFraction(8/8))

        # Make sure negatives are supported
        self.assertEqual((-1, 2), qh.findFraction(-1/2))

    def test_findFractionComplex(self):
        self.assertEqual((1, 2, 3, 4), qh.findFraction(1/2 + 3j/4))
        self.assertEqual((1, 2, 0, 0), qh.findFraction(1/2 + 0j))
        self.assertEqual((0, 0, 1, 1), qh.findFraction(0 + 1j))
        self.assertEqual((1, 2, 3, 4), qh.findFraction(1/2 + 3j/4))

    def test_prettyFraction(self):
        self.assertEqual("1", qh.prettyFraction(1))
        self.assertEqual("1/2", qh.prettyFraction(1/2))
        self.assertEqual("0", qh.prettyFraction(0))
        self.assertEqual("0", qh.prettyFraction(0 + 0j))
        self.assertEqual("1/3", qh.prettyFraction(1/3))
        self.assertEqual("1/10", qh.prettyFraction(1/10))

    def test_printPrettyWaveFunctionAmplitude(self):
        self.assertEqual("1", qh.prettyWaveFunctionAmplitude(1))
        self.assertEqual("1/{s}2".format(s=self.sqrtSymbol), qh.prettyWaveFunctionAmplitude(1/np.sqrt(2)))
        self.assertEqual("1/{s}3".format(s=self.sqrtSymbol), qh.prettyWaveFunctionAmplitude(1/np.sqrt(3)))
        self.assertEqual("1/2", qh.prettyWaveFunctionAmplitude(1/2))
        self.assertEqual("{s}3/2".format(s=self.sqrtSymbol), qh.prettyWaveFunctionAmplitude(np.sqrt(3)/2))
        self.assertEqual("{s}7/{s}8".format(s=self.sqrtSymbol), qh.prettyWaveFunctionAmplitude(np.sqrt(7)/np.sqrt(8)))
        self.assertEqual("1/{s}2+1/{s}2j".format(s=self.sqrtSymbol), qh.prettyWaveFunctionAmplitude(1/np.sqrt(2) + 1j/np.sqrt(2)))

        #Make sure negatives are supported
        self.assertEqual("-1/{s}2".format(s=self.sqrtSymbol),qh. prettyWaveFunctionAmplitude(-1/np.sqrt(2)))

    def test_exponentiateMatrix(self):
        A = np.array([[1, -1],[2,4]])
        result = qh.exponentiateMatrix(A)
        expectedResult = np.array([[2*np.e**2 - np.e**3, np.e**2 - np.e**3],[-2*np.e**2 + 2*np.e**3, -1*np.e**2 + 2*np.e**3]])
        self.compareMatricies(expectedResult, result, places=7)

    def compareMatricies(self, a, b, places=0):
        if not (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
            if places == 0:
                self.assertEqual(a,b)
            else:
                self.assertAlmostEqual(a,b,places=places)
        elif(a.shape != b.shape):
            self.fail("Shapes do not match. " + str(a.shape) + " != " + str(b.shape))
        else:
            for row in range(a.shape[0]):
                self.compareMatricies(a[row],b[row], places=places)
    
    def test_toString(self):
        self.assertEqual("1/{s}2 |00> + 1/{s}2 |11>".format(s=self.sqrtSymbol), qh.toString(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])))

    def test_tokenizeSingle(self):
        testPsi = "|01>"
        expectedTokens = ["|01>"]
        self.tokenizeCompare(testPsi=testPsi, expectedTokens=expectedTokens)

    def test_tokenizeGates(self):
        testPsi = "HX|0>"
        expectedTokens = ["H","X", "|0>"]
        self.tokenizeCompare(testPsi=testPsi, expectedTokens=expectedTokens)

    def test_tokenizeAddition(self):
        testPsi = "0.434534|01> + 0.234253|00>"
        expectedTokens = ["0.434534", "|01>", "+", "0.234253", "|00>"]
        self.tokenizeCompare(testPsi=testPsi, expectedTokens=expectedTokens)

        testPsi = "0.434534|01>+0.234253|00>"
        self.tokenizeCompare(testPsi=testPsi, expectedTokens=expectedTokens)

    def test_tokenizeParens(self):
        testPsi = "0.704(|01> + |00>)"
        expectedTokens = ["0.704", "(", "|01>", "+", "|00>", ")"]
        self.tokenizeCompare(testPsi=testPsi, expectedTokens=expectedTokens)

    def test_tokenizeBellState(self):
        testPsi = "Cnot(HI)|00>"
        expectedTokens = ["Cnot", "(", "H", "I", ")", "|00>"]
        self.tokenizeCompare(testPsi=testPsi, expectedTokens=expectedTokens)

    def test_tokenizeSqrt(self):
        testPsi = "(1/√2)(|0>+|1>)"
        expectedTokens = ["(", "1", "/", "√", "2", ")", "(", "|0>", "+", "|1>", ")"]
        self.tokenizeCompare(testPsi=testPsi, expectedTokens=expectedTokens)

    def test_tokenizeExponentials(self):
        testPsi = "2Exp(2\u03c0)|0>"
        expectedTokens = ["2", "Exp", "(", "2", "\u03c0", ")", "|0>"]
        self.tokenizeCompare(testPsi=testPsi, expectedTokens=expectedTokens)
    
    def tokenizeCompare(self, testPsi, expectedTokens):
        rtnArray = qh.tokenizeWaveFunctionString(testPsi)
        self.assertEqual(len(rtnArray), len(expectedTokens), "Sizes not equal. Got: {t} Expected: {e}".format(t=rtnArray, e=expectedTokens))
        for i in range(len(expectedTokens)):
            self.assertEqual(rtnArray[i], expectedTokens[i])

    def test_BuildWaveFunctionSuperposition(self):
        testPsi = "{c}(|0> + |1>)".format(c=1/np.sqrt(2))
        tokens = qh.tokenizeWaveFunctionString(testPsi)
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([1/np.sqrt(2), 1/np.sqrt(2)]))

    def test_BuildWaveFunctionXGate(self):
        tokens = ['X', "|0>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([0,1]))
    
    def test_BuildWaveFunctionCnotGate(self):
        tokens = ['Cnot', "|00>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([1,0,0,0]))
        tokens = ['Cnot', "|10>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([0,0,0,1]))

    def test_BuildWaveFunctionFlipFirst(self):
        tokens = ["(", "X", "I", ")", "|00>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([0,0,1,0]))

    def test_BuildWaveFunctionFlipSecond(self):
        tokens = ["(", "I", "X", ")", "|00>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([0,1,0,0]))

    def test_BuildWaveFunctionBellState(self):
        tokens = ["Cnot", "(", "H", "I", ")", "|00>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)]))

    def test_BuildWaveFunctionSqrtSymbol(self):
        tokens = ["(", "1", "/", "√", "2", ")", "(", "|0>", "+", "|1>", ")"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([1/np.sqrt(2),1/np.sqrt(2)]))

    def test_BuildWaveFunctionSqrtWord(self):
        tokens = ["(", "1", "/", "Sr", "2", ")", "(", "|0>", "+", "|1>", ")"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([1/np.sqrt(2),1/np.sqrt(2)]))
    
    def test_evalKetBra(self):
        tokens = ["|0>", "<0|"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([[1,0],[0,0]]))

    def test_OperatorSubtract(self):
        tokens = ["I", "-", "I"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([[0,0],[0,0]]))

    def test_OperatorAdd(self):
        tokens = ["I", "+", "I"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([[2,0],[0,2]]))

    def test_buildWaveFunctionNestedParams(self):
        testPsiString = "((IX)|10>)|0>"
        tokens = qh.tokenizeWaveFunctionString(testPsiString)
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, qh.buildKet("|110>").data)

    def test_buildWaveFunctionExponential(self):
        tokens = ["(", "Exp", "2", ")", "|0>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.e**2 * qh.buildKet("|0>").data)

    def test_buildWaveFunctionComplexScalar(self):
        tokens = [str(np.pi * 1j), "|0>"]
        rtnPsi = qh.buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi.data, np.array([np.pi * 1j, 0]))

    def test_buildWaveFunctionProb(self):
        testPsiString = "Prob(<0|((1/Sr(2))(|0> + |1>)))"
        tokens = qh.tokenizeWaveFunctionString(testPsiString)
        rtnProb = qh.buildWaveFunction(tokens)
        self.assertEqual(qh.WaveFunctionTokens.SCALAR, rtnProb.type)
        self.assertAlmostEqual(0.5, rtnProb.data)

    def test_QEAddBraBra(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.BRA)
        y = qh.QuantumElement(np.array([0,1]), qh.WaveFunctionTokens.BRA)
        z = x + y
        self.compareMatricies(z.data, np.array([1,1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.BRA)
    
    def test_QEAddKetKet(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.KET)
        y = qh.QuantumElement(np.array([0,1]), qh.WaveFunctionTokens.KET)
        z = x + y
        self.compareMatricies(z.data, np.array([1,1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.KET)
    
    def test_QEAddOpOp(self):
        x = qh.QuantumElement(np.array([[1,0],[0,-1]]), qh.WaveFunctionTokens.OPERATOR)
        y = qh.QuantumElement(np.array([[1,0],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        z = x + y
        self.compareMatricies(z.data, np.array([[2,0],[0,0]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEAddMismatch(self):
        x = qh.QuantumElement(np.array([[1,0]]), qh.WaveFunctionTokens.KET)
        y = qh.QuantumElement(np.array([[1,0],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        z = x + y
        self.assertEqual(z, None)

    def test_QESubBraBra(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.BRA)
        y = qh.QuantumElement(np.array([0,1]), qh.WaveFunctionTokens.BRA)
        z = x - y
        self.compareMatricies(z.data, np.array([1,-1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.BRA)
    
    def test_QESubKetKet(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.KET)
        y = qh.QuantumElement(np.array([0,1]), qh.WaveFunctionTokens.KET)
        z = x - y
        self.compareMatricies(z.data, np.array([1,-1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.KET)
    
    def test_QESubOpOp(self):
        x = qh.QuantumElement(np.array([[1,0],[0,-1]]), qh.WaveFunctionTokens.OPERATOR)
        y = qh.QuantumElement(np.array([[1,0],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        z = x - y
        self.compareMatricies(z.data, np.array([[0,0],[0,-2]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QESubMismatch(self):
        x = qh.QuantumElement(np.array([[1,0]]), qh.WaveFunctionTokens.KET)
        y = qh.QuantumElement(np.array([[1,0],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        z = x - y
        self.assertEqual(z, None)

    def test_QEMulOpFloat(self):
        x = qh.QuantumElement(np.array([[1,0],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        z = x * 5
        self.compareMatricies(z.data, np.array([[5,0],[0,5]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEMulBraBra(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.BRA)
        y = qh.QuantumElement(np.array([0,1]), qh.WaveFunctionTokens.BRA)
        z = x * y
        self.compareMatricies(z.data, np.array([0,1,0,0]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.BRA)

    def test_QEMulBraKet(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.BRA)
        y = qh.QuantumElement(np.array([1/np.sqrt(2),1/np.sqrt(2)]), qh.WaveFunctionTokens.KET)
        z = x * y
        self.assertEqual(z.data, 1/np.sqrt(2))
        self.assertEqual(z.type, qh.WaveFunctionTokens.SCALAR)
    
    def test_QEMulBraOp(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.BRA)
        y = qh.QuantumElement(np.array([[1,3],[2,1]]), qh.WaveFunctionTokens.OPERATOR)
        z = x * y
        self.compareMatricies(z.data, np.array([1,3]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.BRA)

    def test_QEMulKetBra(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.KET)
        y = qh.QuantumElement(np.array([0,1]), qh.WaveFunctionTokens.BRA)
        z = x * y
        self.compareMatricies(z.data, np.array([[0,1],[0,0]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEMulKetKet(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.KET)
        y = qh.QuantumElement(np.array([0,1]), qh.WaveFunctionTokens.KET)
        z = x * y
        self.compareMatricies(z.data, np.array([0,1,0,0]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.KET)
    
    def test_QEOpKet(self):
        x = qh.QuantumElement(np.array([[0,1],[1,0]]), qh.WaveFunctionTokens.OPERATOR)
        y = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.KET)
        z = x * y
        self.compareMatricies(z.data, np.array([0,1]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.KET)

    def test_QEMulOpOp(self):
        x = qh.QuantumElement(np.array([[1,2],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        y = qh.QuantumElement(np.array([[1,3],[2,1]]), qh.WaveFunctionTokens.OPERATOR)
        z = x * y
        self.compareMatricies(z.data, np.array([[5,5],[2,1]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEdivOpFloat(self):
        x = qh.QuantumElement(np.array([[1,0],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        y = np.sqrt(2)
        z = x / y
        self.compareMatricies(z.data, np.array([[1/np.sqrt(2),0],[0,1/np.sqrt(2)]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEkronOpOp(self):
        x = qh.QuantumElement(np.array([[1,0],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        y = qh.QuantumElement(np.array([[0,1],[1,0]]), qh.WaveFunctionTokens.OPERATOR)
        z = x & y
        self.compareMatricies(z.data, np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]))
        self.assertEqual(z.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEToString(self):
        x = qh.eval("<0|((1/Sr(2))(|0> + |1>))")
        y = qh.eval("Prob(<0|((1/Sr(2))(|0> + |1>)))")
        z = qh.eval("H|0>")
        w = qh.eval("<0|")

        self.assertEqual(str(x), "1/{s}2".format(s=self.sqrtSymbol))
        self.assertEqual(str(y), "1/2")
        self.assertEqual(str(z), "[[1/{s}2]\n [1/{s}2]]".format(s=self.sqrtSymbol))
        self.assertEqual(str(w), "[1 0]")

    def test_HadamardInRotation(self):
        x = qh.eval("H|0>")
        y = qh.eval("-1j(Rz(π))(Ry(π/2))|0>")

        self.compareMatricies(x.data, y.data, places=7)

    def test_QEDaggerKet(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.KET)
        x = x.dagger()
        self.compareMatricies(x.data, np.array([1,0]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.BRA)

    def test_QEDaggerKetImag(self):
        x = qh.QuantumElement(np.array([1j,0]), qh.WaveFunctionTokens.KET)
        x = x.dagger()
        self.compareMatricies(x.data, np.array([-1j,0]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.BRA)

    def test_QEDaggerBra(self):
        x = qh.QuantumElement(np.array([1,0]), qh.WaveFunctionTokens.BRA)
        x = x.dagger()
        self.compareMatricies(x.data, np.array([1,0]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.KET)

    def test_QEDaggerOp(self):
        x = qh.QuantumElement(np.array([[1,1],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        x = x.dagger()
        self.compareMatricies(x.data, np.array([[1,0],[1,1]]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.OPERATOR)

    def test_QEDaggerOpImag(self):
        x = qh.QuantumElement(np.array([[1,1j],[0,1]]), qh.WaveFunctionTokens.OPERATOR)
        x = x.dagger()
        self.compareMatricies(x.data, np.array([[1,0],[-1j,1]]))
        self.assertEqual(x.type, qh.WaveFunctionTokens.OPERATOR)

    def test_MakeControlGateCX122(self):
        cx = qh.makeControlGate(1, 2, "X", 2)
        inputs = ['|00>', '|01>', '|10>', '|11>']
        expected = ['|00>', '|01>', '|11>', '|10>']
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            self.compareMatricies(result.data, y.data)

    def test_MakeControlGateCX123(self):
        cx = qh.makeControlGate(1, 2, "X", 3)
        inputs = ['|000>', '|001>', '|010>', '|011>', '|100>', '|101>', '|110>', '|111>']
        expected = ['|000>', '|001>', '|010>', '|011>', '|110>', '|111>', '|100>', '|101>']
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            self.compareMatricies(result.data, y.data)

    def test_MakeControlGateCX133(self):
        cx = qh.makeControlGate(1, 3, "X", 3)
        inputs = ['|000>', '|001>', '|010>', '|011>', '|100>', '|101>', '|110>', '|111>']
        expected = ['|000>', '|001>', '|010>', '|011>', '|101>', '|100>', '|111>', '|110>']
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            self.compareMatricies(result.data, y.data)

    def test_MakeControlGateCX213(self):
        cx = qh.makeControlGate(2, 1, "X", 3)
        inputs = ['|000>', '|001>', '|010>', '|011>', '|100>', '|101>', '|110>', '|111>']
        expected = ['|000>', '|001>', '|110>', '|111>', '|100>', '|101>', '|010>', '|011>']
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            self.compareMatricies(result.data, y.data)

    def test_MakeControlGateCX214(self):
        cx = qh.makeControlGate(3, 2, "X", 4)
        inputs = ['|0000>', '|0001>', '|0010>', '|0011>', '|0100>', '|0101>', '|0110>', '|0111>',
        '|1000>', '|1001>', '|1010>', '|1011>', '|1100>', '|1101>', '|1110>', '|1111>']
        expected = ['|0000>', '|0001>', '|0110>', '|0111>', '|0100>', '|0101>', '|0010>', '|0011>',
        '|1000>', '|1001>', '|1110>', '|1111>', '|1100>', '|1101>', '|1010>', '|1011>']
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            self.compareMatricies(result.data, y.data)

# Run unit tests if run as a script
if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
