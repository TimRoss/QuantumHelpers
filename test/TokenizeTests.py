import numpy as np
import unittest
import TestHelpers

TestHelpers.addSourceDirectoryToPath()

import QuantumHelpers as qh

class TokenizeTests(unittest.TestCase):
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