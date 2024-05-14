import numpy as np
import unittest
import TestHelpers

TestHelpers.addSourceDirectoryToPath()

import QuantumHelpers as qh


class OtherTests(unittest.TestCase):
    def test_findFraction(self):
        self.assertEqual((1, 1), qh.findFraction(1))

        self.assertEqual((1, 2), qh.findFraction(1 / 2))

        self.assertEqual((1, 3), qh.findFraction(1 / 3))
        self.assertEqual((2, 3), qh.findFraction(2 / 3))

        self.assertEqual((1, 4), qh.findFraction(1 / 4))
        self.assertEqual((1, 2), qh.findFraction(2 / 4))
        self.assertEqual((3, 4), qh.findFraction(3 / 4))

        self.assertEqual((1, 5), qh.findFraction(1 / 5))
        self.assertEqual((2, 5), qh.findFraction(2 / 5))
        self.assertEqual((3, 5), qh.findFraction(3 / 5))
        self.assertEqual((4, 5), qh.findFraction(4 / 5))
        self.assertEqual((1, 1), qh.findFraction(5 / 5))

        self.assertEqual((1, 6), qh.findFraction(1 / 6))
        self.assertEqual((1, 3), qh.findFraction(2 / 6))
        self.assertEqual((1, 2), qh.findFraction(3 / 6))
        self.assertEqual((2, 3), qh.findFraction(4 / 6))
        self.assertEqual((5, 6), qh.findFraction(5 / 6))
        self.assertEqual((1, 1), qh.findFraction(6 / 6))

        self.assertEqual((1, 7), qh.findFraction(1 / 7))
        self.assertEqual((2, 7), qh.findFraction(2 / 7))
        self.assertEqual((3, 7), qh.findFraction(3 / 7))
        self.assertEqual((4, 7), qh.findFraction(4 / 7))
        self.assertEqual((5, 7), qh.findFraction(5 / 7))
        self.assertEqual((6, 7), qh.findFraction(6 / 7))
        self.assertEqual((1, 1), qh.findFraction(7 / 7))

        self.assertEqual((1, 8), qh.findFraction(1 / 8))
        self.assertEqual((1, 4), qh.findFraction(2 / 8))
        self.assertEqual((3, 8), qh.findFraction(3 / 8))
        self.assertEqual((1, 2), qh.findFraction(4 / 8))
        self.assertEqual((5, 8), qh.findFraction(5 / 8))
        self.assertEqual((3, 4), qh.findFraction(6 / 8))
        self.assertEqual((7, 8), qh.findFraction(7 / 8))
        self.assertEqual((1, 1), qh.findFraction(8 / 8))

        # Make sure negatives are supported
        self.assertEqual((-1, 2), qh.findFraction(-1 / 2))

    def test_findFractionComplex(self):
        self.assertEqual((1, 2, 3, 4), qh.findFraction(1 / 2 + 3j / 4))
        self.assertEqual((1, 2, 0, 0), qh.findFraction(1 / 2 + 0j))
        self.assertEqual((0, 0, 1, 1), qh.findFraction(0 + 1j))
        self.assertEqual((1, 2, 3, 4), qh.findFraction(1 / 2 + 3j / 4))

    def test_prettyFraction(self):
        self.assertEqual("1", qh.prettyFraction(1))
        self.assertEqual("1/2", qh.prettyFraction(1 / 2))
        self.assertEqual("0", qh.prettyFraction(0))
        self.assertEqual("0", qh.prettyFraction(0 + 0j))
        self.assertEqual("1/3", qh.prettyFraction(1 / 3))
        self.assertEqual("1/10", qh.prettyFraction(1 / 10))

    def test_printPrettyWaveFunctionAmplitude(self):
        self.assertEqual("1", qh.prettyWaveFunctionAmplitude(1))
        self.assertEqual(
            "1/{s}2".format(s=TestHelpers.sqrtSymbol),
            qh.prettyWaveFunctionAmplitude(1 / np.sqrt(2)),
        )
        self.assertEqual(
            "1/{s}3".format(s=TestHelpers.sqrtSymbol),
            qh.prettyWaveFunctionAmplitude(1 / np.sqrt(3)),
        )
        self.assertEqual("1/2", qh.prettyWaveFunctionAmplitude(1 / 2))
        self.assertEqual(
            "{s}3/2".format(s=TestHelpers.sqrtSymbol),
            qh.prettyWaveFunctionAmplitude(np.sqrt(3) / 2),
        )
        self.assertEqual(
            "{s}7/{s}8".format(s=TestHelpers.sqrtSymbol),
            qh.prettyWaveFunctionAmplitude(np.sqrt(7) / np.sqrt(8)),
        )
        self.assertEqual(
            "1/{s}2+1/{s}2j".format(s=TestHelpers.sqrtSymbol),
            qh.prettyWaveFunctionAmplitude(1 / np.sqrt(2) + 1j / np.sqrt(2)),
        )
        self.assertEqual(
            "1/{s}2-1/{s}2j".format(s=TestHelpers.sqrtSymbol),
            qh.prettyWaveFunctionAmplitude(1 / np.sqrt(2) - 1j / np.sqrt(2)),
        )

        # Make sure negatives are supported
        self.assertEqual(
            "-1/{s}2".format(s=TestHelpers.sqrtSymbol),
            qh.prettyWaveFunctionAmplitude(-1 / np.sqrt(2)),
        )

    def test_exponentiateMatrix(self):
        A = np.array([[1, -1], [2, 4]])
        result = qh.exponentiateMatrix(A)
        expectedResult = np.array(
            [
                [2 * np.e**2 - np.e**3, np.e**2 - np.e**3],
                [-2 * np.e**2 + 2 * np.e**3, -1 * np.e**2 + 2 * np.e**3],
            ]
        )
        TestHelpers.compareMatricies(self, expectedResult, result, places=7)

    def test_toString(self):
        self.assertEqual(
            "1/{s}2 |00> + 1/{s}2 |11>".format(s=TestHelpers.sqrtSymbol),
            qh.toString(np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])),
        )

    def test_MakeControlGateCX122(self):
        cx = qh.make_control_gate(1, 2, "X", 2)
        inputs = ["|00>", "|01>", "|10>", "|11>"]
        expected = ["|00>", "|01>", "|11>", "|10>"]
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            TestHelpers.compareMatricies(self, result.data, y.data)

    def test_MakeControlGateCX123(self):
        cx = qh.make_control_gate(1, 2, "X", 3)
        inputs = [
            "|000>",
            "|001>",
            "|010>",
            "|011>",
            "|100>",
            "|101>",
            "|110>",
            "|111>",
        ]
        expected = [
            "|000>",
            "|001>",
            "|010>",
            "|011>",
            "|110>",
            "|111>",
            "|100>",
            "|101>",
        ]
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            TestHelpers.compareMatricies(self, result.data, y.data)

    def test_MakeControlGateCX133(self):
        cx = qh.make_control_gate(1, 3, "X", 3)
        inputs = [
            "|000>",
            "|001>",
            "|010>",
            "|011>",
            "|100>",
            "|101>",
            "|110>",
            "|111>",
        ]
        expected = [
            "|000>",
            "|001>",
            "|010>",
            "|011>",
            "|101>",
            "|100>",
            "|111>",
            "|110>",
        ]
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            TestHelpers.compareMatricies(self, result.data, y.data)

    def test_MakeControlGateCX213(self):
        cx = qh.make_control_gate(2, 1, "X", 3)
        inputs = [
            "|000>",
            "|001>",
            "|010>",
            "|011>",
            "|100>",
            "|101>",
            "|110>",
            "|111>",
        ]
        expected = [
            "|000>",
            "|001>",
            "|110>",
            "|111>",
            "|100>",
            "|101>",
            "|010>",
            "|011>",
        ]
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            TestHelpers.compareMatricies(self, result.data, y.data)

    def test_MakeControlGateCX214(self):
        cx = qh.make_control_gate(3, 2, "X", 4)
        inputs = [
            "|0000>",
            "|0001>",
            "|0010>",
            "|0011>",
            "|0100>",
            "|0101>",
            "|0110>",
            "|0111>",
            "|1000>",
            "|1001>",
            "|1010>",
            "|1011>",
            "|1100>",
            "|1101>",
            "|1110>",
            "|1111>",
        ]
        expected = [
            "|0000>",
            "|0001>",
            "|0110>",
            "|0111>",
            "|0100>",
            "|0101>",
            "|0010>",
            "|0011>",
            "|1000>",
            "|1001>",
            "|1110>",
            "|1111>",
            "|1100>",
            "|1101>",
            "|1010>",
            "|1011>",
        ]
        for i, input in enumerate(inputs):
            x = qh.eval(input)
            y = qh.eval(expected[i])
            result = cx * x
            TestHelpers.compareMatricies(self, result.data, y.data)
