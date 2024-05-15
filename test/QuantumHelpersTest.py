import numpy as np
import unittest
import TestHelpers

TestHelpers.addSourceDirectoryToPath()

import QuantumHelpers as qh


class OtherTests(unittest.TestCase):
    def test_findFraction(self):
        self.assertEqual((1, 1), qh.find_fraction(1))

        self.assertEqual((1, 2), qh.find_fraction(1 / 2))

        self.assertEqual((1, 3), qh.find_fraction(1 / 3))
        self.assertEqual((2, 3), qh.find_fraction(2 / 3))

        self.assertEqual((1, 4), qh.find_fraction(1 / 4))
        self.assertEqual((1, 2), qh.find_fraction(2 / 4))
        self.assertEqual((3, 4), qh.find_fraction(3 / 4))

        self.assertEqual((1, 5), qh.find_fraction(1 / 5))
        self.assertEqual((2, 5), qh.find_fraction(2 / 5))
        self.assertEqual((3, 5), qh.find_fraction(3 / 5))
        self.assertEqual((4, 5), qh.find_fraction(4 / 5))
        self.assertEqual((1, 1), qh.find_fraction(5 / 5))

        self.assertEqual((1, 6), qh.find_fraction(1 / 6))
        self.assertEqual((1, 3), qh.find_fraction(2 / 6))
        self.assertEqual((1, 2), qh.find_fraction(3 / 6))
        self.assertEqual((2, 3), qh.find_fraction(4 / 6))
        self.assertEqual((5, 6), qh.find_fraction(5 / 6))
        self.assertEqual((1, 1), qh.find_fraction(6 / 6))

        self.assertEqual((1, 7), qh.find_fraction(1 / 7))
        self.assertEqual((2, 7), qh.find_fraction(2 / 7))
        self.assertEqual((3, 7), qh.find_fraction(3 / 7))
        self.assertEqual((4, 7), qh.find_fraction(4 / 7))
        self.assertEqual((5, 7), qh.find_fraction(5 / 7))
        self.assertEqual((6, 7), qh.find_fraction(6 / 7))
        self.assertEqual((1, 1), qh.find_fraction(7 / 7))

        self.assertEqual((1, 8), qh.find_fraction(1 / 8))
        self.assertEqual((1, 4), qh.find_fraction(2 / 8))
        self.assertEqual((3, 8), qh.find_fraction(3 / 8))
        self.assertEqual((1, 2), qh.find_fraction(4 / 8))
        self.assertEqual((5, 8), qh.find_fraction(5 / 8))
        self.assertEqual((3, 4), qh.find_fraction(6 / 8))
        self.assertEqual((7, 8), qh.find_fraction(7 / 8))
        self.assertEqual((1, 1), qh.find_fraction(8 / 8))

        # Make sure negatives are supported
        self.assertEqual((-1, 2), qh.find_fraction(-1 / 2))

    def test_findFractionComplex(self):
        self.assertEqual((1, 2, 3, 4), qh.find_fraction(1 / 2 + 3j / 4))
        self.assertEqual((1, 2, 0, 0), qh.find_fraction(1 / 2 + 0j))
        self.assertEqual((0, 0, 1, 1), qh.find_fraction(0 + 1j))
        self.assertEqual((1, 2, 3, 4), qh.find_fraction(1 / 2 + 3j / 4))

    def test_prettyFraction(self):
        self.assertEqual("1", qh.pretty_fraction(1))
        self.assertEqual("1/2", qh.pretty_fraction(1 / 2))
        self.assertEqual("0", qh.pretty_fraction(0))
        self.assertEqual("0", qh.pretty_fraction(0 + 0j))
        self.assertEqual("1/3", qh.pretty_fraction(1 / 3))
        self.assertEqual("1/10", qh.pretty_fraction(1 / 10))

    def test_printPrettyWaveFunctionAmplitude(self):
        self.assertEqual("1", qh.pretty_wave_function_amplitude(1))
        self.assertEqual(
            "1/{s}2".format(s=TestHelpers.sqrtSymbol),
            qh.pretty_wave_function_amplitude(1 / np.sqrt(2)),
        )
        self.assertEqual(
            "1/{s}3".format(s=TestHelpers.sqrtSymbol),
            qh.pretty_wave_function_amplitude(1 / np.sqrt(3)),
        )
        self.assertEqual("1/2", qh.pretty_wave_function_amplitude(1 / 2))
        self.assertEqual(
            "{s}3/2".format(s=TestHelpers.sqrtSymbol),
            qh.pretty_wave_function_amplitude(np.sqrt(3) / 2),
        )
        self.assertEqual(
            "{s}7/{s}8".format(s=TestHelpers.sqrtSymbol),
            qh.pretty_wave_function_amplitude(np.sqrt(7) / np.sqrt(8)),
        )
        self.assertEqual(
            "1/{s}2+1/{s}2j".format(s=TestHelpers.sqrtSymbol),
            qh.pretty_wave_function_amplitude(1 / np.sqrt(2) + 1j / np.sqrt(2)),
        )
        self.assertEqual(
            "1/{s}2-1/{s}2j".format(s=TestHelpers.sqrtSymbol),
            qh.pretty_wave_function_amplitude(1 / np.sqrt(2) - 1j / np.sqrt(2)),
        )

        # Make sure negatives are supported
        self.assertEqual(
            "-1/{s}2".format(s=TestHelpers.sqrtSymbol),
            qh.pretty_wave_function_amplitude(-1 / np.sqrt(2)),
        )

    def test_exponentiateMatrix(self):
        A = np.array([[1, -1], [2, 4]])
        result = qh.exponentiate_matrix(A)
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
            qh.to_string(np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])),
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
    
    def test_buildKetBinary1(self):
        result0 = qh.build_ket("|0>")
        result1 = qh.build_ket("|1>")
        TestHelpers.compareMatricies(self, result0.data, np.array([1,0]))
        TestHelpers.compareMatricies(self, result1.data, np.array([0,1]))

    def test_buildKetBinary2(self):
        result0 = qh.build_ket("|00>")
        result1 = qh.build_ket("|01>")
        result2 = qh.build_ket("|10>")
        result3 = qh.build_ket("|11>")
        TestHelpers.compareMatricies(self, result0.data, np.array([1,0,0,0]))
        TestHelpers.compareMatricies(self, result1.data, np.array([0,1,0,0]))
        TestHelpers.compareMatricies(self, result2.data, np.array([0,0,1,0]))
        TestHelpers.compareMatricies(self, result3.data, np.array([0,0,0,1]))

    def test_buildKetDecimal2(self):
        result0 = qh.build_ket("|2d0>")
        result1 = qh.build_ket("|2d1>")
        result2 = qh.build_ket("|2d2>")
        result3 = qh.build_ket("|2d3>")
        TestHelpers.compareMatricies(self, result0.data, np.array([1,0,0,0]))
        TestHelpers.compareMatricies(self, result1.data, np.array([0,1,0,0]))
        TestHelpers.compareMatricies(self, result2.data, np.array([0,0,1,0]))
        TestHelpers.compareMatricies(self, result3.data, np.array([0,0,0,1]))
