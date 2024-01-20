import numpy as np
import re
import unittest


hadamard = np.array([[1,1],[1,-1]])

# Index unitKets with the index of the state that you want
# unitKets(0) = |0> and unitKets(1) = |1>
unitKets = np.array([[1,0], [0,1]])

pauli_X = np.array([[0,1],[1,0]])
pauli_Y = np.array([[0,-1j],[1j,0]])
pauli_Z = np.array([[1, 0],[0, -1]])
pauli_plus = (1/2)*(pauli_X + 1j * pauli_Y)
pauli_minus = (1/2)*(pauli_X - 1j * pauli_Y)

cNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
cZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
hadamard = (1 / np.sqrt(2)) * np.array([[1,1],[1,-1]])

def buildKet(aKet):
    # Verify input has the correct format
    if not re.match("^|[0-1]+>", aKet):
        print("Argument passed to buildKet does not match expected ket format.")
        return -1
    localKet = 1
    # Goes through each character from the argument excluding the start and end characters
    for i in aKet[1:-1]:
        localKet = np.kron(localKet, unitKets[int(i)])
    return localKet

def buildBra(aBra):
    # Verify input has the correct format
    if not re.match("^<[0-1]+|", aBra):
        print("Argument passed to buildKet does not match expected ket format.")
        return -1
    localBra = 1
    # Goes through each character from the argument excluding the start and end characters
    for i in aBra[1:-1]:
        localBra = np.kron(localBra, unitKets[int(i)])
    return localBra

def printStates(aKet):
    numberOfQubits = int(np.log2(aKet.size))
    currentState = -1
    for i in aKet:
        currentState = currentState + 1
        if i == 0:
            continue
        print(prettyWaveFunctionAmplitude(i) + " |" + bin(currentState)[2:].zfill(numberOfQubits) + ">")

# Density Matrix
def makeDensityMatrix(waveFunction):
    numberOfQubits = int(np.log2(waveFunction.size))
    totalDensity = np.zeros((waveFunction.size, waveFunction.size))
    currentOuterState = -1
    currentInnerState = -1
    for outerState in waveFunction:
        currentOuterState = currentOuterState + 1
        localBra = buildBra("<" + bin(currentOuterState)[2:].zfill(numberOfQubits) + "|")
        localBra = localBra * outerState
        currentInnerState = -1
        for innerState in waveFunction:
            currentInnerState = currentInnerState + 1
            localKet = buildKet("|" + bin(currentInnerState)[2:].zfill(numberOfQubits) + ">")
            localKet = localKet * innerState            
            stateDensity = np.outer(localBra, localKet)
            totalDensity = totalDensity + stateDensity
    return totalDensity
        

def chainedKron(aListToKron):
    localKron = 1
    for i in aListToKron:
        localKron = np.kron(localKron, i)
    return localKron

def findFraction(n: float | complex) -> tuple[int, int] | tuple[int, int, int, int]:
    maxDenom = 10
    tolerance = 1e-8

    isComplex = True if type(n) == complex else False
    isNegative = False

    # Local variables to keep track of return values
    denominator = 0
    numerator = 0
    imagNumerator = 0
    imagDenominator = 0
    
    # If the passed in value in complex, make a recursive call to find the fraction for the imaginary part
    if isComplex:
        imagNumerator, imagDenominator = findFraction(n.imag)
        isNegative = (n.real < 0)
        p = np.abs(n.real)
    else:
        isNegative = n < 0
        p = np.abs(n)

    # Check some edge cases and return fast if n is 0 or one
    if p < tolerance and p >= 0:
        return (0,0) if not isComplex else (0,0,imagNumerator,imagDenominator)
    if p < 1 + tolerance and p > 1 - tolerance:
        return (1, 1) if not isComplex else (1, 1, imagNumerator, imagDenominator)
    
    for denom in range(1, maxDenom + 1):
        if numerator != 0: break
        for numer in reversed(range(1, denom)):
            distanceFromInt = ((p / numer) * denom) % 1
            if distanceFromInt < tolerance or (1 - distanceFromInt) < tolerance:
                if np.abs((numer / denom) - p) < tolerance:
                        numerator = numer
                        denominator = denom
                        break

    if isNegative:
        numerator = numerator * -1
    
    if isComplex:
        return numerator, denominator, imagNumerator, imagDenominator
    else:
        return numerator, denominator

def prettyWaveFunctionAmplitude(n) -> str:
    # n can be float or complex float
    tolerance = 1e-8
    sqrtSymbol = "\u221A"
    numerator, denominator = findFraction(n * np.conj(n))
    
    # If a fraction for the number cannot be found
    if denominator == 0:
        return str(n)
    if numerator / denominator < tolerance:
        return "0"
    if numerator / denominator > (1 - tolerance) and numerator / denominator < (1 + tolerance):
        if n < 0:
            return "-1"
        return "1"
    
    numeratorIsRootable = False
    denominatorIsRootable = False
    if np.sqrt(np.abs(numerator)) % 1 < tolerance or ( 1 - (np.sqrt(np.abs(numerator)) % 1)) < tolerance:
        numeratorIsRootable = True
    if np.sqrt(denominator) % 1 < tolerance or ( 1 - (np.sqrt(denominator) % 1)) < tolerance:
        denominatorIsRootable = True

    numeratorString = str(int(np.sqrt(numerator))) if numeratorIsRootable else sqrtSymbol + str(int(numerator))
    denominatorString = str(int(np.sqrt(denominator))) if denominatorIsRootable else sqrtSymbol + str(int(denominator))

    if n < 0:
        numeratorString = "-" + numeratorString

    return "{n}/{d}".format(n=numeratorString, d=denominatorString)

vPrettyWaveFunctionAmplitude = np.vectorize(prettyWaveFunctionAmplitude)

def prettyFraction(n) -> str:
    tolerance = 1e-8
    numerator, denominator = findFraction(n)

    # If a fraction for the number cannot be found
    if numerator == 0:
        return str(n)
    if numerator / denominator < tolerance:
        return "0"
    if numerator / denominator > (1 - tolerance) and numerator / denominator < (1 + tolerance):
        if n < 0:
            return "-1"
        return "1"
    
    numeratorString = str(numerator)
    denominatorString = str(denominator)

    if n < 0:
        numeratorString = "-" + numeratorString
    
    return "{n}/{d}".format(n=numeratorString, d=denominatorString)

vPrettyFraction = np.vectorize(prettyFraction)

def makeControlGate(gate, controlPosition):
    zeroState = np.outer(buildKet("|0>"), buildBra("<0|"))
    oneState = np.outer(buildKet("|1>"), buildBra("<1|"))

    if controlPosition == 0:
        return np.kron(zeroState, np.eye(2)) + np.kron(oneState, gate)
    elif controlPosition == 1:
        return np.kron(np.kron(oneState, np.eye(2)) + np.kron(zeroState, gate))



# -------------------- UNIT TESTS --------------------

class TestQuantumHelpers(unittest.TestCase):
    def test_findFraction(self):
        self.assertEqual((1,1), findFraction(1))

        self.assertEqual((1,2), findFraction(1/2))

        self.assertEqual((1,3), findFraction(1/3))
        self.assertEqual((2,3), findFraction(2/3))

        self.assertEqual((1,4), findFraction(1/4))
        self.assertEqual((1,2), findFraction(2/4))
        self.assertEqual((3,4), findFraction(3/4))

        self.assertEqual((1,5), findFraction(1/5))
        self.assertEqual((2,5), findFraction(2/5))
        self.assertEqual((3,5), findFraction(3/5))
        self.assertEqual((4,5), findFraction(4/5))
        self.assertEqual((1,1), findFraction(5/5))

        self.assertEqual((1,6), findFraction(1/6))
        self.assertEqual((1,3), findFraction(2/6))
        self.assertEqual((1,2), findFraction(3/6))
        self.assertEqual((2,3), findFraction(4/6))
        self.assertEqual((5,6), findFraction(5/6))
        self.assertEqual((1,1), findFraction(6/6))

        self.assertEqual((1,7), findFraction(1/7))
        self.assertEqual((2,7), findFraction(2/7))
        self.assertEqual((3,7), findFraction(3/7))
        self.assertEqual((4,7), findFraction(4/7))
        self.assertEqual((5,7), findFraction(5/7))
        self.assertEqual((6,7), findFraction(6/7))
        self.assertEqual((1,1), findFraction(7/7))

        self.assertEqual((1,8), findFraction(1/8))
        self.assertEqual((1,4), findFraction(2/8))
        self.assertEqual((3,8), findFraction(3/8))
        self.assertEqual((1,2), findFraction(4/8))
        self.assertEqual((5,8), findFraction(5/8))
        self.assertEqual((3,4), findFraction(6/8))
        self.assertEqual((7,8), findFraction(7/8))
        self.assertEqual((1,1), findFraction(8/8))

        # Make sure negatives are supported
        self.assertEqual((-1, 2), findFraction(-1/2))

    def test_findFractionComplex(self):
        self.assertEqual((1, 2, 3, 4), findFraction(1/2 + 3j/4))
        self.assertEqual((1, 2, 0, 0), findFraction(1/2 + 0j))
        self.assertEqual((0, 0, 1, 1), findFraction(0 + 1j))
        self.assertEqual((1, 2, 3, 4), findFraction(1/2 + 3j/4))

    def test_prettyFraction(self):
        self.assertEqual("1", prettyFraction(1))
        self.assertEqual("1/2", prettyFraction(1/2))
        self.assertEqual("0", prettyFraction(0))
        self.assertEqual("1/3", prettyFraction(1/3))
        self.assertEqual("1/10", prettyFraction(1/10))

    def test_printPrettyWaveFunctionAmplitude(self):
        sqrtSymbol = "\u221A"
        self.assertEqual("1", prettyWaveFunctionAmplitude(1))
        self.assertEqual("1/{s}2".format(s=sqrtSymbol), prettyWaveFunctionAmplitude(1/np.sqrt(2)))
        self.assertEqual("1/{s}3".format(s=sqrtSymbol), prettyWaveFunctionAmplitude(1/np.sqrt(3)))
        self.assertEqual("1/2", prettyWaveFunctionAmplitude(1/2))
        self.assertEqual("{s}3/2".format(s=sqrtSymbol), prettyWaveFunctionAmplitude(np.sqrt(3)/2))
        self.assertEqual("{s}7/{s}8".format(s=sqrtSymbol), prettyWaveFunctionAmplitude(np.sqrt(7)/np.sqrt(8)))

        #Make sure negatives are supported
        self.assertEqual("-1/{s}2".format(s=sqrtSymbol), prettyWaveFunctionAmplitude(-1/np.sqrt(2)))

    def test_makeControlGate(self):
        self.compareMatricies(cNOT, makeControlGate(pauli_X, 0))

    def compareMatricies(self, a, b):
        if(a.shape != b.shape):
            self.fail("Shapes do not match. " + str(a.shape) + " != " + str(b.shape))
        for row in range(a.shape[0]):
            for col in range(a.shape[1]):
                self.assertEqual(a[row][col], b[row][col])

        



# Run unit tests if run as a script
if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
