import numpy as np
import re
import unittest
from enum import Enum

DEBUG = False

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
    for state, coefficient in enumerate(aKet):
        if coefficient == 0:
            continue
        print("{c} |{s}>".format(c = prettyWaveFunctionAmplitude(coefficient), s=bin(state)[2:].zfill(numberOfQubits)))

def toString(aKet):
    if type(aKet) == tuple:
        match aKet[1]:
            case WaveFunctionTokens.SCALAR:
                return str(aKet[0])
            case WaveFunctionTokens.OPERATOR:
                return vPrettyWaveFunctionAmplitude(aKet[0])
            case WaveFunctionTokens.KET:
                return toString(aKet[0])
            case WaveFunctionTokens.BRA:
                return toString(aKet[0])
    numberOfQubits = int(np.log2(aKet.size))
    psi = ""
    for state, coefficient in enumerate(aKet):
        if coefficient == 0:
            continue
        if len(psi) > 0:
            psi = psi + " + "
        psi = psi + "{c} |{s}>".format(c = prettyWaveFunctionAmplitude(coefficient), s=bin(state)[2:].zfill(numberOfQubits))
    return psi

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
    
    # Brute force check every possible numerator for each denominator between 0 and maxDenom--**************+**+++*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    

def tokenizeWaveFunctionString(stringstrong):
    # Tokenize a string
    # Characters to tokenize on: <, >, |, Capitol Letters, spaces
    soloTokenPattern = r"[+,*,-,/,(,),√, ]"
    beginPattern = r"[<,A-Z]"
    endPattern = r"[>]"
    vert = '|'
    tokens = []
    # Is it easiest to just loop through characters?
    currentToken = ""
    for character in stringstrong:
        startOfToken = False
        endOfToken = False
        if re.search(soloTokenPattern, character) is not None:
            startOfToken = True
            endOfToken = True
        elif re.search(beginPattern, character) is not None:
            startOfToken = True
        elif re.search(endPattern, character) is not None:
            endOfToken = True
        elif character == vert:
            if currentToken == "":
                startOfToken = True
            elif currentToken[0] == '<':
                endOfToken = True
            else:
                startOfToken = True
        
        # Handle the tokens and currentToken for if it is the start, end, or middle of the token
        if startOfToken and endOfToken:
            if currentToken != "":
                tokens.append(currentToken)
            if character != " ":
                tokens.append(character)
            currentToken = ""
        elif startOfToken:
            if currentToken != "":
                tokens.append(currentToken)
            currentToken = character
        elif endOfToken:
            currentToken = currentToken + character
            tokens.append(currentToken)
            currentToken = ""
        else:
            currentToken = currentToken + character
    
    # If there is anything left at the end, add it to tokens
    if currentToken != "":
        tokens.append(currentToken)
    return tokens

class WaveFunctionTokens(Enum):
    BRA = 1
    KET = 2
    OPERATOR = 3
    SCALAR = 4
    ARITHMETIC = 5

operators = {
    "X": pauli_X,
    "Y": pauli_Y,
    "Z": pauli_Z,
    "H": hadamard,
    "I": np.eye(2),
    "Cnot": cNOT
}

def buildWaveFunction(tokens):
    operatorsPattern = r"^[A-Z][a-z]*"
    braPattern = r"^\<[0,1]+\|"
    ketPattern = r"^\|[0,1]+\>$"
    scalarPattern = r"^[0-9,.]+$"
    parenPattern = r"^[(,)]$"
    endTermPattern = r"^[+,-]$"
    arithmaticPattern = r"^[*,/,√]$|^Sr$"

    openParenStack = []
    overallStack = []
    currentTermStack = []

    if DEBUG: print("building " + str(tokens))

    # Figure out what type each token in and add it into the current term stack in a tuple
    # as (token, type) where the type is a WaveFunctionTokens type and the token is
    # evaluated in into its numeric form.
    # The order of these pattern matche matter because there are specific patterns
    # farther up that will also match more general patterns below.
    for i, token in enumerate(tokens):
        if re.search(parenPattern, token):
            if DEBUG: print("paren")
            if token == "(":
                openParenStack.append(i)
            if token == ")":
                if len(openParenStack) == 0:
                    print("ERROR: Got a closing paren without a matching opening paren")
                    return None
                loc, loctype = buildWaveFunction(tokens[openParenStack.pop() + 1:i])
                currentTermStack.append((loc, loctype))
        elif len(openParenStack) > 0:
            continue
        elif re.search(arithmaticPattern, token):
            if DEBUG: print("arithmatic")
            currentTermStack.append((token, WaveFunctionTokens.ARITHMETIC))
        elif re.search(operatorsPattern,token):
            if DEBUG: print("operator")
            if token in operators:
                currentTermStack.append((operators[token], WaveFunctionTokens.OPERATOR))
            else:
                print("ERROR: Unrecognized Operator: " + token)
        elif re.search(ketPattern, token):
            if DEBUG: print("ket")
            currentTermStack.append((buildKet(token), WaveFunctionTokens.KET))
        elif re.search(braPattern, token):
            if DEBUG: print("bra")
            currentTermStack.append((buildBra(token), WaveFunctionTokens.BRA))
        elif re.search(scalarPattern, token):
            if DEBUG: print("scalar")
            currentTermStack.append((float(token), WaveFunctionTokens.SCALAR))
        elif re.search(endTermPattern, token):
            if DEBUG: print("end of term")
            #Evaluate current term and put result into overall stack
            overallStack.append(evaluateStack(currentTermStack))
            currentTermStack = []
            # Put arithmetic onto overall stack
            overallStack.append((token, WaveFunctionTokens.ARITHMETIC))
        else:
            print("token not recognized")
    
    if len(openParenStack) > 0:
        print("ERROR: Unclosed parenthesis")
    # Evaluate the full stack and what is left over in the overall stack
    return evaluateStack(overallStack + currentTermStack)

def evaluateStack(stack):
    if DEBUG: print("evaluating stack " + str(stack))
    while len(stack) > 1:
        # evaluate
        right = stack.pop()
        left = stack.pop()
        arithmetic = None
        result = None
        if left[1] == WaveFunctionTokens.ARITHMETIC and not (left[0] == "√" or left[0] == "Sr"):
            arithmetic = left
            left = stack.pop()
            result = evaluateExplicit(left=left, arithmetic=arithmetic, right=right)
        else:
            result = evaluateImplicit(left=left,right=right)
        stack.append(result)

    rtn = stack.pop()
    if DEBUG: print("Evaluated stack as: " + str(rtn))
    return rtn

def evaluateExplicit(left, arithmetic, right):
    # Really only need to handle BRA, KET, and SCALAR in this method
    # Left side BRA
    if left[1] == WaveFunctionTokens.BRA:
        # Arithmetic +
        if arithmetic[0] == "+":
            if right[1] == WaveFunctionTokens.BRA:
                return (left[0] + right[0], WaveFunctionTokens.BRA)
        # Arithmetic -
        if arithmetic[0] == "-":
            if right[1] == WaveFunctionTokens.BRA:
                return (left[0] - right[0], WaveFunctionTokens.BRA)
        # Arithmetic *
        if arithmetic[0] == "*":
            if right[1] == WaveFunctionTokens.SCALAR:
                return (left[0] * right[0], WaveFunctionTokens.BRA)
        # Arithmetic /
        if arithmetic[0] == "/":
            if right[1] == WaveFunctionTokens.SCALAR:
                return (left[0] * right[0], WaveFunctionTokens.BRA)
    # Left side KET
    if left[1] == WaveFunctionTokens.KET:
        # Arithmetic +
        if arithmetic[0] == "+":
            if right[1] == WaveFunctionTokens.KET:
                return (left[0] + right[0], WaveFunctionTokens.KET)
        # Arithmetic -
        if arithmetic[0] == "-":
            if right[1] == WaveFunctionTokens.KET:
                return (left[0] - right[0], WaveFunctionTokens.KET)
        # Arithmetic *
        if arithmetic[0] == "*":
            if right[1] == WaveFunctionTokens.SCALAR:
                return (left[0] * right[0], WaveFunctionTokens.KET)
        # Arithmetic /
        if arithmetic[0] == "/":
            if right[1] == WaveFunctionTokens.SCALAR:
                return (left[0] * right[0], WaveFunctionTokens.KET)
    # Left side SCALAR
    if left[1] == WaveFunctionTokens.SCALAR:
        # Arithmetic +
        if arithmetic[0] == "+":
            if right[1] == WaveFunctionTokens.SCALAR:
                return (left[0] + right[0], WaveFunctionTokens.SCALAR)
         # Arithmetic -
        if arithmetic[0] == "-":
            if right[1] == WaveFunctionTokens.SCALAR:
                return (left[0] + right[0], WaveFunctionTokens.SCALAR)
         # Arithmetic *
        if arithmetic[0] == "*":
            if right[1] == WaveFunctionTokens.SCALAR:
                return (left[0] * right[0], WaveFunctionTokens.SCALAR)
         # Arithmetic /
        if arithmetic[0] == "/":
            if right[1] == WaveFunctionTokens.SCALAR:
                return (left[0] / right[0], WaveFunctionTokens.SCALAR)
        
    
    print("Something was not handled, evaluateExplicit. Left:{l} Arithmetic:{a} Right:{r}".format(\
        l=str(left), r=str(right),a=str(arithmetic)))

def evaluateImplicit(left, right):
    # Left side BRA
    if left[1] == WaveFunctionTokens.BRA:
        # BRA BRA
        if right[1] == WaveFunctionTokens.BRA:
            return (np.kron(left[0], right[0]), WaveFunctionTokens.BRA)
        # BRA KET
        if right[1] == WaveFunctionTokens.KET:
            return (np.inner(left[0], right[0]), WaveFunctionTokens.SCALAR)
        # BRA OPERATOR
            # Doesn't make sense to have operator after Bra
        # BRA SCALAR
            # Doesn't make sense to have scalar after Bra
        # BRA ARITHMETIC
            # Doesn't make sense to have arithmetic as the right
    # Left side KET
    if left[1] == WaveFunctionTokens.KET:
        # KET BRA
        if right[1] == WaveFunctionTokens.BRA:
            return (np.outer(left[0],right[0]), WaveFunctionTokens.OPERATOR)
        # KET KET
        if right[1] == WaveFunctionTokens.KET:
            return (np.kron(left[0], right[0]), WaveFunctionTokens.KET)
        # KET OPERATOR
            # Doesn't make sense to have operator after KET
        # KET SCALAR
            # Doesn't make sense to have scalar after KET
        # KET ARITHMETIC
            # Doesn't make sense to have arithmetic as the right
    # Left side operator
    if left[1] == WaveFunctionTokens.OPERATOR:
        # OPERATOR BRA
            # Doesn't make sense, dimensions won't work
        # OPERATOR KET
        if right[1] == WaveFunctionTokens.KET:
            return (np.matmul(left[0], right[0]), WaveFunctionTokens.KET)
        # OPERATOR OPERATOR
        if right[1] == WaveFunctionTokens.OPERATOR:
            return (np.kron(left[0], right[0]), WaveFunctionTokens.OPERATOR)
        # OPERATOR SCALAR
            # Doesn't make sense to have a scalar after an operator
        # OPERATOR ARITHMETIC
            # Doesn't make sense to have arithmetic as the right
    # left side SCALAR
    if left[1] == WaveFunctionTokens.SCALAR:
        # SCALAR BRA
        if right[1] == WaveFunctionTokens.BRA:
            return (left[0] * right[0], WaveFunctionTokens.BRA)
        # SCALAR KET
        if right[1] == WaveFunctionTokens.KET:
            return (left[0] * right[0], WaveFunctionTokens.KET)
        # SCALAR OPERATOR
        if right[1] == WaveFunctionTokens.OPERATOR:
            return (left[0] * right[0], WaveFunctionTokens.OPERATOR)
        # SCALAR SCALAR
        if right[1] == WaveFunctionTokens.SCALAR:
            return (left[0] * right[0], WaveFunctionTokens.SCALAR)
        # SCALAR ARITHMETIC
            # Doesn't make sense to have arithmetic as the right
    # Left side Arithmetic
    if left[1] == WaveFunctionTokens.ARITHMETIC and (left[0] == "√" or left[0] == "Sr"):
        # Only handling square root in this method
        if right[1] == WaveFunctionTokens.SCALAR:
            return (np.sqrt(right[0]), WaveFunctionTokens.SCALAR)
    
    print("Something was not handled, evaluateImplicit. Left:{l} Right:{r}".format(l=str(left), r=str(right)))

def readInWaveFunction(psi):
    tokens = tokenizeWaveFunctionString(psi)
    evaluatedPsi = buildWaveFunction(tokens)
    return toString(evaluatedPsi[0])

# -------------------- UNIT TESTS --------------------

class TestQuantumHelpers(unittest.TestCase):
    sqrtSymbol = "\u221A"
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
        self.assertEqual("1", prettyWaveFunctionAmplitude(1))
        self.assertEqual("1/{s}2".format(s=self.sqrtSymbol), prettyWaveFunctionAmplitude(1/np.sqrt(2)))
        self.assertEqual("1/{s}3".format(s=self.sqrtSymbol), prettyWaveFunctionAmplitude(1/np.sqrt(3)))
        self.assertEqual("1/2", prettyWaveFunctionAmplitude(1/2))
        self.assertEqual("{s}3/2".format(s=self.sqrtSymbol), prettyWaveFunctionAmplitude(np.sqrt(3)/2))
        self.assertEqual("{s}7/{s}8".format(s=self.sqrtSymbol), prettyWaveFunctionAmplitude(np.sqrt(7)/np.sqrt(8)))

        #Make sure negatives are supported
        self.assertEqual("-1/{s}2".format(s=self.sqrtSymbol), prettyWaveFunctionAmplitude(-1/np.sqrt(2)))

    def test_makeControlGate(self):
        self.compareMatricies(cNOT, makeControlGate(pauli_X, 0))

    def compareMatricies(self, a, b):
        if(a.shape != b.shape):
            self.fail("Shapes do not match. " + str(a.shape) + " != " + str(b.shape))
        for row in range(a.shape[0]):
            for col in range(a.shape[1]):
                self.assertEqual(a[row][col], b[row][col])

    def compareVectors(self, a, b):
        if(a.shape != b.shape):
            self.fail("Shapes do not match. " + str(a.shape) + " != " + str(b.shape))
        for i in range(a.shape[0]):
            self.assertEqual(a[i], b[i], "{a} does not match {b}".format(a=a,b=b))
    
    def test_toString(self):
        self.assertEqual("1/{s}2 |00> + 1/{s}2 |11>".format(s=self.sqrtSymbol), toString(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])))

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
    
    def tokenizeCompare(self, testPsi, expectedTokens):
        rtnArray = tokenizeWaveFunctionString(testPsi)
        self.assertEqual(len(rtnArray), len(expectedTokens), "Sizes not equal. Got: {t} Expected: {e}".format(t=rtnArray, e=expectedTokens))
        for i in range(len(expectedTokens)):
            self.assertEqual(rtnArray[i], expectedTokens[i])

    def test_BuildWaveFunctionSuperposition(self):
        testPsi = "{c}(|0> + |1>)".format(c=1/np.sqrt(2))
        tokens = tokenizeWaveFunctionString(testPsi)
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([1/np.sqrt(2), 1/np.sqrt(2)]))

    def test_BuildWaveFunctionXGate(self):
        tokens = ['X', "|0>"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([0,1]))
    
    def test_BuildWaveFunctionCnotGate(self):
        tokens = ['Cnot', "|00>"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([1,0,0,0]))
        tokens = ['Cnot', "|10>"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([0,0,0,1]))

    def test_BuildWaveFunctionFlipFirst(self):
        tokens = ["(", "X", "I", ")", "|00>"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([0,0,1,0]))

    def test_BuildWaveFunctionFlipSecond(self):
        tokens = ["(", "I", "X", ")", "|00>"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([0,1,0,0]))

    def test_BuildWaveFunctionBellState(self):
        tokens = ["Cnot", "(", "H", "I", ")", "|00>"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)]))

    def test_BuildWaveFunctionSqrtSymbol(self):
        tokens = ["(", "1", "/", "√", "2", ")", "(", "|0>", "+", "|1>", ")"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([1/np.sqrt(2),1/np.sqrt(2)]))

    def test_BuildWaveFunctionSqrtWord(self):
        tokens = ["(", "1", "/", "Sr", "2", ")", "(", "|0>", "+", "|1>", ")"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareVectors(rtnPsi[0], np.array([1/np.sqrt(2),1/np.sqrt(2)]))
    
    def test_evalKetBra(self):
        tokens = ["|0>", "<0|"]
        rtnPsi = buildWaveFunction(tokens)
        self.compareMatricies(rtnPsi[0], np.array([[1,0],[0,0]]))
    


# Run unit tests if run as a script
if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
