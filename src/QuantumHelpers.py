import numpy as np
import re
import numbers
import unittest
from enum import Enum


# Figure out if matplotlib is installed. If not, plotting is not available
plottingAvailable = False
try:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    plottingAvailable = True
except:
    plottingAvailable = False


# Choose whether to print debug statements.
DEBUG = False

# Index unitKets with the index of the state that you want
# unitKets(0) = |0> and unitKets(1) = |1>
unitKets = np.array([[1, 0], [0, 1]])

# Common matricies
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Y = np.array([[0, -1j], [1j, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])
pauli_plus = (1 / 2) * (pauli_X + 1j * pauli_Y)
pauli_minus = (1 / 2) * (pauli_X - 1j * pauli_Y)

cNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
cZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
y_hadamard = (1 / np.sqrt(2)) * np.array([[-1, 1j], [-1j, 1]])

# Map string representation of known matricies to actual matrix.
operators = {
    "X": pauli_X,
    "Y": pauli_Y,
    "Z": pauli_Z,
    "H": hadamard,
    "Hy": y_hadamard,
    "I": np.eye(2),
    "Cnot": cNOT,
}

knownScalars = {"Pi": np.pi, "π": np.pi}


class WaveFunctionTokens(Enum):
    """Types of tokens for the wavefunction tokens. These will determine the interactions between different elements."""

    BRA = 1
    KET = 2
    OPERATOR = 3
    SCALAR = 4
    ARITHMETIC = 5
    FUNCTION = 6


class WaveFunctionElement:
    """Holds an element of quantum information. Operators are overloaded to determine interaction based on type."""

    data = []
    type: WaveFunctionTokens

    def __init__(self, data, type: WaveFunctionTokens) -> None:
        self.data = data
        self.type = type

    def __add__(self, other):
        """
        Overloaded + : Add 2 elements together. Elements must be of the same type.
        """
        if not isinstance(other, WaveFunctionElement):
            return self.printNotSupported()

        if self.type == other.type:
            return WaveFunctionElement(self.data + other.data, self.type)
        else:
            self.printError(other, "add")

    def __sub__(self, other):
        """
        Overloaded - : Subtract other element from this element. Elements must be of the same type.
        """
        if not isinstance(other, WaveFunctionElement):
            return self.printNotSupported()

        if self.type == other.type:
            return WaveFunctionElement(self.data - other.data, self.type)
        else:
            self.printError(other, "subtract")

    def __mul__(self, other):
        """
        Overloaded * : Multiply elements together, or apply left onto right.
        Supported type interactions:
            Scalar * Any = Any
            Any * Scalar = Any
            Bra * Bra = Bra (kronecker product)
            Bra * Ket = Scalar
            Bra * Operator = Bra
            Ket * Bra = Operator
            Ket * Ket = Ket (kronecker product)
            Operator * Ket = Ket
            Operator * Operator = Operator
        """
        # Support multiplying by a normal number
        if isinstance(other, numbers.Number):
            return WaveFunctionElement(self.data * other, self.type)
        if not isinstance(other, WaveFunctionElement):
            return self.printNotSupported()

        # Some quick logic to handle scalars because it is simple
        if self.type == WaveFunctionTokens.SCALAR:
            return WaveFunctionElement(self.data * other.data, other.type)
        elif other.type == WaveFunctionTokens.SCALAR:
            return WaveFunctionElement(self.data * other.data, self.type)

        match self.type:
            case WaveFunctionTokens.BRA:
                match other.type:
                    case WaveFunctionTokens.BRA:
                        return self & other
                    case WaveFunctionTokens.KET:
                        return WaveFunctionElement(
                            np.inner(self.data, other.data), WaveFunctionTokens.SCALAR
                        )
                    case WaveFunctionTokens.OPERATOR:
                        return WaveFunctionElement(
                            self.data @ other.data, WaveFunctionTokens.BRA
                        )
                    case _:
                        return self.printError(other, "multiply")
            case WaveFunctionTokens.KET:
                match other.type:
                    case WaveFunctionTokens.BRA:
                        return WaveFunctionElement(
                            np.outer(self.data, other.data), WaveFunctionTokens.OPERATOR
                        )
                    case WaveFunctionTokens.KET:
                        return self & other
                    case _:
                        return self.printError(other, "multiply")
            case WaveFunctionTokens.OPERATOR:
                match other.type:
                    case WaveFunctionTokens.KET:
                        return WaveFunctionElement(
                            self.data @ other.data, WaveFunctionTokens.KET
                        )
                    case WaveFunctionTokens.OPERATOR:
                        return WaveFunctionElement(
                            self.data @ other.data, WaveFunctionTokens.OPERATOR
                        )
                    case _:
                        return self.printError(other, "multiply")
            case _:
                return self.printError(other, "multiply")

    def __truediv__(self, other):
        """
        Overloaded / : Divide by a number or scalar
        """
        # Support dividing by a normal number
        if isinstance(other, numbers.Number):
            return WaveFunctionElement(self.data / other, self.type)
        if not isinstance(other, WaveFunctionElement):
            return self.printNotSupported()

        # Division is only supported with scalars
        if other.type == WaveFunctionTokens.SCALAR:
            return WaveFunctionElement(self.data / other.data, self.type)
        else:
            self.printError(other, "divide")

    def __and__(self, other):
        """
        Overloaded & : Kronecker product elements together. Types must be the same
        """
        if not isinstance(other, WaveFunctionElement):
            return self.printNotSupported()

        # Since python does not have a kron, use & as kron symbol
        if self.type == other.type and self.type in [
            WaveFunctionTokens.BRA,
            WaveFunctionTokens.KET,
            WaveFunctionTokens.OPERATOR,
        ]:
            return WaveFunctionElement(np.kron(self.data, other.data), self.type)
        else:
            self.printError(other, "kron")

    def printError(self, other, operation):
        """
        Print an error to stdout in a standardized way.

        Params:
            self: this element
            other: other element being interacted with
            operation: Which operation was being attempted.
        """
        print(
            "Cannot {operation} {s} and {o}".format(
                operation=operation, s=self.type.name, o=other.type.name
            )
        )

    def printNotSupported(self):
        """
        Standard error message when an interation is not supported.
        """
        print("Operation with non-QuantumElement object not supported.")

    def __str__(self):
        """
        Overloaded to string. Convert element to a more readable format.
        """
        match self.type:
            case WaveFunctionTokens.BRA:
                return str(vPrettyWaveFunctionAmplitude(self.data)).replace("'", "")
            case WaveFunctionTokens.KET:
                return str(
                    vPrettyWaveFunctionAmplitude(
                        np.reshape(self.data, (len(self.data), 1))
                    )
                ).replace("'", "")
            case WaveFunctionTokens.OPERATOR:
                return str(vPrettyWaveFunctionAmplitude(self.data)).replace("'", "")
            case WaveFunctionTokens.SCALAR:
                return str(prettyWaveFunctionAmplitude(self.data)).replace("'", "")
            case WaveFunctionTokens.ARITHMETIC:
                return self.data
            case _:
                return "Type: {type} to string not implemented".format(type=type)

    def dagger(self):
        """
        Return a hermetian conjugate of the element.
        """
        newType = self.type
        if self.type == WaveFunctionTokens.BRA:
            newType = WaveFunctionTokens.KET
        elif self.type == WaveFunctionTokens.KET:
            newType = WaveFunctionTokens.BRA

        return WaveFunctionElement(np.conj(np.transpose(self.data)), newType)

    def draw(self):
        """
        Draw a plot of the state.
        """

        # TODO: Write helpful description

        if not plottingAvailable:
            print("Plotting not available. Make sure matplotlib is installed.")
            return
        plt.style.use("Solarize_Light2")

        axs = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))

        self._addStatePlot(axs[1][0], False)
        self._addStatePlot(axs[1][1], True)

    def _addStatePlot(self, ax, imag: bool):
        if imag:
            x = self.data.imag
            title = "Imaginary"
        else:
            x = self.data.real
            title = "Real"

        # Center window around origin
        xlim = [-1.2, 1.2]
        ylim = [-1.2, 1.2]

        # Set tick marks to quadrants and add labels
        # Not sure if there is a way to do it without doing it 3 times
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([0], labels="")
        ax.set_yticks([0], labels="")
        ax.set_xlabel("-|0>")
        ax.set_ylabel("-|1>")
        ax2 = ax.twinx()
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_xticks([0], labels="")
        ax2.set_yticks([0], labels="")
        ax2.set_ylabel("|1>")
        ax3 = ax.twiny()
        ax3.set_xlabel("|0>")
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        ax3.set_xticks([0], labels="")
        ax3.set_yticks([0], labels="")

        # Draw circle at len=1
        normCircle = Circle((0, 0), 1, linestyle="--", fill=False, color="black")
        ax.add_patch(normCircle)

        # Draw x-axis
        plt.plot([-2, 2], [-2, 2], "--", color="green")

        # Draw the actual state
        arrow_head_length = np.sqrt(x[1] ** 2 + x[0] ** 2) * 0.1
        plt.arrow(
            0,
            0,
            x[1],
            x[0],
            color="blue",
            width=0.02,
            head_width=arrow_head_length,
            head_length=arrow_head_length,
            length_includes_head=True,
        )

    def print(self):
        if self.type == WaveFunctionTokens.KET:
            printStates(self.data)


def buildKet(aKet):
    """
    Build a numpy array of the passed in ket string.

    Args:
        aKet (string): String of 1s and 0s for the states of the Ket. Must match '|xxx>' format (with the xs representing 1s and 0s)
    """
    # Verify input has the correct format
    if not re.match("^|[0-1]+>", aKet):
        print("Argument passed to buildKet does not match expected ket format.")
        return -1
    localKet = 1 + 0j
    # Goes through each character from the argument excluding the start and end characters
    for i in aKet[1:-1]:
        localKet = np.kron(localKet, unitKets[int(i)])
    return WaveFunctionElement(localKet, WaveFunctionTokens.KET)


def buildBra(aBra):
    # Verify input has the correct format
    if not re.match("^<[0-1]+|", aBra):
        print("Argument passed to buildKet does not match expected ket format.")
        return -1
    localBra = 1 + 0j
    # Goes through each character from the argument excluding the start and end characters
    for i in aBra[1:-1]:
        localBra = np.kron(localBra, unitKets[int(i)])
    return WaveFunctionElement(localBra, WaveFunctionTokens.BRA)


def printStates(aKet):
    numberOfQubits = int(np.log2(aKet.size))
    currentState = -1
    for state, coefficient in enumerate(aKet):
        if coefficient == 0:
            continue
        print(
            "{c} |{s}>".format(
                c=prettyWaveFunctionAmplitude(coefficient),
                s=bin(state)[2:].zfill(numberOfQubits),
            )
        )


def toString(aKet):
    if type(aKet) == tuple:
        match aKet.type:
            case WaveFunctionTokens.SCALAR:
                return str(aKet.data)
            case WaveFunctionTokens.OPERATOR:
                return vPrettyWaveFunctionAmplitude(aKet.data)
            case WaveFunctionTokens.KET:
                return toString(aKet.data)
            case WaveFunctionTokens.BRA:
                return toString(aKet.data)
    numberOfQubits = int(np.log2(aKet.size))
    psi = ""
    for state, coefficient in enumerate(aKet):
        if coefficient == 0:
            continue
        if len(psi) > 0:
            psi = psi + " + "
        psi = psi + "{c} |{s}>".format(
            c=prettyWaveFunctionAmplitude(coefficient),
            s=bin(state)[2:].zfill(numberOfQubits),
        )
    return psi


# Density Matrix
def makeDensityMatrix(waveFunction):
    numberOfQubits = int(np.log2(waveFunction.size))
    totalDensity = np.zeros((waveFunction.size, waveFunction.size))
    currentOuterState = -1
    currentInnerState = -1
    for outerState in waveFunction:
        currentOuterState = currentOuterState + 1
        localBra = buildBra(
            "<" + bin(currentOuterState)[2:].zfill(numberOfQubits) + "|"
        )
        localBra = localBra * outerState
        currentInnerState = -1
        for innerState in waveFunction:
            currentInnerState = currentInnerState + 1
            localKet = buildKet(
                "|" + bin(currentInnerState)[2:].zfill(numberOfQubits) + ">"
            )
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
    maxDenom = 16
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
        isNegative = n.real < 0
        p = np.abs(n.real)
    else:
        isNegative = n < 0
        p = np.abs(n)

    # Check some edge cases and return fast if n is 0 or one
    if p < tolerance and p >= 0:
        return (0, 0) if not isComplex else (0, 0, imagNumerator, imagDenominator)
    if p < 1 + tolerance and p > 1 - tolerance:
        return (1, 1) if not isComplex else (1, 1, imagNumerator, imagDenominator)

    # Brute force check every possible numerator for each denominator between 0 and maxDenom
    for denom in range(1, maxDenom + 1):
        if numerator != 0:
            break
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
    tolerance = 1e-8
    sqrtSymbol = "\u221A"

    complexString = ""
    if np.abs(n.imag) > tolerance:
        complexString = prettyWaveFunctionAmplitude(n.imag)
        if n.imag > 0:
            complexString = "+" + complexString
        complexString = complexString + "j"

    if abs(n.real) < tolerance:
        return "0{c}".format(c=complexString)

    numerator, denominator = findFraction(n.real**2)

    # If a fraction for the number cannot be found
    if denominator == 0:
        return "{:.3f}{c}".format(n.real, c=complexString)

    # If fraction is nearly zero
    if numerator / denominator < tolerance:
        return "0{c}".format(c=complexString)
    # If fraction is nearly 1
    if numerator / denominator > (1 - tolerance) and numerator / denominator < (
        1 + tolerance
    ):
        if n.real < 0:
            return "-1{c}".format(c=complexString)
        return "1{c}".format(c=complexString)

    numeratorIsRootable = False
    denominatorIsRootable = False
    if (
        np.sqrt(np.abs(numerator)) % 1 < tolerance
        or (1 - (np.sqrt(np.abs(numerator)) % 1)) < tolerance
    ):
        numeratorIsRootable = True
    if (
        np.sqrt(denominator) % 1 < tolerance
        or (1 - (np.sqrt(denominator) % 1)) < tolerance
    ):
        denominatorIsRootable = True

    numeratorString = (
        str(int(np.sqrt(numerator)))
        if numeratorIsRootable
        else sqrtSymbol + str(int(numerator))
    )
    denominatorString = (
        str(int(np.sqrt(denominator)))
        if denominatorIsRootable
        else sqrtSymbol + str(int(denominator))
    )

    if n.real < 0:
        numeratorString = "-" + numeratorString

    return "{n}/{d}{c}".format(n=numeratorString, d=denominatorString, c=complexString)


vPrettyWaveFunctionAmplitude = np.vectorize(prettyWaveFunctionAmplitude)


def prettyFraction(n) -> str:
    tolerance = 1e-8

    complexString = ""
    if n.imag > tolerance:
        complexString = prettyFraction(n.imag)
        if "-" not in complexString:
            complexString = "+" + complexString
        complexString = complexString + "j"

    if n.real == 0:
        return "0{c}".format(c=complexString)

    numerator, denominator = findFraction(n.real)

    # If a fraction for the number cannot be found
    if denominator == 0:
        return "{r}{c}".format(r=str(n.real), c=complexString)

    # If fraction is nearly zero
    if numerator / denominator < tolerance:
        return "0{c}".format(c=complexString)
    # If fraction is nearly 1
    if numerator / denominator > (1 - tolerance) and numerator / denominator < (
        1 + tolerance
    ):
        if n < 0:
            return "-1{c}".format(c=complexString)
        return "1{c}".format(c=complexString)

    numeratorString = str(numerator)
    denominatorString = str(denominator)

    if n < 0:
        numeratorString = "-" + numeratorString

    return "{n}/{d}{c}".format(n=numeratorString, d=denominatorString, c=complexString)


vPrettyFraction = np.vectorize(prettyFraction)


def make_control_gate(control, target, gate, total_qubits):
    """
    Make a control gate with the specified arguments.

    Args:
        control (int): position of control qubit (1-indexed)
        target (int): position of target qubit (1-indexed)
        gate (string): gate to enact on target qubit
        totalQubits (int): total number of qubits in circuit

    Return:
        QuantumElement: Operator for the control gate
    """
    n = np.repeat("I", total_qubits)
    control_string = ""
    target_string = ""
    for i, qubit in enumerate(n):
        if i + 1 == control:
            control_string = control_string + "(|0><0|)"
            target_string = target_string + "(|1><1|)"
        elif i + 1 == target:
            control_string = control_string + qubit
            target_string = target_string + gate
        else:
            control_string = control_string + qubit
            target_string = target_string + qubit
    return eval(control_string + " + " + target_string)


def tokenizeWaveFunctionString(stringstrong):
    # Tokenize a string
    # Characters to tokenize on: <, >, |, Capitol Letters, spaces
    soloTokenPattern = r"^[\+*-,/()√π ]"
    beginPattern = r"[<A-Z]"
    endPattern = r">"
    vert = "|"
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
            elif currentToken[0] == "<":
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


def eval(psi: str) -> WaveFunctionElement:
    tokens = tokenizeWaveFunctionString(psi)
    return buildWaveFunction(tokens)

def getOperator(op: str) -> WaveFunctionElement:
    return WaveFunctionElement(operators[op], WaveFunctionTokens.OPERATOR)

def getScalar(s: str) -> WaveFunctionElement:
    return WaveFunctionElement(complex(s), WaveFunctionTokens.SCALAR)

def getArithmetic(a: str) -> WaveFunctionElement:
    return WaveFunctionElement(a, WaveFunctionTokens.ARITHMETIC)

def buildWaveFunction(tokens, over_function: str = None):
    patterns = []

    patterns.append(("operator", r"^[A-Z][a-z]*", getOperator))
    patterns.append(("ket", r"^\|[01]+\>$", buildKet))
    patterns.append(("bra", r"^\<[01]+\|", buildBra))
    patterns.append(("scalar", r"^[0-9.j]+$", getScalar))
    patterns.append(("neg scalar", r"^-[0-9.j]+$", getScalar))
    patterns.append(("arithmetic", r"^[+\-*/]$", getArithmetic))
    parenPattern = r"^[(,)]$"

    openParenStack = []
    overallStack = []
    currentTermStack = []

    token_element = None
    prev_type = None
    current_function = None
    expected_args = 0
    if over_function is not None:
        if over_function not in wavefunction_functions:
            print(f"ERROR: Function {over_function} is not a known function.")
            return None
        expected_args = wavefunction_functions[over_function][0]
        openParenStack.append(-1)

    if DEBUG:
        print("building " + str(tokens))

    # Figure out what type each token in and add it into the current term stack as a
    # as a QuantumElement
    # The order of these pattern matche matter because there are specific patterns
    # farther up that will also match more general patterns below.
    for i, token in enumerate(tokens):
        token_handled = False # Once the current token is considered to be handled, this is set to true
        if re.search(parenPattern, token):
            if DEBUG:
                print(f"paren {token}")
            if token == "(":
                openParenStack.append(i)
                token_handled = True
            elif token == ",":
                # Only handle comma if at top level, which is when open paren stack has 1
                if len(openParenStack) == 1 and over_function is not None:
                    # Evaluate from the previous comma to the comma, which will eval the argument
                    openingParenIndex = openParenStack.pop()
                    arg = buildWaveFunction(tokens[openingParenIndex + 1 : i])
                    overallStack.append(arg)
                    openParenStack.append(i)
                    expected_args -= 1
                token_handled = True
            elif token == ")":
                if len(openParenStack) == 0:
                    print("ERROR: Got a closing paren without a matching opening paren")
                    return None
                # Only handle the outermost parens, inner parens will be handled by the recursive call
                openingParenIndex = openParenStack.pop()
                if len(openParenStack) == 0:
                    # Make a recursive call to this function to handle the stuff inside the parens
                    token_element = buildWaveFunction(
                        tokens[openingParenIndex + 1 : i], current_function
                    )
                    current_function = None
                else:
                    token_handled = True
        elif len(openParenStack) > 0:
            token_handled = True
        elif token in wavefunction_functions:
            if DEBUG:
                print("function")
            # Keep track of the function, it will be put onto the stack after the args
            current_function = token
            token_handled = True
        elif token in knownScalars.keys():
            if DEBUG:
                print("scalar")
            token_element = WaveFunctionElement(knownScalars[token], WaveFunctionTokens.SCALAR)
        else:
            # Figure out token based on pattern
            for pattern in patterns:
                if re.search(pattern[1], token):
                    if DEBUG:
                        print(f"Identified {token} as {pattern[0]}")
                    token_element = pattern[2](token)                    
                    break
        if token_handled:
            continue
        if token_element is None:
            print(f"ERROR: token not recognized: {token}")
            return None
        if prev_type is not None and token_element.type != prev_type:
            overallStack.append(evaluateStack(currentTermStack))
            currentTermStack = []
        currentTermStack.append(token_element)
        prev_type = token_element.type


    # Evaluate the full stack and what is left over in the overall stack
    # overallStack.append(evaluateStack(currentTermStack))
    if over_function is not None:
        if expected_args != 1:
            print(
                f"ERROR: Incorrect number of arguments for {over_function}, expected {wavefunction_functions[over_function][0]}"
            )
        else:
            openingParenIndex = openParenStack.pop()
            overallStack.append(buildWaveFunction(tokens[openingParenIndex + 1 :]))
            overallStack.append(
                WaveFunctionElement(over_function, WaveFunctionTokens.FUNCTION)
            )
    if len(openParenStack) > 0:
        print("ERROR: Unclosed parenthesis")
    return evaluateStack(overallStack + currentTermStack)


def evaluateStack(stack):
    if DEBUG:
        print("Evaluating stack: ")
        for i in stack:
            print(i)
    while len(stack) > 1:
        right = stack.pop()
        if right.type == WaveFunctionTokens.FUNCTION:
            args = []
            for i in range(wavefunction_functions[right.data][0]):
                args.append(stack.pop())
            args.reverse()
            stack.append(wavefunction_functions[right.data][1](*args))
            continue
        left = stack.pop()
        arithmetic = None
        result = None
        if (
            left.type == WaveFunctionTokens.ARITHMETIC
            and left.data not in wavefunction_functions.keys()
        ):
            arithmetic = left
            left = stack.pop()
            result = evaluateExplicit(left=left, arithmetic=arithmetic, right=right)
        else:
            result = evaluateImplicit(left=left, right=right)
        stack.append(result)

    rtn = stack.pop()
    if DEBUG:
        print("Evaluated stack as: " + str(rtn))
    return rtn


def evaluateExplicit(left, arithmetic, right):
    # Really only need to handle BRA, KET, and SCALAR in this method
    match arithmetic.data:
        case "+":
            return left + right
        case "-":
            return left - right
        case "*":
            return left * right
        case "/":
            return left / right

    print(
        "Something was not handled, evaluateExplicit. Left:{l} Arithmetic:{a} Right:{r}".format(
            l=str(left), r=str(right), a=str(arithmetic)
        )
    )


def evaluateImplicit(left, right):
    # For two operators together, use a kron product
    if (
        left.type == WaveFunctionTokens.OPERATOR
        and right.type == WaveFunctionTokens.OPERATOR
    ):
        return left & right
    return left * right


def readInWaveFunction(psi):
    tokens = tokenizeWaveFunctionString(psi)
    evaluatedPsi = buildWaveFunction(tokens)
    return toString(evaluatedPsi.data)


def exponentiateMatrix(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    tmp = np.diag(np.exp(eigenvalues))
    return eigenvectors @ tmp @ np.linalg.inv(eigenvectors)


def make_control_gate_tokens(acontrol: int, atarget: int, agate, atotal_qubits: int):
    """
    Make a control gate where the gate passed in is a matrix.

    Args:
        control (int): position of control qubit (1-indexed)
        target (int): position of target qubit (1-indexed)
        gate (np.ndarray): Matrix representation of 1-qubit operator to use on target bit
        totalQubits (int): total number of qubits in circuit

    Return:
        QuantumElement: Operator for the control gate
    """
    control = int(acontrol.data.real)
    target = int(atarget.data.real)
    gate = agate.data
    total_qubits = int(atotal_qubits.data.real)
    control_arr = []
    target_arr = []
    for i in range(1, total_qubits + 1):
        if i == control:
            control_arr.append(eval("|0><0|").data)
            target_arr.append(eval("|1><1|").data)
        elif i == target:
            control_arr.append(np.eye(2))
            target_arr.append(gate)
        else:
            control_arr.append(np.eye(2))
            target_arr.append(np.eye(2))
    return WaveFunctionElement(
        chainedKron(control_arr) + chainedKron(target_arr), WaveFunctionTokens.OPERATOR
    )


def sqrt(a: WaveFunctionElement):
    return WaveFunctionElement(np.sqrt(a.data), a.type)


def exponentiate_matrix_wv(a: WaveFunctionElement):
    return WaveFunctionElement(exponentiateMatrix(a.data), a.type)


def rx(theta: WaveFunctionElement):
    return WaveFunctionElement(
        exponentiateMatrix(-1j * theta.data / 2 * pauli_X),
        WaveFunctionTokens.OPERATOR,
    )


def ry(theta: WaveFunctionElement):
    return WaveFunctionElement(
        exponentiateMatrix(-1j * theta.data / 2 * pauli_Y),
        WaveFunctionTokens.OPERATOR,
    )


def rz(theta: WaveFunctionElement):
    return WaveFunctionElement(
        exponentiateMatrix(-1j * theta.data / 2 * pauli_Z),
        WaveFunctionTokens.OPERATOR,
    )


def prob(a: WaveFunctionElement):
    return WaveFunctionElement(a.data * np.conj(a.data), WaveFunctionTokens.SCALAR)


# Supported function. Key is the string representation of the function, the first letter must be capitol, and the rest lowercase.
# The value is a tuple with the left being the number of arguments and the right the function that evaluates it.
wavefunction_functions = {
    "√": (1, sqrt),
    "Sr": (1, sqrt),
    "Exp": (1, exponentiate_matrix_wv),
    "Prob": (1, prob),
    "Rx": (1, rx),
    "Ry": (1, ry),
    "Rz": (1, rz),
    "Ctrl": (4, make_control_gate_tokens),
}
