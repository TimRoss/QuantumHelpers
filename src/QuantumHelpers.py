import numpy as np
import re
import numbers
import unittest
from enum import Enum


# Figure out if matplotlib is installed. If not, plotting is not available
plottingAvailable = False
try:
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
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

ketPattern = r"^\|[0-1]+>"
braPattern = r"^\<[01]+\|"
ketDecPattern = r"^\|[1-9]d\d+\>$"
braDecPattern = r"^\<[1-9]d\d+\|"


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
            return self.print_not_supported()

        if self.type == other.type:
            return WaveFunctionElement(self.data + other.data, self.type)
        else:
            self.print_error(other, "add")

    def __sub__(self, other):
        """
        Overloaded - : Subtract other element from this element. Elements must be of the same type.
        """
        if not isinstance(other, WaveFunctionElement):
            return self.print_not_supported()

        if self.type == other.type:
            return WaveFunctionElement(self.data - other.data, self.type)
        else:
            self.print_error(other, "subtract")

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
            return self.print_not_supported()

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
                        return self.print_error(other, "multiply")
            case WaveFunctionTokens.KET:
                match other.type:
                    case WaveFunctionTokens.BRA:
                        return WaveFunctionElement(
                            np.outer(self.data, other.data), WaveFunctionTokens.OPERATOR
                        )
                    case WaveFunctionTokens.KET:
                        return self & other
                    case _:
                        return self.print_error(other, "multiply")
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
                        return self.print_error(other, "multiply")
            case _:
                return self.print_error(other, "multiply")

    def __truediv__(self, other):
        """
        Overloaded / : Divide by a number or scalar
        """
        # Support dividing by a normal number
        if isinstance(other, numbers.Number):
            return WaveFunctionElement(self.data / other, self.type)
        if not isinstance(other, WaveFunctionElement):
            return self.print_not_supported()

        # Division is only supported with scalars
        if other.type == WaveFunctionTokens.SCALAR:
            return WaveFunctionElement(self.data / other.data, self.type)
        else:
            self.print_error(other, "divide")

    def __and__(self, other):
        """
        Overloaded & : Kronecker product elements together. Types must be the same
        """
        if not isinstance(other, WaveFunctionElement):
            return self.print_not_supported()

        # Since python does not have a kron, use & as kron symbol
        if self.type == other.type and self.type in [
            WaveFunctionTokens.BRA,
            WaveFunctionTokens.KET,
            WaveFunctionTokens.OPERATOR,
        ]:
            return WaveFunctionElement(np.kron(self.data, other.data), self.type)
        else:
            self.print_error(other, "kron")

    def print_error(self, other, operation):
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

    def print_not_supported(self):
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
                return str(pretty_wave_function_amplitude(self.data)).replace("'", "")
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

        num_qubits = int(np.log2(len(self.data)))
        fig_height_per_qubit = 5
        fig_width = 10
        axs = plt.subplots(num_qubits, 2, layout="constrained", figsize=(fig_width, fig_height_per_qubit * num_qubits))
        if num_qubits == 1:
            self._add_state_plot(axs[1][0], 0, False)
            self._add_state_plot(axs[1][1], 0, True)
        else:
            for qubit in range(num_qubits):
                self._add_state_plot(axs[1][num_qubits - (qubit + 1)][0], qubit, False)
                self._add_state_plot(axs[1][num_qubits - (qubit + 1)][1], qubit, True)



    def _add_state_plot(self, ax, qubit_index: int, imag: bool):
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
        norm_circle = Circle((0, 0), 1, linestyle="--", fill=False, color="black")
        ax.add_patch(norm_circle)

        # Draw x-axis
        plt.plot([-2, 2], [-2, 2], "--", color="green")

        # Draw the actual state
        num_states = len(self.data)
        colors = cm.rainbow(np.linspace(0, 1, num_states))
        i = -1
        j = -1
        overall_zero_part = 0
        overall_one_part = 0
        one_parts = []
        zero_parts = []
        part_colors = []
        for state in range(num_states // 2):
            i += 1
            j += 1
            if j == 2**qubit_index:
                i += 2**qubit_index
                j = 0
            zero_part = self.data[i]
            one_part = self.data[i + 2**qubit_index]
            if imag:
                zero_part = zero_part.imag
                one_part = one_part.imag
            else:
                zero_part = zero_part.real
                one_part = one_part.real
            one_parts.append(one_part)
            zero_parts.append(zero_part)
            overall_one_part += one_part
            overall_zero_part += zero_part
            if zero_part > 0:
                part_colors.append(colors[i])
            else:
                part_colors.append(colors[i + 2**qubit_index])
        # normalize
        norm_factor = np.sqrt(overall_one_part**2 + overall_zero_part**2)
        if norm_factor > 0:
            zero_parts = zero_parts / norm_factor
            one_parts = one_parts / norm_factor
        
        overall_zero_part = 0
        overall_one_part = 0
        for state in range(num_states // 2):
            arrow_head_length = np.sqrt(one_parts[state] ** 2 + zero_parts[state] ** 2) * 0.1
            plt.arrow(
                overall_one_part,
                overall_zero_part,
                one_parts[state],
                zero_parts[state],
                color=part_colors[state],
                width=0.02,
                head_width=arrow_head_length,
                head_length=arrow_head_length,
                length_includes_head=True,
            )
            overall_one_part += one_parts[state]
            overall_zero_part += zero_parts[state]

    def print(self):
        if self.type == WaveFunctionTokens.KET:
            print_states(self.data)


def build_ket(a_ket: str) -> WaveFunctionElement:
    """
    Build a WaveFunctionElement of the string ket passed in.

    Args:
        a_ket (string): String representation of ket in (assumed) binary or (specified with xd) decimal. Ex. |001> or |3d1>
    """
    ket_string = ""
    if re.match(ketDecPattern, a_ket):
        ket_string = str(bin(int(a_ket[3:4])))[2:].zfill(int(a_ket[1:2]))
    elif re.match(ketPattern, a_ket):
        ket_string = a_ket[1:-1]
    else:
        print(f"Argument passed to buildKet does not match expected ket format. Got {a_ket}")
        return None
    local_ket = 1
    # Goes through each character from the argument excluding the start and end characters
    for i in ket_string:
        local_ket = np.kron(local_ket, unitKets[int(i)])
    return WaveFunctionElement(local_ket, WaveFunctionTokens.KET)


def build_bra(a_bra: str) -> WaveFunctionElement:
    """
    Build a WaveFunctionElement of the string bra passed in.

    Args:
        a_bra (string): String representation of bra in (assumed) binary or (specified with xd) decimal. Ex. <001| or <3d1|
    """
    bra_string = ""
    if re.match(braDecPattern, a_bra):
        bra_string = str(bin(int(a_bra[3:4])))[2:].zfill(int(a_bra[1:2]))
    elif re.match(braPattern, a_bra):
        bra_string = a_bra[1:-1]
    else:
        print(f"Argument passed to buildBra does not match expected bra format. Got {a_bra}")
        return None
    local_bra = 1
    # Goes through each character from the argument excluding the start and end characters
    for i in bra_string:
        local_bra = np.kron(local_bra, unitKets[int(i)])
    return WaveFunctionElement(local_bra, WaveFunctionTokens.BRA)


def print_states(a_ket):
    number_of_qubits = int(np.log2(a_ket.size))
    for state, coefficient in enumerate(a_ket):
        if coefficient == 0:
            continue
        print(
            "{c} |{s}>".format(
                c=pretty_wave_function_amplitude(coefficient),
                s=bin(state)[2:].zfill(number_of_qubits),
            )
        )


def to_string(a_ket):
    if type(a_ket) == tuple:
        match a_ket.type:
            case WaveFunctionTokens.SCALAR:
                return str(a_ket.data)
            case WaveFunctionTokens.OPERATOR:
                return vPrettyWaveFunctionAmplitude(a_ket.data)
            case WaveFunctionTokens.KET:
                return to_string(a_ket.data)
            case WaveFunctionTokens.BRA:
                return to_string(a_ket.data)
    number_of_qubits = int(np.log2(a_ket.size))
    psi = ""
    for state, coefficient in enumerate(a_ket):
        if coefficient == 0:
            continue
        if len(psi) > 0:
            psi = psi + " + "
        psi = psi + "{c} |{s}>".format(
            c=pretty_wave_function_amplitude(coefficient),
            s=bin(state)[2:].zfill(number_of_qubits),
        )
    return psi


# Density Matrix
def make_density_matrix(wave_function):
    number_of_qubits = int(np.log2(wave_function.size))
    total_density = np.zeros((wave_function.size, wave_function.size))
    current_outer_state = -1
    current_inner_state = -1
    for outer_state in wave_function:
        current_outer_state = current_outer_state + 1
        local_bra = build_bra(
            "<" + bin(current_outer_state)[2:].zfill(number_of_qubits) + "|"
        )
        local_bra = local_bra * outer_state
        current_inner_state = -1
        for inner_state in wave_function:
            current_inner_state = current_inner_state + 1
            local_ket = build_ket(
                "|" + bin(current_inner_state)[2:].zfill(number_of_qubits) + ">"
            )
            local_ket = local_ket * inner_state
            state_density = np.outer(local_bra, local_ket)
            total_density = total_density + state_density
    return total_density


def chained_kron(a_list_to_kron):
    local_kron = 1
    for i in a_list_to_kron:
        local_kron = np.kron(local_kron, i)
    return local_kron


def find_fraction(n: float | complex) -> tuple[int, int] | tuple[int, int, int, int]:
    max_denominator = 16
    tolerance = 1e-8

    is_complex = True if type(n) == complex else False
    is_negative = False

    # Local variables to keep track of return values
    denominator = 0
    numerator = 0
    imag_numerator = 0
    image_denominator = 0

    # If the passed in value in complex, make a recursive call to find the fraction for the imaginary part
    if is_complex:
        imag_numerator, image_denominator = find_fraction(n.imag)
        is_negative = n.real < 0
        p = np.abs(n.real)
    else:
        is_negative = n < 0
        p = np.abs(n)

    # Check some edge cases and return fast if n is 0 or one
    if p < tolerance and p >= 0:
        return (0, 0) if not is_complex else (0, 0, imag_numerator, image_denominator)
    if p < 1 + tolerance and p > 1 - tolerance:
        return (1, 1) if not is_complex else (1, 1, imag_numerator, image_denominator)

    # Brute force check every possible numerator for each denominator between 0 and maxDenom
    for denom in range(1, max_denominator + 1):
        if numerator != 0:
            break
        for numer in reversed(range(1, denom)):
            distanceFromInt = ((p / numer) * denom) % 1
            if distanceFromInt < tolerance or (1 - distanceFromInt) < tolerance:
                if np.abs((numer / denom) - p) < tolerance:
                    numerator = numer
                    denominator = denom
                    break

    if is_negative:
        numerator = numerator * -1

    if is_complex:
        return numerator, denominator, imag_numerator, image_denominator
    else:
        return numerator, denominator


def pretty_wave_function_amplitude(n) -> str:
    tolerance = 1e-8
    sqrt_symbol = "\u221A"

    complex_string = ""
    if np.abs(n.imag) > tolerance:
        complex_string = pretty_wave_function_amplitude(n.imag)
        if n.imag > 0:
            complex_string = "+" + complex_string
        complex_string = complex_string + "j"

    if abs(n.real) < tolerance:
        return "0{c}".format(c=complex_string)

    numerator, denominator = find_fraction(n.real**2)

    # If a fraction for the number cannot be found
    if denominator == 0:
        return "{:.3f}{c}".format(n.real, c=complex_string)

    # If fraction is nearly zero
    if numerator / denominator < tolerance:
        return "0{c}".format(c=complex_string)
    # If fraction is nearly 1
    if numerator / denominator > (1 - tolerance) and numerator / denominator < (
        1 + tolerance
    ):
        if n.real < 0:
            return "-1{c}".format(c=complex_string)
        return "1{c}".format(c=complex_string)

    numerator_is_rootable = False
    denominator_is_rootable = False
    if (
        np.sqrt(np.abs(numerator)) % 1 < tolerance
        or (1 - (np.sqrt(np.abs(numerator)) % 1)) < tolerance
    ):
        numerator_is_rootable = True
    if (
        np.sqrt(denominator) % 1 < tolerance
        or (1 - (np.sqrt(denominator) % 1)) < tolerance
    ):
        denominator_is_rootable = True

    numerator_string = (
        str(int(np.sqrt(numerator)))
        if numerator_is_rootable
        else sqrt_symbol + str(int(numerator))
    )
    denominator_string = (
        str(int(np.sqrt(denominator)))
        if denominator_is_rootable
        else sqrt_symbol + str(int(denominator))
    )

    if n.real < 0:
        numerator_string = "-" + numerator_string

    return "{n}/{d}{c}".format(n=numerator_string, d=denominator_string, c=complex_string)


vPrettyWaveFunctionAmplitude = np.vectorize(pretty_wave_function_amplitude)


def pretty_fraction(n) -> str:
    tolerance = 1e-8

    complex_string = ""
    if n.imag > tolerance:
        complex_string = pretty_fraction(n.imag)
        if "-" not in complex_string:
            complex_string = "+" + complex_string
        complex_string = complex_string + "j"

    if n.real == 0:
        return "0{c}".format(c=complex_string)

    numerator, denominator = find_fraction(n.real)

    # If a fraction for the number cannot be found
    if denominator == 0:
        return "{r}{c}".format(r=str(n.real), c=complex_string)

    # If fraction is nearly zero
    if numerator / denominator < tolerance:
        return "0{c}".format(c=complex_string)
    # If fraction is nearly 1
    if numerator / denominator > (1 - tolerance) and numerator / denominator < (
        1 + tolerance
    ):
        if n < 0:
            return "-1{c}".format(c=complex_string)
        return "1{c}".format(c=complex_string)

    numerator_string = str(numerator)
    denominator_string = str(denominator)

    if n < 0:
        numerator_string = "-" + numerator_string

    return "{n}/{d}{c}".format(n=numerator_string, d=denominator_string, c=complex_string)


vPrettyFraction = np.vectorize(pretty_fraction)


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


def tokenize_wave_function_string(stringstrong: str):
    '''
    Tokenize the string passed in into tokens that can be converted into WaveFunctionElements
    Args:
        stringstrong (str): String to be tokenized. This string should represent a wavefunction.
    '''
    solo_token_pattern = r"^[\+*-,/()√π? ]"
    begin_token_pattern = r"[<A-Z]"
    end_token_pattern = r">"
    vert = "|"
    tokens = []
    current_token = ""
    for character in stringstrong:
        start_of_token = False
        end_of_token = False
        if re.search(solo_token_pattern, character) is not None:
            start_of_token = True
            end_of_token = True
        elif re.search(begin_token_pattern, character) is not None:
            start_of_token = True
        elif re.search(end_token_pattern, character) is not None:
            end_of_token = True
        elif character == vert:
            if current_token == "":
                start_of_token = True
            elif current_token[0] == "<":
                end_of_token = True
            else:
                start_of_token = True

        # Handle the tokens and currentToken for if it is the start, end, or middle of the token
        if start_of_token and end_of_token:
            if current_token != "":
                tokens.append(current_token)
            if character != " ":
                tokens.append(character)
            current_token = ""
        elif start_of_token:
            if current_token != "":
                tokens.append(current_token)
            current_token = character
        elif end_of_token:
            current_token = current_token + character
            tokens.append(current_token)
            current_token = ""
        else:
            current_token = current_token + character

    # If there is anything left at the end, add it to tokens
    if current_token != "":
        tokens.append(current_token)
    return tokens


def eval(psi: str, *insert_elements) -> WaveFunctionElement:
    tokens = tokenize_wave_function_string(psi)
    insert_elements_stack = list(insert_elements)
    insert_elements_stack.reverse() # reverse so that elements can be popped off
    return build_wave_function(tokens, insert_elements=insert_elements_stack)

def build_operator(op: str) -> WaveFunctionElement:
    return WaveFunctionElement(operators[op], WaveFunctionTokens.OPERATOR)

def build_scalar(s: str) -> WaveFunctionElement:
    return WaveFunctionElement(complex(s), WaveFunctionTokens.SCALAR)

def build_arithmetic(a: str) -> WaveFunctionElement:
    return WaveFunctionElement(a, WaveFunctionTokens.ARITHMETIC)

def build_wave_function(tokens:list, insert_elements:list = None, over_function: str = None):
    patterns = []

    patterns.append(("operator", r"^[A-Z][a-z]*", build_operator))
    patterns.append(("ket", ketPattern, build_ket))
    patterns.append(("ket", ketDecPattern, build_ket))
    patterns.append(("bra", braPattern, build_bra))
    patterns.append(("bra", braDecPattern, build_bra))
    patterns.append(("scalar", r"^[0-9.j]+$", build_scalar))
    patterns.append(("neg scalar", r"^-[0-9.j]+$", build_scalar))
    patterns.append(("arithmetic", r"^[+\-*/]$", build_arithmetic))
    paren_pattern = r"^[(,)]$"

    open_paren_stack = []
    overall_stack = []
    current_term_stack = []

    token_element = None
    prev_type = None
    current_function = None
    expected_args = 0
    if over_function is not None:
        if over_function not in wavefunction_functions:
            print(f"ERROR: Function {over_function} is not a known function.")
            return None
        expected_args = wavefunction_functions[over_function][0]

        # Add a paren onto the stack if this we are evaluating the arguments for a function
        # Each argument gets evaluated and placed on the stack individually
        open_paren_stack.append(-1)

    if DEBUG:
        print("building " + str(tokens))

    # Figure out what type each token in and add it into the current term stack as a
    # as a QuantumElement
    # The order of these pattern matche matter because there are specific patterns
    # farther up that will also match more general patterns below.
    for i, token in enumerate(tokens):
        token_handled = False # Once the current token is considered to be handled, this is set to true
        if re.search(paren_pattern, token):
            if DEBUG:
                print(f"paren {token}")
            if token == "(":
                open_paren_stack.append(i)
                token_handled = True
            elif token == ",":
                # Only handle comma if at top level, which is when open paren stack has 1
                if len(open_paren_stack) == 1 and over_function is not None:
                    # Evaluate from the previous comma to the comma, which will eval the argument
                    opening_paren_index = open_paren_stack.pop()
                    arg = build_wave_function(tokens[opening_paren_index + 1 : i], insert_elements=insert_elements)
                    overall_stack.append(arg)
                    open_paren_stack.append(i)
                    expected_args -= 1
                token_handled = True
            elif token == ")":
                if len(open_paren_stack) == 0:
                    print("ERROR: Got a closing paren without a matching opening paren")
                    return None
                # Only handle the outermost parens, inner parens will be handled by the recursive call
                opening_paren_index = open_paren_stack.pop()
                if len(open_paren_stack) == 0:
                    # Make a recursive call to this function to handle the stuff inside the parens
                    token_element = build_wave_function(
                        tokens[opening_paren_index + 1 : i], insert_elements=insert_elements, over_function= current_function
                    )
                    current_function = None
                else:
                    token_handled = True
        elif len(open_paren_stack) > 0:
            token_handled = True
        elif token in wavefunction_functions:
            if DEBUG:
                print("function")
            # Keep track of the function, it will be put onto the stack after the args
            current_function = token
            token_handled = True
        elif token == "?":
            if len(insert_elements) == 0:
                print("ERROR: Ran out of elements to insert")
                return None
            token_element = insert_elements.pop()
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
        # Handle token by putting the token_element onto the stack.
        if token_handled:
            continue
        if token_element is None:
            print(f"ERROR: token not recognized: {token}")
            return None
        # If they type is changing from the last token, evaluate the current stack and then stick onto overall stack
        # This basically defines an order of operations by keeping similar types together
        if prev_type is not None and token_element.type != prev_type:
            overall_stack.append(evaluate_stack(current_term_stack))
            current_term_stack = []
        current_term_stack.append(token_element)
        prev_type = token_element.type


    # Evaluate the full stack and what is left over in the overall stack
    # overallStack.append(evaluateStack(currentTermStack))
    if over_function is not None:
        if expected_args != 1:
            print(
                f"ERROR: Incorrect number of arguments for {over_function}, expected {wavefunction_functions[over_function][0]}"
            )
        else:
            opening_paren_index = open_paren_stack.pop()
            overall_stack.append(build_wave_function(tokens[opening_paren_index + 1 :]))
            overall_stack.append(
                WaveFunctionElement(over_function, WaveFunctionTokens.FUNCTION)
            )
    if len(open_paren_stack) > 0:
        print("ERROR: Unclosed parenthesis")
    return evaluate_stack(overall_stack + current_term_stack)


def evaluate_stack(stack):
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
            result = evaluate_explicit(left=left, arithmetic=arithmetic, right=right)
        else:
            result = evalute_implicit(left=left, right=right)
        stack.append(result)

    rtn = stack.pop()
    if DEBUG:
        print("Evaluated stack as: " + str(rtn))
    return rtn


def evaluate_explicit(left, arithmetic, right):
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


def evalute_implicit(left, right):
    # For two operators together, use a kron product
    if (
        left.type == WaveFunctionTokens.OPERATOR
        and right.type == WaveFunctionTokens.OPERATOR
    ):
        return left & right
    return left * right


def read_in_wave_function(psi):
    tokens = tokenize_wave_function_string(psi)
    evaluated_psi = build_wave_function(tokens)
    return to_string(evaluated_psi.data)


def exponentiate_matrix(a):
    eigenvalues, eigenvectors = np.linalg.eig(a)
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
        chained_kron(control_arr) + chained_kron(target_arr), WaveFunctionTokens.OPERATOR
    )


def sqrt(a: WaveFunctionElement):
    return WaveFunctionElement(np.sqrt(a.data), a.type)


def exponentiate_matrix_wv(a: WaveFunctionElement):
    return WaveFunctionElement(exponentiate_matrix(a.data), a.type)


def rx(theta: WaveFunctionElement):
    return WaveFunctionElement(
        exponentiate_matrix(-1j * theta.data / 2 * pauli_X),
        WaveFunctionTokens.OPERATOR,
    )


def ry(theta: WaveFunctionElement):
    return WaveFunctionElement(
        exponentiate_matrix(-1j * theta.data / 2 * pauli_Y),
        WaveFunctionTokens.OPERATOR,
    )


def rz(theta: WaveFunctionElement):
    return WaveFunctionElement(
        exponentiate_matrix(-1j * theta.data / 2 * pauli_Z),
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
