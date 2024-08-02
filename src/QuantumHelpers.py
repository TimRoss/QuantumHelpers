import numpy as np
import re
import numbers
from enum import Enum
from colorama import Style, Fore


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

class ErrorTypes(Enum):
    TYPE_ERROR = 1
    SHAPE_ERROR = 2

class WaveFunctionTokens(Enum):
    """Types of tokens for the wavefunction tokens. These will determine the interactions between different elements."""

    BRA = 1
    KET = 2
    OPERATOR = 3
    SCALAR = 4
    ARITHMETIC = 5
    FUNCTION = 6
    ERROR = 7


class WaveFunctionElement:
    """Holds an element of quantum information. Operators are overloaded to determine interaction based on type."""

    data = []
    type: WaveFunctionTokens
    token: str
    valid: bool

    def __init__(self, data, type: WaveFunctionTokens, token: str = None) -> None:
        self.data = data
        self.type = type
        self.token = token

        if self.type == WaveFunctionTokens.ERROR:
            self.data = Fore.RED + "ERROR:\t" + Fore.RESET + self.data
            self.print()

        match self.type:
            case WaveFunctionTokens.BRA:
                self.__validate_ket__()
            case WaveFunctionTokens.KET:
                self.__validate_ket__()
            case WaveFunctionTokens.OPERATOR:
                self.__validate_operator__()
            case _:
                self.valid = True

    def __validate_ket__(self):
        if not isinstance(self.data, np.ndarray):
            print(
                f"INVALID: Ket/Bra data expected to be a numpy array, but is: {type(self.data)}"
            )
            self.valid = False
            return
        num_dimensions = len(self.data.shape)
        if num_dimensions != 1:
            print(
                f"INVALID: Ket/Bra data expected to be 1-dimensional, got {num_dimensions} dimension(s)"
            )
            self.valid = False
            return
        num_states = self.data.shape[0]
        if not is_power_of_2_x01(num_states):
            print(f"INVALID: Expected a number of states to be a power of 2, got {num_states} states / elements in data array.")
            self.valid = False
            return
        self.valid = True
        
    def __validate_operator__(self):
        if not isinstance(self.data, np.ndarray):
            print(
                f"INVALID: Operator data expected to be a numpy array, but is: {type(self.data)}"
            )
            self.valid = False
            return
        num_dimensions = len(self.data.shape)
        if num_dimensions != 2:
            print(
                f"INVALID: Operator data expected to be 2-dimensional, got {num_dimensions} dimension(s)"
            )
            self.valid = False
            return
        if self.data.shape[0] != self.data.shape[1]:
            print(f"INVALID: Operator expected to be a square array, but is {self.data.shape}")
            self.valid = False
            return
        num_states = self.data.shape[0]
        if not is_power_of_2_x01(num_states):
            print(f"INVALID: Expected a number of states to be a power of 2, got {num_states} states.")
            self.valid = False
            return
        self.valid = True


    def __add__(self, other):
        """
        Overloaded + : Add 2 elements together. Elements must be of the same type.
        """
        if not isinstance(other, WaveFunctionElement):
            return self.print_not_supported()

        if self.type == other.type:
            if self.data.shape[0] != other.data.shape[0]:
                return self.handle_error(other, "add", "+", ErrorTypes.SHAPE_ERROR)
            return WaveFunctionElement(self.data + other.data, self.type)
        else:
            return self.handle_error(other, "add", "+", ErrorTypes.TYPE_ERROR)

    def __sub__(self, other):
        """
        Overloaded - : Subtract other element from this element. Elements must be of the same type.
        """
        if not isinstance(other, WaveFunctionElement):
            return self.print_not_supported()

        if self.type == other.type:
            return WaveFunctionElement(self.data - other.data, self.type)
        else:
            return self.handle_error(other, "subtract", "-", ErrorTypes.TYPE_ERROR)

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
                        return self.handle_error(other, "multiply", "*", ErrorTypes.TYPE_ERROR)
            case _:
                return self.handle_error(other, "multiply", "*", ErrorTypes.TYPE_ERROR)

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

    def handle_error(self, other, operation_name: str, operation_symbol: str, error_type: ErrorTypes):
        if self.type == WaveFunctionTokens.ERROR:
            return self
        elif other.type == WaveFunctionTokens.ERROR:
            return other
    
        errstr = ""
        if self.token is not None and other.token is not None:
            errstr += f"ISSUE: {self.token} {operation_symbol} {other.token}\n"

        match error_type:
            case ErrorTypes.TYPE_ERROR:
                errstr += (
                    f"\tCannot {operation_name} {self.type.name} and {other.type.name}"
                )
            case ErrorTypes.SHAPE_ERROR:
                errstr += (
                    f"\t Cannot {operation_name} shape {self.data.shape} and {self.data.shape}"
                )
        return WaveFunctionElement(
            errstr,
            WaveFunctionTokens.ERROR,
        )

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
            case WaveFunctionTokens.ERROR:
                return self.data
            case _:
                return "Type: {type} to string not implemented".format(type=type)

    def dagger(self):
        """
        Return a hermetian conjugate of the element.
        """
        new_type = self.type
        if self.type == WaveFunctionTokens.BRA:
            new_type = WaveFunctionTokens.KET
        elif self.type == WaveFunctionTokens.KET:
            new_type = WaveFunctionTokens.BRA

        return WaveFunctionElement(np.conj(np.transpose(self.data)), new_type)

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

        axs = plt.subplots(
            num_qubits,
            2,
            layout="constrained",
            figsize=(fig_width, fig_height_per_qubit * num_qubits),
        )
        if num_qubits == 1:
            axs[1][0].set_title("real")
            axs[1][1].set_title("imaginary")
            self._add_state_plot(axs[1][0], 0, False)
            self._add_state_plot(axs[1][1], 0, True)
        else:
            axs[1][0][0].set_title("real")
            axs[1][0][1].set_title("imaginary")
            for qubit in range(num_qubits):
                self._add_state_plot(axs[1][num_qubits - (qubit + 1)][0], qubit, False)
                self._add_state_plot(axs[1][num_qubits - (qubit + 1)][1], qubit, True)

    def _add_state_plot(self, ax, qubit_index: int, imag: bool):
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
        colors = self._get_colors_for_states()
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
            arrow_head_length = (
                np.sqrt(one_parts[state] ** 2 + zero_parts[state] ** 2) * 0.1
            )
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

    def _get_colors_for_states(self):
        """
        Get an array of color values matching the states to be drawn.
        Only chooses color values for non_zero states so that more distinct colors are chosen.

        Return:
            ndarray of color values. Shape is num_states x 4. The 4 values are RGBA
        """
        tolerance = 1e-8
        non_zero_indexes = []
        for i, state in enumerate(self.data):
            if state > tolerance:
                non_zero_indexes.append(i)

        colors = cm.rainbow(np.linspace(0, 1, len(non_zero_indexes)))
        state_colors = np.full((len(self.data), 4), (0.0, 0.0, 0.0, 0.0))
        for color_index, state_index in enumerate(non_zero_indexes):
            state_colors[state_index] = colors[color_index]
        return state_colors

    def print(self):
        if self.type == WaveFunctionTokens.KET:
            print_states(self.data)
        elif self.type == WaveFunctionTokens.ERROR:
            print(self.data)


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
        return WaveFunctionElement("Argument passed to buildKet does not match expected ket format.", WaveFunctionTokens.ERROR, token=a_ket)
    local_ket = 1
    # Goes through each character from the argument excluding the start and end characters
    for i in ket_string:
        local_ket = np.kron(local_ket, unitKets[int(i)])
    return WaveFunctionElement(local_ket, WaveFunctionTokens.KET, a_ket)


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
        return WaveFunctionElement("Argument passed to buildBra does not match expected bra format.", WaveFunctionTokens.ERROR, token=a_bra)
    local_bra = 1
    # Goes through each character from the argument excluding the start and end characters
    for i in bra_string:
        local_bra = np.kron(local_bra, unitKets[int(i)])
    return WaveFunctionElement(local_bra, WaveFunctionTokens.BRA, a_bra)


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
    if isinstance(a_ket, tuple):
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
            distance_from_int = ((p / numer) * denom) % 1
            if distance_from_int < tolerance or (1 - distance_from_int) < tolerance:
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

    return "{n}/{d}{c}".format(
        n=numerator_string, d=denominator_string, c=complex_string
    )


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

    return "{n}/{d}{c}".format(
        n=numerator_string, d=denominator_string, c=complex_string
    )


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
    """
    Tokenize the string passed in into tokens that can be converted into WaveFunctionElements
    Args:
        stringstrong (str): String to be tokenized. This string should represent a wavefunction.
    """
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
    insert_elements_stack.reverse()  # reverse so that elements can be popped off
    return build_wave_function(tokens, insert_elements=insert_elements_stack)


def build_operator(op: str) -> WaveFunctionElement:
    return WaveFunctionElement(operators[op], WaveFunctionTokens.OPERATOR, op)


def build_scalar(s: str) -> WaveFunctionElement:
    return WaveFunctionElement(complex(s), WaveFunctionTokens.SCALAR, s)


def build_arithmetic(a: str) -> WaveFunctionElement:
    return WaveFunctionElement(a, WaveFunctionTokens.ARITHMETIC, a)


def build_wave_function(
    tokens: list, insert_elements: list = None, over_function: str = None
):
    # The order of these pattern matches matter because there are specific patterns
    # farther up that will also match more general patterns below.
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
            return WaveFunctionElement(
                f"Function {over_function} is not a known function.",
                WaveFunctionTokens.ERROR,
                over_function,
            )
        expected_args = wavefunction_functions[over_function][0]

        # Add a paren onto the stack if we are evaluating the arguments for a function
        # Each argument gets evaluated and placed on the stack individually
        open_paren_stack.append(-1)

    if DEBUG:
        print("building " + str(tokens))

    # Figure out what type each token is and add it into the current term stack as a
    # as a QuantumElement
    for i, token in enumerate(tokens):
        token_handled = False  # Once the current token is considered to be handled, this is set to true
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
                    arg = build_wave_function(
                        tokens[opening_paren_index + 1 : i],
                        insert_elements=insert_elements,
                    )
                    overall_stack.append(arg)
                    open_paren_stack.append(i)
                    expected_args -= 1
                token_handled = True
            elif token == ")":
                if len(open_paren_stack) == 0:
                    return WaveFunctionElement(
                        "Got a closing paren without a matching opening paren",
                        WaveFunctionTokens.ERROR,
                        ")",
                    )
                # Only handle the outermost parens, inner parens will be handled by the recursive call
                opening_paren_index = open_paren_stack.pop()
                if len(open_paren_stack) == 0:
                    # Make a recursive call to this function to handle the stuff inside the parens
                    token_element = build_wave_function(
                        tokens[opening_paren_index + 1 : i],
                        insert_elements=insert_elements,
                        over_function=current_function,
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
                return WaveFunctionElement(
                    "Ran out of elements to insert after ?", WaveFunctionTokens.ERROR
                )
            token_element = insert_elements.pop()
        elif token in knownScalars.keys():
            if DEBUG:
                print("scalar")
            token_element = WaveFunctionElement(
                knownScalars[token], WaveFunctionTokens.SCALAR, token
            )
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
            return WaveFunctionElement(
                f"Token not recognized: {token}", WaveFunctionTokens.ERROR, token
            )
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
            return WaveFunctionElement(
                f"Incorrect number of arguments for {over_function}, expected {wavefunction_functions[over_function][0]}",
                WaveFunctionTokens.ERROR,
                over_function,
            )
        else:
            opening_paren_index = open_paren_stack.pop()
            overall_stack.append(build_wave_function(tokens[opening_paren_index + 1 :]))
            overall_stack.append(
                WaveFunctionElement(
                    over_function, WaveFunctionTokens.FUNCTION, over_function
                )
            )
    if len(open_paren_stack) > 0:
        return WaveFunctionElement(
            "Unclosed parenthesis", WaveFunctionTokens.ERROR, "("
        )
    return evaluate_stack(overall_stack + current_term_stack)


def evaluate_wavefunction_function(afunction_token, stack):
    args = []
    try:
        for _ in range(wavefunction_functions[afunction_token.data][0]):
            args.append(stack.pop())
        args.reverse()
        # Block below is where wavefunction function is actually called
        function_result = None
        try:
            function_result = wavefunction_functions[afunction_token.data][1](
                *args, atrigger_token=afunction_token.data
            )
        except TypeError:
            # This may have happened because the function does not take a atrigger_token argument, try again without that.
            try:
                function_result = wavefunction_functions[afunction_token.data][1](*args)
            except TypeError:
                # Try again but this time with the data of each argument instead of WaveFunctionElement
                # This will allow calling functions from other libararies
                args = [a.data for a in args]
                function_result = wavefunction_functions[afunction_token.data][1](*args)

        if not isinstance(function_result, WaveFunctionElement):
            if isinstance(function_result, np.ndarray):
                if len(function_result.shape) == 1:
                    function_result = WaveFunctionElement(
                        function_result, WaveFunctionTokens.KET, afunction_token.data
                    )
                elif len(function_result.shape) == 2:
                    function_result = WaveFunctionElement(
                        function_result,
                        WaveFunctionTokens.OPERATOR,
                        afunction_token.data,
                    )
            else:
                try:
                    function_result = WaveFunctionElement(
                        complex(function_result),
                        WaveFunctionTokens.SCALAR,
                        afunction_token.data,
                    )
                except Exception:
                    function_result = WaveFunctionElement(
                        "Could not turn function result into WaveFunctionElement. Function token: "
                        + afunction_token.data,
                        WaveFunctionTokens.ERROR,
                        afunction_token.data,
                    )

        stack.append(function_result)
    except Exception as e:
        print(f"--- ERROR in function call triggered from {afunction_token} ---")
        raise e


def evaluate_stack(stack):
    if DEBUG:
        print("Evaluating stack: ")
        for i in stack:
            print(i)
    while len(stack) > 1:
        right = stack.pop()
        if right.type == WaveFunctionTokens.FUNCTION:
            evaluate_wavefunction_function(right, stack)
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


def print_error(msg: str):
    print(f"ERROR: {msg}")


def make_control_gate_tokens(
    acontrol: WaveFunctionElement,
    atarget: WaveFunctionElement,
    agate: WaveFunctionElement,
    atotal_qubits: WaveFunctionElement,
    atrigger_token: str = None,
) -> WaveFunctionElement:
    """
    Make a control gate where the gate passed in is an operator

    Usage:
        Ctrl(int, int, operator, int)
        Ex. Ctrl(1,2,X,3)
            Creates a 3-qubit controlled X-gate with qubit 1 as the control bit,
              qubit 2 as the target, and nothing happening to the third qubit.

    Args:
        control (WaveFunctionElement): Scalar - position of control qubit (1-indexed)
        target (WaveFunctionElement): Scalar - position of target qubit (1-indexed)
        gate (WaveFunctionElement): Operator - Matrix representation of 1-qubit operator to use on target bit
        totalQubits (WaveFunctionElement): Scalar - total number of qubits for control gate

    Return:
        QuantumElement: Operator for the control gate
    """
    help_msg = "See Ctrl function documentation for usage by using help(make_control_gate_tokens)"
    trigger_msg = (
        f"Triggered from {atrigger_token} function --\n"
        if atrigger_token is not None
        else None
    )
    # Verify arguments
    if (
        acontrol.type != WaveFunctionTokens.SCALAR
        or atarget.type != WaveFunctionTokens.SCALAR
        or agate.type != WaveFunctionTokens.OPERATOR
        or atotal_qubits.type != WaveFunctionTokens.SCALAR
    ):
        print_error(f"{trigger_msg} Invalid arguments for Ctrl function. \n {help_msg}")
        print(f"HERE: {acontrol.type} {atarget.type} {agate.type} {atotal_qubits.type}")
    if acontrol.data == 0 or atarget.data == 0 or atotal_qubits.data == 0:
        print_error(
            f"{trigger_msg} Qubit index arguments for Ctrl function are 1-indexed. 0 is not valid. \n {help_msg}"
        )

    if (
        acontrol.data.real > atotal_qubits.data.real
        or atarget.data.real > atotal_qubits.data.real
    ):
        print_error(
            f"{trigger_msg} Control and Targets qubits must be less than or equal to total qubits."
        )

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
        chained_kron(control_arr) + chained_kron(target_arr),
        WaveFunctionTokens.OPERATOR,
    )


def sqrt(a: WaveFunctionElement, atrigger_token: str = None):
    return WaveFunctionElement(np.sqrt(a.data), a.type)


def exponentiate_matrix_wv(a: WaveFunctionElement, atrigger_token: str = None):
    return WaveFunctionElement(exponentiate_matrix(a.data), a.type)


def rx(theta: WaveFunctionElement, atrigger_token: str = None):
    return WaveFunctionElement(
        exponentiate_matrix(-1j * theta.data / 2 * pauli_X),
        WaveFunctionTokens.OPERATOR,
    )


def ry(theta: WaveFunctionElement, atrigger_token: str = None):
    return WaveFunctionElement(
        exponentiate_matrix(-1j * theta.data / 2 * pauli_Y),
        WaveFunctionTokens.OPERATOR,
    )


def rz(theta: WaveFunctionElement, atrigger_token: str = None):
    return WaveFunctionElement(
        exponentiate_matrix(-1j * theta.data / 2 * pauli_Z),
        WaveFunctionTokens.OPERATOR,
    )


def prob(a: WaveFunctionElement, atrigger_token: str = None):
    return WaveFunctionElement(a.data * np.conj(a.data), WaveFunctionTokens.SCALAR)

def is_power_of_2_x01(n: int):
    '''
    Determines if an integer is a power of 2, but 0 and 1 are not considered powers of 2 because they are not valid ket sizes.

    Return:
        True if the number is a power of 2, False if not or 1 or 0
    '''
    if n == 0 or n == 1:
        return False
    return (n & (n - 1)) == 0


# Supported functions. Key is the string representation of the function, the first letter must be capitol, and the rest lowercase.
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
    "Abs": (1, np.abs),
}
