This libarary provides the ability to evaluate a wave function written as a string to see the result. It is intended to help with learning and exploring quantum computing by providing a way to easily write wave functions and see what they evaluate to. See the Demo.ipynb file for many examples of what this libary can do.
To use: Download the QuantumHelpers.py file from the src folder and then import wherever it is wanted. This library will hopefully be published as a package one day.

Note: This library evaluates by using full-size statevectors, and may not appropriate for large numbers of qubits.

To run tests: Run `python -m unittest discover -s test -p "*"`

To run specific test: Run `python -m unittest discover -s test -p filename -k testName`
Ex. `python -m unittest discover -s test -p BuildWaveFunctionTests.py -k BuildWaveFunctionTests.test_buildControlMatrix`
