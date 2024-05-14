To run tests: Run `python -m unittest discover -s test -p "*"`

To run specific test: Run `python -m unittest discover -s test -p filename -k testName`
Ex. `python -m unittest discover -s test -p BuildWaveFunctionTests.py -k BuildWaveFunctionTests.test_buildControlMatrix`