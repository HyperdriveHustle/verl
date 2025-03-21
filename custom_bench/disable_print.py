import io

import sys
import contextlib


# Create a context manager to redirect stdout only during the function call
@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def work():
    print('work work work')
    return 1


def main():

    result = work()
    print('results: 1, ', result)
    print('=' * 100)

    with suppress_stdout():
        result = work()
    print('results: 2: ', result)
    print('=' * 100)

    # Now you have the return value in 'result' and anything that would have been
    # printed is in 'printed_output' (which you can ignore if you don't need it)

    result = work()
    print('results: 3, ', result)
    print('=' * 100)


if __name__ == "__main__":
    main()
