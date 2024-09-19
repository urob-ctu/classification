from .test import test
from .submit import submit

def hello_world():
    import sys
    # Access arguments passed to the script
    args = sys.argv[1:]
    print("Hello, world!")
    print("Arguments:", args)