# main.py
import sys
from loop import run

if __name__ == "__main__":
    if len(sys.argv) < 2:
        task = input("What do you want me to do? ")
    else:
        task = " ".join(sys.argv[1:])

    run(task)
