# ---
# role: [core]
# purpose: Main entry point
# dependencies: []
# mutation_ready: true
# test_status: [ci_passed]
# ---
from .core import Engine

def main():
    Engine()

if __name__ == "__main__":
    main()
