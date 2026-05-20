import sys

from main_train import main


if __name__ == "__main__":
    if "--architecture" not in sys.argv:
        sys.argv.extend(["--architecture", "bert"])
    main()
