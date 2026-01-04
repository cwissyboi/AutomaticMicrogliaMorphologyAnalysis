def main():
    print("Job is working!!!!!!")

    try:
        import pandas as pd
        print("pandas is available")
        print("pandas version:", pd.__version__)
    except ImportError:
        print("pandas is NOT available")


if __name__ == "__main__":
    main()
