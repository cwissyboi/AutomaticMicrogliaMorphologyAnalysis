import argparse

def main():
    parser = argparse.ArgumentParser(description="Example main with arguments")
    parser.add_argument("--name", type=str, default="world", help="Your name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print(f"Hello {args.name}")
    print(f"Epochs: {args.epochs}")
    print(f"Debug: {args.debug}")

if __name__ == "__main__":
    main()
