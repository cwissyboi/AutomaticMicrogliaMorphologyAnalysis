from utils import parse_args


def main():
    args = parse_args()

    print(args["cfg"])
    print(args["conf_threshold"])


if __name__ == "__main__":
    main()
