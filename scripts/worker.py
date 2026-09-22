import sys


def main():
    stdin = sys.stdin
    stdout = sys.stdout

    while True:
        line = stdin.readline()

        if not line:
            return 0

        fields = line.split()

        if len(fields) != 2:
            print(f"worker: expected 2 values, got {len(fields)}: {line!r}", file=sys.stderr, flush=True)
            return 1

        x1 = float(fields[0])
        x2 = float(fields[1])

        f1 = (x1 - 1.0) * (x1 - 1.0) + (x2 - 2.0) * (x2 - 2.0)
        f2 = (x1 + 1.0) * (x1 + 1.0) + (x2 + 2.0) * (x2 + 2.0)

        stdout.write(f"{f1!r} {f2!r}\n")
        stdout.flush()


if __name__ == '__main__':
    sys.exit(main())