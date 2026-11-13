"""`sentinel-semantic` command: run the API server."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="sentinel-semantic")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    uvicorn.run("sentinel_semantic.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
