"""Command-line interface for the semantic photo search system.

Two subcommands:
    python cli.py build-index
    python cli.py search "your query" -k 5

"""
import argparse

from src.config import Config
from src import search as search_mod


def cmd_build_index(_args: argparse.Namespace) -> None:
    """Handle the `build-index` subcommand."""
    cfg = Config()
    print(f"Building index from {cfg.images_dir} ...")
    search_mod.build_index(cfg)
    print(f"Index written to {cfg.index_path}")


def cmd_search(args: argparse.Namespace) -> None:
    """Handle the `search` subcommand."""
    cfg = Config()
    results = search_mod.search(args.query, cfg, k=args.k)
    if not results:
        print("No results.")
        return
    for path, score in results:
        print(f"{score:.4f}  {path}")


def main() -> None:
    """Parse CLI arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="cli.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Build FAISS index from images")
    p_build.set_defaults(func=cmd_build_index)

    p_search = sub.add_parser("search", help="Search images by text query")
    p_search.add_argument("query", type=str, help="text query")
    p_search.add_argument("-k", type=int, default=None, help="top-k results")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()