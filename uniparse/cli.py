"""UniParse Command Line Application."""

import argparse
import logging
from importlib import import_module

from uniparse.model_runners.template_runner import train, run
from uniparse.models import INCLUDED_PARSERS
MAINPARSER_HELP = "print current version number"
SUBPARSERS_HELP = "%(prog)s must be called with a command:"

VERSION = 0.1


def main():
    """Set up command line parser."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")

    description = "Scripts for Extracting, Transforming and Loading documents."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s {}".format(VERSION), help=MAINPARSER_HELP)

    subparsers = parser.add_subparsers(help=SUBPARSERS_HELP, dest='command')
    subparsers.required = True

    for subparsers_func in [_train_parser, _run_parser]:
        subparsers_func(subparsers)

    args, unk = parser.parse_known_args()
    args.func(parser, args)


def _train_parser(subparsers):
    """Create parser for the 'enqueue' command."""
    parser = subparsers.add_parser("train", help="train dependency parser model.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--model", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--vocab")
    parser.set_defaults(func=_train)
    return parser


def _train(_, args):
    """Continuously retrieve new document IDs and add them to queue."""
    implementation = args.implementation.split(":")
    if len(implementation) == 1:
        # attempt to from included parser implementations
        model_class = INCLUDED_PARSERS.get(args.implementation, None)
        if not model_class:
            print("included parsers are: %s" % " ".join(INCLUDED_PARSERS.keys()))
            raise ValueError("Implementation doesn't match any")

    else:
        if "/" in args.implementation:
            print(">> Warning. Path includes ")
        print("Retrieving class-like object %s from module %s" % (implementation[1], implementation[0]))
        model_class = getattr(import_module(implementation[0]), implementation[1])

    train(args.train, args.dev, args.test, args.n_epochs, args.model, args.vocab, model_class)


def _run_parser(subparsers):
    """Create parser for the 'import' command."""
    parser = subparsers.add_parser("run", help="import documents using queue.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--vocab", required=True)
    parser.set_defaults(func=_run)
    return parser


def _run(_, args, unks):
    """Run parser model from saved parameters and vocab file."""
    implementation = args.implementation.split(":")
    if len(implementation) == 1:
        # attempt to retrieve included parser implementations
        model_class = INCLUDED_PARSERS.get(args.implementation, None)
        if not model_class:
            print("included parsers are: %s" % " ".join(INCLUDED_PARSERS.keys()))
            raise ValueError("Implementation doesn't match any")

    else:
        model_class = getattr(import_module(implementation[0]), implementation[1])

    run(args.dataset, args.model, args.vocab, model_class)


if __name__ == "__main__":
    main()
