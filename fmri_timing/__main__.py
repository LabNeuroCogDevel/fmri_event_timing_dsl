import pandas as pd
from .event_tree import grammar_to_event
from .deconvolve import deconvolve
import sys
from argparse import ArgumentParser, ArgumentError

def get_args(argv):
    parser=ArgumentParser(exit_on_error=False)
    parser.add_argument("-e", help="event layout")
    parser.add_argument("--syms", help="GLT tests for deconvolve", nargs="+")
    parser.add_argument("--outdir", help="out/path/loc", default="/tmp/timing_example")
    parser.add_argument("--seed", help="", default=None, type=int)
    args = parser.parse_args(argv)
    return args

#get_args("-e 50/10x@1.5 A=1.5,B,{.15*C=.5,D=.3,$},E --syms A B".split())

def run(args):
    (events, rootnode, egrammar) = grammar_to_event(args.e)

    d = pd.DataFrame(events)
    deconvolve(
        d,
        egrammar.total_duration,
        egrammar.tr,
        outpath=args.outdir,
        syms=args.syms
    )

def main():
    if len(sys.argv) == 1:
        args = get_args(["-h"])
    else:
        args = get_args(sys.argv[1:])
    run(args)

if __name__ == "__main__":
    main()
