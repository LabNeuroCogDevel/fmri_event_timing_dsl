#!/usr/bin/env python3
"""
build a tree from a task event DSL
descend tree to build a list of random events

200/10x@1.3 A=1.5,{B=1.5,0.15*C=1.5,$}<3>,C=1.5;=1.5
  ###/#x   total duraiton/number repeats
  =#.#     duration
  {,}      permutation, tree siblings
  $        catch end
  #.#*     proportion of permutaitons
  <#>      max repeats
  ,        add next in sequence
  ;=#.#    variable iti w/stepsize; end of root node

thoughts on duration
  =#.#...#.#  = variable duration
  =#...#(exp) = var exponential

need additional node max/min to when specified for repeats

TODO: add 3dDeconvolve model (BLOCK, GAM, TENT) to grammar

https://github.com/LabNeuroCogDevel/slipstask/tree/345a223242c4d96a1229c60384e5f2f939c534ac/timing
https://afni.nimh.nih.gov/afni/community/board/read.php?1,42880,42890
>  the optimal experimental design is chosen by minimizing the "norm. std. dev.
"""

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from anytree import Node, RenderTree
from random import shuffle
import numpy as np
import numpy.random as rnd
from warnings import warn
import subprocess
import os
import re
task_dsl = Grammar(
    """
    main  = info anyevent+

    info         = total_dur "/" total_trials "x" ("@" tr)? " "
    total_dur    = float
    total_trials = num
    tr           = float

    anyevent    = ( children / event / catch_end / iti )
    in_children = ( children / event / catch_end / iti )
    children    = "{" in_children+ "}" maxrep? sep?
    catch_end   = "$"
    event       = reps? prop? name dur? maxrep? sep?
    iti = ";" dur?

    name        = ~"[A-Za-z0-9.:_-]+"i
    reps        = num "x"
    prop        = float "*"
    dur         = "=" float dur_range? dur_type?
    dur_range   = "..." float
    dur_type    = "(exp)"
    maxrep      = "<" num ">"
    sep         = ","
    float       = ~"[0-9.]+"
    num         = ~"[0-9]+"
    """
)


class EventMaker(NodeVisitor):
    def visit_main(self, node, children):
        "final collection at 'main' after through all children"
        out = []
        for c in children:
            if not c:
                continue
            if c and type(c) in [list, dict]:
                out.append(c)
        return out

    def visit_info(self, node, children):
        "side effect: update self.total_duraiton and total_trials"
        assert node.expr_name == 'info'
        self.total_duration = children[0]
        self.total_trials = children[2]
        self.info = children
        # TODO: clean up gammar here
        if len(children) >= 4 and children[4]:
            self.tr = children[4][0][1]
        return None

    def visit_event(self, node, children):
        "main node. collects reps? prop? name dur? maxrep? sep?"
        event = {"dur": 1, "descend": True}
        for c in children:
            if not c:
                continue
            elif type(c) == list:
                print(c)
                if c[0] and type(c[0]) == dict:
                    event.update(c[0])
                continue
            elif not c.expr_name:
                continue
            print(c)
            key = c.expr_name
            if key == "name":
                value = c.text
            # these are never hit?
            elif key == "prop":
                value = c.children[0]
            elif key == "dur":
                value = c["dur"]
            elif key == "reps":
                value = c.children[0]
            elif key == "maxrep":
                value = c.children[1]
            else:
                raise Exception(f"unknown event attribute '{key}': '{c}'")
            event[key] = value
        return event

    def visit_catch_end(self, node, children):
        "when grammar sees '$', the tree should not be followed down"
        assert not children
        return {'name': "CATCH", "dur": 0, "descend": False}

    def visit_dur(self, node, children):
        # TODO: duration range
        # TODO: duration type
        dur = children[1]
        return {'dur': dur}

    def visit_reps(self, node, children):
        reps = children[1]
        # child[1] is 'x' but now None
        return {'reps': reps}

    def visit_prop(self, node, children):
        # child[1] was '*' but now None
        return {'prop': children[0]}

    def visit_float(self, node, children):
        assert not children
        return float(node.text)

    def visit_num(self, node, children):
        assert not children
        return int(node.text)

    def visit_sep(self, node, children): return None

    def visit_maxsep(self, node, children):
        assert not children
        return int(node.text)

    def generic_visit(self, node, children):
        # remove literals used in grammar
        if node.text in ["{", "}", ">", "<", '=', '*']:
            return None
        # remove empty optionals
        if not children and not node.text and not node.expr_name:
            return None
        # things we haven't identified yet (anyevent, in_children)
        return children or node


def shake(l):
    "remove None, empty lists, and raise nested single item lists"
    if type(l) == list:
        l = [shake(x) for x in l if x]
        l = [x for x in l if x]
        if len(l) == 1:
            l = l[0]
    return l


class Event():
    def __init__(self, d: dict):
        self.dur = d.get("dur", 0)
        self.name = d.get("name")
        self.prop = d.get("prop")
        self.descend = d.get("descend", True)
        self.maxrep = d.get("maxrep", -1)

    def __repr__(self):
        disp = f"{self.name}={self.dur}"
        if self.prop:
            disp = f"{self.prop}*{disp}"
        return disp


def build_tree(input_list: list, roots: list = None, append=False):
    """
    progressively add leaves to a tree. returns the final leaf/leaves
    [A,B,C] = A->B->C     (returns [C])
    [A,[B,C],D] = A->B->D
                   ->C->D (returns [D,D])
    """
    if not roots:
        roots = [Node(Event({"name": "root", "prop": 1}))]

    leaves = []
    for n in input_list:
        if type(n) == list:
            print(f"#** {n} is list. recurse")
            roots = build_tree(n, roots, append=True)
        elif append:
            for p in roots:
                if p.name.descend:
                    leaves.append(Node(Event(n), parent=p))
        else:
            roots = [Node(Event(n), parent=p) for p in roots if p.name.descend]
            leaves = roots
        print(f"# leaves: {len(leaves)} @ {n}")
    return leaves


def set_children_proption(node):
    """update node's children 'prop'
    the proportion of trials with this sequence. defaults to symetic"""
    n = len(node.children)
    existing = [x.name.prop for x in node.children if x.name.prop is not None]
    n_remain = n - len(existing)
    if(n_remain <= 0):
        return
    start = 1 - sum(existing)
    eq_dist = start / n_remain
    for n in node.children:
        if n.name.prop is None:
            n.name.prop = eq_dist
    # TODO: assert value is close to 1


def clean_tree(node):
    "set proprotions for entire tree"
    set_children_proption(node)
    for n in node.children:
        clean_tree(n)


def find_leaves(node):
    "inefficent. use with shake. find catch leaves"
    if not node.children:
        return node
    return [find_leaves(n) for n in node.children]


def calc_leaf(leaf):
    """
    summary metrics for leaves (final event sequences)
    into event sequence instead of tree
    remove no-duration events(root, catch)
    output dict: prop (final proportion of total trials)
                 dur  (total duration of event sequence)
                 seq  (start-finish list of Events)
    """
    prop = 1  # geometric  product
    dur  = 0  # arithmetic sum
    seq = []
    while leaf:
      n = leaf.name # actually type Event
      leaf = leaf.parent
      # CATCH is dur 0 but the proportion is important
      if n.dur == 0 and n.prop == 1:
          continue
      prop *= n.prop
      dur  += n.dur
      seq = [n, *seq]
    return {"prop":prop,"dur":dur,"seq":seq}


def trial_list(root, total_trials):
    """ unique trial (sequence) combinations from leaves of tree
    add n property (number of times this sequence should be seen)
    total_trials likely from grammer parser itself
    """
    real_leaves = shake(find_leaves(root))
    clean_tree(root)
    res = [calc_leaf(l) for l in real_leaves]
    print(res)
    res = [{**x, "n": int(x['prop']*total_trials)} for x in res]
    trials_needed = total_trials - int(sum([x.get("n",0) for x in res]))
    # todo: warning if need!=0. add additional trials to random leaves
    for i in range(trials_needed):
        # trials_needed shouldn't excede len(res).
        # only need any added b/c rounding issues.
        # but just in case, wrap around
        # TODO: instead we should pick randomly
        ii = i%len(res)
        res[ii]["n"] += 1
    return res

# todo. repeat each seq "n" times. intersperce with itis. generate 1d files
def uncount(res):
    out = []
    for seq in res:
        for i in range(seq["n"]):
            out.append({k: seq[k] for k in ["seq","dur"]})
    return out


def geo_dist_discrete(n_iti, total_iti_dur, step=None, mn=1, mx=6):
    "TODO: unfinished"
    if step:
        steps = np.arange(mn,mx+step,step)
    else:
        n_bins = np.log2(n_iti)
        mean_iti = total_iti_dur/n_iti

def geo_dist(n_iti, total_iti_dur, mn=.5, mx=6, seed=None):
    """random exp limited in range by mn to mx
     will meet total_iti_dur +/- mn/2 and then try to adjust"""
    if not seed:
        seed = rnd.default_rng()
    itis = []
    if n_iti * mn > total_iti_dur:
        raise Exception("total iti duration is too short for even all itis = minimum length")
    if n_iti * mx < total_iti_dur:
        #raise Exception(f"total iti duration {total_iti_dur}s too large to fill with {mx}s*{n_iti}trials")
        warn(f"total iti duration {total_iti_dur}s too large to fill with {mx}s*{n_iti}trials")
    while sum(itis) < total_iti_dur and len(itis) < n_iti:
        x = seed.exponential(scale=(mx-mn)/2, size=1)
        if x < mn or x > mx:
            continue
        itis.append(x[0])
    # kludge: run recusively if we're short on trials
    if len(itis) != n_iti:
        itis = geo_dist(n_iti, total_iti_dur, mn, mx, seed)

    # remove too large first before filling too small
    # order (fill then remove) shouldn't matter (?)
    while sum(itis) > total_iti_dur:
        large_enough = [i for i in range(len(itis)) if itis[i] >= 2*mn ]
        i = rnd.choice(large_enough)
        itis[i] = itis[i] - mn

    # fill up to iti_duration by adding min size
    # min size could be large. this should maybe be another parameter?
    while sum(itis) < total_iti_dur - mn/2:
        small_enough = [i for i in range(len(itis)) if itis[i] < (mx -mn)]
        if not small_enough:
            warn(f"ITI durations cannot backfill. have {sum(itis)} < {total_iti_dur} total. but no itis < {mx - mn}")
            break
        #if len(small_enough) < total_iti_dur - sum(itis):
        #    raise Exception("have to many large ITIs. cannot backfill to get to max duration")
        i = rnd.choice(small_enough)
        itis[i] = itis[i] + mn

    # fill the mn/2 hole left by the the prev two while loops
    gap = total_iti_dur - sum(itis)
    goldylocks = [i for i in range(len(itis)) if itis[i] + mn < mx and itis[i] > mn*2]
    if goldylocks:
        i = rnd.choice(goldylocks)
        itis[i] = itis[i] + gap
    return(itis)



def event_list(root, total_trials, total_duration):
    """
    builds list of dictionary (onset(time), name, dur)
    TODO: use tree structure instead of converting to list
    """
    tlist = trial_list(root, total_trials)
    events_dur = sum([x['n']*x['dur'] for x in tlist])
    iti_dur = total_duration - events_dur
    itis = geo_dist(total_trials, iti_dur)
    trials = sum([[node['seq']]*node['n'] for node in tlist],[]) # flatten (slow, but easy to write)

    # TODO: shuffle shouldn't repeat an instance more than X times as specified by the grammar
    # need to rework to use tree intead of list?
    # TODO: use seed?
    shuffle(trials)

    # onset accumulates
    onset = 0
    events = []
    for trialnum,t in enumerate(trials):
        for event in t:
            # catch trial. dont need to record this event
            if event.dur == 0:
                continue
            events.append({'onset': onset, 'name': event.name, 'dur': event.dur})
            onset = onset + event.dur
        events.append({'onset': onset, 'name': 'iti', 'dur': itis[trialnum]})
        onset += itis[trialnum]
    return events

def grammar_to_event(desc):
    """
    returns event list, root node, and parsed grammer
     desc="70/10x A=1.5,D,{.15*B=.5,C=.3,$},D"
    """
    egrammar = EventMaker()
    gtree = egrammar.visit(task_dsl.parse(desc))
    etree = build_tree(shake(gtree))
    events = event_list(etree[0].root, egrammar.total_trials, egrammar.total_duration)
    return (events, etree[0].root, egrammar)

def deconvolve(d, total_dur, tr, syms=[], outpath="/tmp"):
    """
    run 3dDeconvolve with nodata

    syms = ["+A -B", "+.5*D +.5*C -E"]
    """
    os.makedirs(outpath+"/1d/", exist_ok=True)
    d.to_csv(outpath+"/event_timing.csv")
    event_names = [x for x in np.unique(d.name) if x != 'iti']
    event_fnames = [f"{outpath}/1d/{n}.1D" for n in event_names]
    for (n, fname) in zip(event_names, event_fnames):
        event_trials = d[d.name == n]
        points = [f"{x.onset:0.02f}:{x.dur:0.02f}"
                for x in event_trials.itertuples()]
        with open(fname, "w") as event_fh:
            event_fh.write("\t".join(points))
    decon_cmd = ["3dDeconvolve", "-nodata", int(total_dur/tr), tr]
    decon_cmd += ["-polort", 3 ]
    decon_cmd += ["-x1D", f"{outpath}/X.xmat.1D"]
    decon_cmd += ["-GOFORIT", 99]
    decon_cmd += ["-num_stimts" ,len(event_names)]
    for i,e in enumerate(event_names):
        decon_cmd += ["-stim_times_AM1", i+1, event_fnames[i], 'dmBLOCK(1)']
        decon_cmd += ["-stim_label", i+1, e ]

    if syms:
        decon_cmd += ["-num_glt", len(syms)]
        for i,s in enumerate(syms):
            decon_cmd += ["-gltsym", f"SYM: {s}"]
            decon_cmd += ["-glt_label", i+1, f'{s.replace(" ","")}']

    with open(f"{outpath}/decon_out.txt","w") as f:
        subprocess.run([str(x) for x in decon_cmd], stdout=f, stderr=f)
    corr_1d_tool_cmd = f"cd {outpath}; 1d_tool.py -cormat_cutoff 0.1 -show_cormat_warnings -infile X.xmat.1D 2>timing_cor.warn > timing_cor.txt"
    os.system(corr_1d_tool_cmd)

def decon_stddev(fname):
    """extract test norm std dev from deconvolve nodata output
     min is best. also see timing correlations (timing_cor.txt) from 1d_tool.py"""
    tests = []
    cur_name = "XX"
    with open(fname,"r") as fh:
        for l in fh.readlines():
            name = re.search("^(Stimulus|General Linear Test): (.*)", l)
            value = re.search("(h|LC)\[.*norm. std. dev. = +([0-9.-]+)", l)
            if name:
                #print(name)
                cur_name = name.group(2)
            elif value:
                print(l)
                tests.append({'name': cur_name, 'type': value.group(1), 'value': value.group(2)})
    return tests



example = task_dsl.parse("100/10x test")
task_dsl.parse("100/10x test=2.5")
task_dsl.parse("100/10x test=2.5<2>")
task_dsl.parse("100/10x .3*test=2.5<2>")
task_dsl.parse("100/10x .3*test=2.5<2>;")
task_dsl.parse("100/10x {one,two}")
task_dsl.parse("100/10x {one,two,{three}}")
task_dsl.parse("100/10x {one,two},three")
task_dsl.parse("100/10x A=1.5,D,{.3*B=.5,C=.3,$},D")

ep = EventMaker()
o = ep.visit(example)
# ep.total_duration
# ep.total_trials
x = ep.visit(task_dsl.parse("100/10x A=1.5,D,{.15*B=.5,C=.3,$},D"))
print(shake(x))

t = build_tree(shake(x))
root = t[0].root
print(RenderTree(root))

real_leaves = shake(find_leaves(root))
clean_tree(root)
print(RenderTree(root))
print(real_leaves)

(events, rootnode, egrammar) = grammar_to_event("50/10x@1.5 A=1.5,B,{.15*C=.5,D=.3,$},E")
import pandas as pd
d = pd.DataFrame(events)
deconvolve(d, egrammar.total_duration, egrammar.tr, outpath="/tmp/seed1", syms=["+A -B", "+.5*D +.5*C -E"])
