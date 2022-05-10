#!/usr/bin/env python3
"""
https://github.com/LabNeuroCogDevel/slipstask/tree/345a223242c4d96a1229c60384e5f2f939c534ac/timing
https://afni.nimh.nih.gov/afni/community/board/read.php?1,42880,42890
>  the optimal experimental design is chosen by minimizing the "norm. std. dev.
"""

from anytree import Node
from random import shuffle
from event import Event
from event_maker import EventMaker, task_dsl
from iti import geo_dist
from deconvolve import deconvolve

def shake(l):
    "remove None, empty lists, and raise nested single item lists"
    if type(l) == list:
        l = [shake(x) for x in l if x]
        l = [x for x in l if x]
        if len(l) == 1:
            l = l[0]
    return l



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
    if n_remain <= 0:
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
    dur = 0  # arithmetic sum
    seq = []
    while leaf:
        n = leaf.name  # actually type Event
        leaf = leaf.parent
        # CATCH is dur 0 but the proportion is important
        if n.dur == 0 and n.prop == 1:
            continue
        prop *= n.prop
        dur += n.dur
        seq = [n, *seq]
    return {"prop": prop, "dur": dur, "seq": seq}


def trial_list(root, total_trials):
    """unique trial (sequence) combinations from leaves of tree
    add n property (number of times this sequence should be seen)
    total_trials likely from grammer parser itself
    """
    real_leaves = shake(find_leaves(root))
    clean_tree(root)
    res = [calc_leaf(l) for l in real_leaves]
    print(res)
    res = [{**x, "n": int(x["prop"] * total_trials)} for x in res]
    trials_needed = total_trials - int(sum([x.get("n", 0) for x in res]))
    # todo: warning if need!=0. add additional trials to random leaves
    for i in range(trials_needed):
        # trials_needed shouldn't excede len(res).
        # only need any added b/c rounding issues.
        # but just in case, wrap around
        # TODO: instead we should pick randomly
        ii = i % len(res)
        res[ii]["n"] += 1
    return res


def event_list(root, total_trials, total_duration):
    """
    builds list of dictionary (onset(time), name, dur)
    TODO: use tree structure instead of converting to list
    """
    tlist = trial_list(root, total_trials)
    events_dur = sum([x["n"] * x["dur"] for x in tlist])
    iti_dur = total_duration - events_dur
    itis = geo_dist(total_trials, iti_dur)
    trials = sum(
        [[node["seq"]] * node["n"] for node in tlist], []
    )  # flatten (slow, but easy to write)

    # TODO: shuffle shouldn't repeat an instance more than X times as specified by the grammar
    # need to rework to use tree intead of list?
    # TODO: use seed?
    shuffle(trials)

    # onset accumulates
    onset = 0
    events = []
    for trialnum, t in enumerate(trials):
        for event in t:
            # catch trial. dont need to record this event
            if event.dur == 0:
                continue
            events.append({"onset": onset, "name": event.name, "dur": event.dur})
            onset = onset + event.dur
        events.append({"onset": onset, "name": "iti", "dur": itis[trialnum]})
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
