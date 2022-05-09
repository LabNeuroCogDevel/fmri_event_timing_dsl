from EventMaker import task_dsl, EventMaker
from grammar import build_tree, shake, clean_tree
from anytree import RenderTree


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


import pandas as pd
(events, rootnode, egrammar) = grammar_to_event(
    "50/10x@1.5 A=1.5,B,{.15*C=.5,D=.3,$},E"
)

d = pd.DataFrame(events)
deconvolve(
    d,
    egrammar.total_duration,
    egrammar.tr,
    outpath="/tmp/seed1",
    syms=["+A -B", "+.5*D +.5*C -E"],
)
