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
"""
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar



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
        assert node.expr_name == "info"
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
        return {"name": "CATCH", "dur": 0, "descend": False}

    def visit_dur(self, node, children):
        # TODO: duration range
        # TODO: duration type
        dur = children[1]
        return {"dur": dur}

    def visit_reps(self, node, children):
        reps = children[1]
        # child[1] is 'x' but now None
        return {"reps": reps}

    def visit_prop(self, node, children):
        # child[1] was '*' but now None
        return {"prop": children[0]}

    def visit_float(self, node, children):
        assert not children
        return float(node.text)

    def visit_num(self, node, children):
        assert not children
        return int(node.text)

    def visit_sep(self, node, children):
        return None

    def visit_maxsep(self, node, children):
        assert not children
        return int(node.text)

    def generic_visit(self, node, children):
        # remove literals used in grammar
        if node.text in ["{", "}", ">", "<", "=", "*"]:
            return None
        # remove empty optionals
        if not children and not node.text and not node.expr_name:
            return None
        # things we haven't identified yet (anyevent, in_children)
        return children or node
