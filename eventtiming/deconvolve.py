import numpy as np
import os
import subprocess
import re
import pandas as pd

def deconvolve(d, total_dur : float, tr : float, syms=[], outpath="/tmp"):
    """
    run 3dDeconvolve with nodata on event timing dataframe

    syms = ["+A -B", "+.5*D +.5*C -E"]
    """
    os.makedirs(outpath + "/1d/", exist_ok=True)
    d.to_csv(outpath + "/event_timing.csv")
    event_names = [x for x in np.unique(d.name) if x != "iti"]
    event_fnames = [f"{outpath}/1d/{n}.1D" for n in event_names]
    for (n, fname) in zip(event_names, event_fnames):
        event_trials = d[d.name == n]
        points = [f"{x.onset:0.02f}:{x.dur:0.02f}" for x in event_trials.itertuples()]
        with open(fname, "w") as event_fh:
            event_fh.write("\t".join(points))
    decon_cmd = ["3dDeconvolve", "-nodata", int(total_dur / tr), tr]
    decon_cmd += ["-polort", 3]
    decon_cmd += ["-x1D", f"{outpath}/X.xmat.1D"]
    decon_cmd += ["-GOFORIT", 99]
    decon_cmd += ["-num_stimts", len(event_names)]
    for i, e in enumerate(event_names):
        decon_cmd += ["-stim_times_AM1", i + 1, event_fnames[i], "dmBLOCK(1)"]
        decon_cmd += ["-stim_label", i + 1, e]

    if syms:
        decon_cmd += ["-num_glt", len(syms)]
        for i, s in enumerate(syms):
            decon_cmd += ["-gltsym", f"SYM: {s}"]
            decon_cmd += ["-glt_label", i + 1, f'{s.replace(" ","")}']

    with open(f"{outpath}/decon_out.txt", "w") as f:
        subprocess.run([str(x) for x in decon_cmd], stdout=f, stderr=f)
    corr_1d_tool_cmd = f"cd {outpath}; 1d_tool.py -cormat_cutoff 0.1 -show_cormat_warnings -infile X.xmat.1D 2>timing_cor.warn > timing_cor.txt"
    os.system(corr_1d_tool_cmd)


def decon_stddev(fname):
    """extract test norm std dev from deconvolve nodata output
    min is best. also see timing correlations (timing_cor.txt) from 1d_tool.py"""
    tests = []
    cur_name = "XX"
    with open(fname, "r") as fh:
        for l in fh.readlines():
            name = re.search("^(Stimulus|General Linear Test): (.*)", l)
            value = re.search("(h|LC)\[.*norm. std. dev. = +([0-9.-]+)", l)
            if name:
                # print(name)
                cur_name = name.group(2)
            elif value:
                print(l)
                tests.append(
                    {"name": cur_name, "type": value.group(1), "value": value.group(2)}
                )
    return tests
