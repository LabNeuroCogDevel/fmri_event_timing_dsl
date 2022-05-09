import numpy as np
from warnings import warn
import numpy.random as rnd


def geo_dist_discrete(n_iti, total_iti_dur, step=None, mn=1, mx=6):
    "TODO: unfinished"
    if step:
        steps = np.arange(mn, mx + step, step)
    else:
        n_bins = np.log2(n_iti)
        mean_iti = total_iti_dur / n_iti


def geo_dist(n_iti, total_iti_dur, mn=0.5, mx=6, seed=None):
    """random exp limited in range by mn to mx
    will meet total_iti_dur +/- mn/2 and then try to adjust"""
    if not seed:
        seed = rnd.default_rng()
    itis = []
    if n_iti * mn > total_iti_dur:
        raise Exception(
            "total iti duration is too short for even all itis = minimum length"
        )
    if n_iti * mx < total_iti_dur:
        # not an Exception b/c we might be okay adding iti time at the end
        warn(f"total iti duration {total_iti_dur}s too large to fill with" +
             f"{mx}s*{n_iti}trials")
    while sum(itis) < total_iti_dur and len(itis) < n_iti:
        x = seed.exponential(scale=(mx - mn) / 2, size=1)
        if x < mn or x > mx:
            continue
        itis.append(x[0])

    # kludge: run recusively if we're short on trials
    if len(itis) != n_iti:
        itis = geo_dist(n_iti, total_iti_dur, mn, mx, seed)

    # remove too large first before filling too small
    # order (fill then remove) shouldn't matter (?)
    while sum(itis) > total_iti_dur:
        large_enough = [i for i in range(len(itis)) if itis[i] >= 2 * mn]
        i = rnd.choice(large_enough)
        itis[i] = itis[i] - mn

    # fill up to iti_duration by adding min size
    # min size could be large. this should maybe be another parameter?
    while sum(itis) < total_iti_dur - mn / 2:
        small_enough = [i for i in range(len(itis)) if itis[i] < (mx - mn)]
        if not small_enough:
            warn("ITI durations cannot backfill. " +
                 f"have {sum(itis)} < {total_iti_dur} total." +
                 f"but no itis < {mx - mn}")
            break
        # if len(small_enough) < total_iti_dur - sum(itis):
        #    raise Exception("have to many large ITIs. cannot backfill to get to max duration")
        i = rnd.choice(small_enough)
        itis[i] = itis[i] + mn

    # fill the mn/2 hole left by the the prev two while loops
    gap = total_iti_dur - sum(itis)
    goldilocks = [i for i in range(len(itis))
                  if itis[i] + mn < mx and itis[i] > mn * 2]
    if goldilocks:
        i = rnd.choice(goldilocks)
        itis[i] = itis[i] + gap
    return itis
