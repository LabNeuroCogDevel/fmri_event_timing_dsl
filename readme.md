# Event related fMRI experimental design timing generator

Generate sequential events organized by trial with interspersed inter trial intervals (e.g. fixation cross) and catch trials.
Evaluate with AFNI's 3dDeconvolve.

## See Also
 * [`optseq2`](https://surfer.nmr.mgh.harvard.edu/fswiki/optseq2)
 * [`make_random_timing.py`](https://afni.nimh.nih.gov/pub/dist/doc/program_help/make_random_timing.py.html)
 * [`RSFgen`](https://afni.nimh.nih.gov/pub/dist/doc/program_help/RSFgen.html) and 
 * [`3dDeconvolve -nodata`](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html)

### Optseq
   ```
   optseq2 --ev A 1.0 3  --ev AB 2.0 3 --ev AC 2.0 3 --tr 1.0 --ntp 50
   fmri_timing -e "50/9x@1 A=1.0,{B=1.0,C=1.0,$}"  # --sym "A -.5*B -.5*C"
   ```

   with optseq
     * there is no way to include B or C in a contrast (?)
     * changing number of trials or trail structure requires editing each ev
     * cannot limit repeats (?) (also a TODO for fmri_timing)
     * multiple forks (`{A,B*.25},{C,D*.33},E`) are more difficult to enumerate/update

# Install
```
## Install
pip install -e git+https://github.com/LabNeuroCogDevel/fmri_timing
# or
#  git clone https://github.com/LabNeuroCogDevel/fmri_timing ~/src/fmri_timing && cd $_ &&  pip install -e .
```
## Use
```
fmri_timing -e "20/3x@1 A,{B,C,$}" --sym "A -.5*B -.5*C"

fmri_timing -e "600/120x@1.3 {choice_good=.7,choice_nogood=.7},{.77*walk=1,$},feedback" --sym "choice_good -choice_nogood" ".5*choice_good +.5*choice_nogood - feedback"
```

### Grammar
Ideally, we'd be able to specify the proportion of trials with each event and ensure some max event repeat over trials

```
200/10x@1.3 A=1.5,{B=1.5,0.15*C=1.5,$}<3>,C=1.5;=1.5
  ###/#x@##  total duraiton / number repeats @ tr

  =#.#       duration
  {,}        permutation, tree siblings
  $          catch end
  *#.#       proportion of permutaitons
  <#>        max repeats                                (TODO)
  ,          add next in sequence
  ;=#.#      variable iti w/stepsize; end of root node  (TODO)
```

additional thoughts on duration (TODO)
```
  =#.#...#.#  = variable duration
  =#...#(exp) = var exponential
```
