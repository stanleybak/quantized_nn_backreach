# Verification of Closed-Loop ACAS Xu Neural Network Control System using State Quantization and Backreachability

This is the code from NFM 2022 submission "Closed-Loop ACAS Xu NNCS is Unsafe:
Quantized State Backreachability for Verification" by Stanley Bak and Hoang-Dung Tran.

# Setup
The code was developed on a Linux machine using Ubuntu 20.04.

Installation commands:

```
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
pip3 install --upgrade pip

git clone https://github.com/stanleybak/quantized_nn_backreach.git

cd quantized_nn_backreach
pip3 install -r requirements.txt
```

At this point, you likely want to test if things look like they're working. run
`python3 backreach.py` and then use ctrl + c to kill the process if it looks like it's running.


If you want to generate the mp4 videos (optional), you'll also need ffmpeg. Do `sudo apt install ffmpeg` and then make sure you can run the `ffmpeg` command in a terminal. I used version `4.2.4-1ubuntu0.1`.


# Simulations

The code to run simulations is in the `simulation` folder. You should run `parallel_acasxu_dubins.py` with python3.

The random states are generated in `acasxu_dubins.py` in the `make_random_input` function, which takes in a `seed`, which is the random seed to use. If you want to change how random state generation is done, this is the function to edit.

Edit the variables in the `main()` function to change the number of simulations (edit `batch_size` and `num_sims`). Right now it's set to run a single batch of 1.5 million sims, which took about 3 minutes on my laptop.

Set the value of `max_tau` to 160 for out-of-plane simulations. Right now it should be set to 0, which runs in-plane simulations.

After you run the code, it prints the unsafe seeds to the terminal. The expected output with 1.5 million in-plane simulations include the line `Collision seeds (14): [57305, 174596, 191434, 207659, 341255, 468582, 872977, 891092, 1084018, 1182535, 1339442, 1379646, 1380106, 1381252]`.

The statistics from the paper about the mean and std deviation can be obtained by using `analyze_seeds.py`. That file right now contains the seeds from the full 150 million simulation run, and those can instead be replaced with other seeds, if desired (for example, if you change how the random state generation is performed). Notice that in the `get_seeds` function, the first 14 seeds match exactly the output we got with 1.5 million simulations: `in_plane_seeds = [57305, 174596, 191434, 207659, ...]`

# Verification

To do a verification run including the falsification algorithm, you need to first assign the settings you want. The main file to edit is `settings.py` and the variables you want to edit are:

```
pos_q = 250
vel_q = 0
theta1_q_deg = 1.5 # should divide 1.5 degrees evenly

range_vel_ownship = (200, 200) # full range: (100, 1200)                                                                         
range_vel_intruder = (185, 185) # full range: (0, 1200)
```

The first 3 are the quantum values to use. `vel_q` can be 0, which means the velocity is a fixed value. The `range_vel` are the velocity ranges to consider. In the current setup, these are fixed values, to do the comparison with the Horizontal CAS system like in evaluation in the paper.

After assigning the settings, you can then run `backreach.py` using python3. The proof first does the `tau_dot = 0` case, and if that succeeds, it does the `tau_dot = -1` case, so it looks like it's running twice. Running with the default settings you get the following output after after two minutes on my laptop:

```
Done! No counterexamples.
original system had no counterexamples
completed proof for tau_dot=-1 case
final proven safe?: True
```

You can then modify the settings to find some counterexamples, let's change the settings to the full range as described in the paper:

```
pos_q = 500
vel_q = 100
theta1_q_deg = 1.5 # should divide 1.5 degrees evenly

range_vel_ownship = (100, 1200) # full range: (100, 1200)                                                                         
range_vel_intruder = (0, 1200) # full range: (0, 1200)
```

Running `backreach.py` now will produce counterexamples. Upon reaching `max_counterexamples` (default 128), the proof will stop trying and refinement will begin. This should quickly generate a counterexample of the original system. The exact counterexample may vary, as it depends on how the partitions were analyzed with multithreading, which can be nondeterministic. The output should look something like this, after about a minute:

```
Non-quantized replay is unsafe (rho=309.29558847833624)! Real counterexample.

alpha_prev_list = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0]
qtheta1 = 73
qv_own = 1
qv_int = 11
tau_init = 0
# chebeshev center radius: 0.0027876091839843017
end = np.array([-2.78760918e-03, -2.78760918e-03,  1.40264128e+02,  6.51710649e+00,
        0.00000000e+00,  1.11319526e+03])
start = np.array([-2.90279886e+03, -6.47159328e+03, -4.95421358e+01,  1.31385216e+02,
       -6.45653252e+04,  1.11319526e+03])

Found reach counterexample at state: (4, -1, -1, 1, 1, 11)
counterexamples were not proven safe
not proven safe for for tau_dot = 0
```

The part in the middle can be copied and pasted to produce a figure, replay, or video, as described in the next section of this README.

# Replaying Traces and Generating Figures, Videos, and Latex Tables
The main code to produce visualizations of unsafe replays is `replay.py`. If you look in this file, several unsafe initial states are defined in different functions like `taudot_faster` or `leftturn_counterexample`, which correspond to some of the situations plotted in the paper.

You can copy one of these functions and give it a new name, like `get_mycounterexample`, and fill in the data output from when we ran `backreach.py`. Put this function after `main()`, or else line numbers in the rest of this document may be incorrect.

```
def get_mycounterexample():
    """counterexample with faster taudo"""

    alpha_prev_list = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0]
    qtheta1 = 73
    qv_own = 1
    qv_int = 11
    tau_init = 0
    # chebeshev center radius: 0.0027876091839843017
    end = np.array([-2.78760918e-03, -2.78760918e-03,  1.40264128e+02,  6.51710649e+00,
            0.00000000e+00,  1.11319526e+03])
    start = np.array([-2.90279886e+03, -6.47159328e+03, -4.95421358e+01,  1.31385216e+02,
           -6.45653252e+04,  1.11319526e+03])


    label = "Plot Title: My counterexample"
    name = "filenametag_mycounterexample"
    rewind_seconds = 0 # optional: rewind state by a few second
    ownship_below = True # plot ownship label below airplane (False = above)

    return alpha_prev_list, qtheta1, qv_own, qv_int, end, start, rewind_seconds, label, name, ownship_below, tau_init
```

Next, in `main()` on line 1021 change `case_funcs` to include your new function `case_funcs = [get_mycounterexample]`, and set `paper` to False on the next line: `paper = False`.

Running `replay.py` now with python3 should produce a live visualization of the situation. The terminal also prints some information:

```
init tau: 0
init rho: 62001.19897399513
init own vel: 140.4154485909307
init int vel: 1113.19526
Simulation completed.

Got first few commands: [0, 0, 2, 2, 2, 2, 2, 2, 2, 2]
Expected first few commands: [0, 2, 2, 2, 2, 2, 2, 2, 2, 2]
WARNING: skipped sanity checks on replay
Plotting sim with min_dist: 309.3 and final dx: -1205.5, dy: 222.5, dist: 1225.88
```

Notice that `min_dist` was 309.3 and `init_rho` was 62001, so this is indeed a real unsafe path.

To produce plots like the ones in the paper, set `paper = True`. This should produce images `filenametag_mycounterexample.png` and `square_filenametag_mycounterexample.png`, as well as a latex table in the terminal with the trace, like the one in the appendices:

```
% Auto-generated
% The unrounded initial state is $\rho$ = 62001.19897399513 ft, $\theta$ = 1.105638365566048 rad, $\psi=-1.9313853026445638$ rad, $v_{own}$ = 140.4154485909307 ft/sec, and $v_{int}$ = 1113.19526 ft/sec.
\toprule
Step & $\alpha_\text{prev}$ & Cmd & $\rho$ (ft) & $\theta$ (deg) & $\psi$ (deg) \\
\midrule
1 & \textsc{coc} & \textsc{coc} & 62001.2 & 63.35 & -110.66 \\
2 & \textsc{coc} & \textsc{coc} & 60831.1 & 63.36 & -110.66 \\
3 & \textsc{coc} & \textsc{wr} & 59661.0 & 63.37 & -110.66 \\
...
```

If you also want to save an mp4 video (and you have ffmpeg setup as described earlier), you can uncomment line 1179: `#plot(s, name=name, save_mp4=True)`. and run `replay.py` again to produce `filenametag_mycounterexample.mp4`.

# Where's Algorithm 1?

Algorithm 1 in the paper closely follows the `get_predecessors` method in `backreach.py`. There are some slight differences, usually code optimizations that were not discussed in the paper for clarity. The checking for initial states is done in `backreach_single_unwrapped` on line 481.

# Notes for long runs over SSH on AWS
If you use an AWS server and connect to it over SSH, calling a process for a long time can sometimes lead to issues. In particular, when the SSH process exits (or disconnects), any processes it started will also be killed. To prevent this and allow you to disconnect and reconnect, you can use something like `tmux`, or the following commands:

For an persistent process that will keep running when ssh closes, and then shuts down the connection use the following commands:
`stdbuf -oL python3 backreach.py >& stdout.txt &`

Alternatively, if you also want to shut down the machine when things are done, try this one instead (untested, not sure if it actually stops it:
`((stdbuf -oL python3 backreach.py >& stdout.txt); sudo halt) &`

Explanation: `stdbuf -oL` disables unnecessary buffering when redirecting stdout, stdout and stderr will be sent to the file `stdout.txt` and the `&` starts the process in the background

Next disconnect all processes whose parent is the ssh shell, so if ssh closes the measurement process keeps running:
`disown -a`

Finally, you can print progress live as it gets sent to stdout.txt:
`tail -f stdout.txt`


