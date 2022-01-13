'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used for falsification, where the opponent is allowed to maneuver over time

This version uses multiprocessing pool for more simulations
'''

import time
import multiprocessing
import argparse

import numpy as np

from acasxu_dubins import State, make_random_input, plot, state7_to_state5, run_network, network_index

def sim_single(seed, intruder_can_turn, max_tau):
    """run single simulation and return min_dist"""

    rv = np.inf

    if seed % 50000 == 0:
        print(f"{(seed//50000) % 10}", end='', flush=True)
    elif seed % 5000 == 0:
        print(".", end='', flush=True)

    tau_dot = -1 if max_tau > 0 else 0

    init_vec, cmd_list, init_velo, tau_init = make_random_input(seed, max_tau=max_tau, intruder_can_turn=intruder_can_turn)

    # run the simulation
    s = State(init_vec, tau_init, tau_dot, init_velo[0], init_velo[1], save_states=False)
    s.simulate(cmd_list)

    rv = s.min_dist

    return rv

def main():
    'main entry point'

    # dt = 0.05
    #Did 1000000 parallel sims in 794.3 secs (0.794ms per sim)
    #Rejected 58480 sims. 5.848%
    #Collision in 3 sims. 0.0003%
    #Seed 350121 has min_dist 325.4ft

    # parse arguments
    parser = argparse.ArgumentParser(description='Run ACASXU Dublins model simulator.')
    parser.add_argument("--save-mp4", action='store_true', default=False, help="Save plotted mp4 files to disk.")
    args = parser.parse_args()

    save_mp4 = args.save_mp4
    intruder_can_turn = False
    max_tau = 0 # 160 or 0
    tau_dot = -1 if max_tau > 0 else 0

    # home laptop (dt=0.05): 10000000 parallel sims take 5714.9 secs (0.571ms per sim)
    batch_size = 1500000
    num_sims = batch_size * 1 # use "* 100" instead for 150 million

    remaining_sims = num_sims
    completed_sims = 0

    collision_dist = 500
    min_dist = np.inf
    min_seed = -1
    num_collisions = 0
    collision_seeds = []
    num_with_dist = 0
    start = time.perf_counter()

    print(f"Running {num_sims} parallel simulations in batches of {batch_size}...")

    with multiprocessing.Pool() as pool:
        while remaining_sims > 0:
            cur_batch = min(batch_size, remaining_sims)
            params = []

            for i in range(cur_batch):
                p = (completed_sims + i, intruder_can_turn, max_tau)
                params.append(p)
            
            results = pool.starmap(sim_single, params, chunksize=200)
            print()
            
            for index, dist in enumerate(results):
                seed = completed_sims + index

                if dist != np.inf:
                    num_with_dist += 1

                if dist < min_dist:
                    min_dist = dist
                    min_seed = seed

                if dist < collision_dist:
                    num_collisions += 1
                    collision_seeds.append(seed)

                    init_vec, cmd_list, init_velo, tau_init = make_random_input(seed, max_tau=max_tau, intruder_can_turn=intruder_can_turn)
                    s = State(init_vec, tau_init, tau_dot, init_velo[0], init_velo[1])

                    print(f"{num_collisions}. Collision (dist={round(dist, 2)}) with seed {seed}: {s}")

            # print progress
            completed_sims += cur_batch
            remaining_sims -= cur_batch

            frac = completed_sims / num_sims
            elapsed = time.perf_counter() - start
            total_estimate = (elapsed / frac)
            total_min = round(total_estimate / 60, 1)
            eta_estimate = total_estimate - elapsed
            eta_min = round(eta_estimate / 60, 1)
            
            print(f"Collision seeds: {collision_seeds}")
            num_collisions = len(collision_seeds)
            print(f"Collision in {num_collisions} sims. {round(100 * num_collisions / completed_sims, 8)}%")

            percent = 100 * num_with_dist / completed_sims
            print(f"num with dist: {num_with_dist} / {completed_sims}: {percent:.4f}%")

            percent = 100 * completed_sims / num_sims
            print(f"{completed_sims}/{num_sims} ({percent:.4f}%) total estimate: {total_min}min, ETA: {eta_min} min, " + \
                f"col: {num_collisions} ({round(100 * num_collisions / completed_sims, 6)}%)")

    diff = time.perf_counter() - start
    ms_per_sim = round(1000 * diff / num_sims, 3)
    print(f"\nDid {num_sims} parallel sims in {round(diff, 1)} secs ({ms_per_sim}ms per sim)")
    print(f"Collision seeds ({len(collision_seeds)}): {collision_seeds}")
    
    print(f"Collision in {num_collisions} sims. {round(100 * num_collisions / num_sims, 6)}%")

    d = round(min_dist, 1)
    print(f"\nSeed {min_seed} has min_dist {d} ft")

    # optional: do plot
    if False:
        init_vec, cmd_list, init_velo, tau_init = make_random_input(min_seed, max_tau=max_tau, intruder_can_turn=intruder_can_turn)
        s = State(init_vec, tau_init, tau_dot, init_velo[0], init_velo[1], save_states=True)
        s.simulate(cmd_list)
        assert abs(s.min_dist - min_dist) < 1e-6, f"got min dist: {s.min_dist}, expected: {min_dist}"

        plot(s, save_mp4)

if __name__ == "__main__":
    main()
