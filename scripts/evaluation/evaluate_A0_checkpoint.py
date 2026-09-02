import argparse
from pathlib import Path
import numpy as np
import gym
import deprl
import sconegym
import sconegym_crutch_v3

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Path to step_XXXXXX.pt")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--store", action="store_true")
args = parser.parse_args()

ENV_ID = "sconewalk_rajagopal_crutch_v3_A0_walk_003-v1"
env = gym.make(ENV_ID)
policy = deprl.load(args.checkpoint, environment=env)

for ep in range(args.episodes):
    if args.store:
        env.store_next_episode()

    obs = env.reset(seed=1000 + ep)
    start_x = float(env.unwrapped.model.com_pos().x)
    score = 0.0

    for step in range(1000):
        action = policy(obs)
        obs, rew, done, info = env.step(action)
        score += rew
        if done:
            break

    end_x = float(env.unwrapped.model.com_pos().x)
    duration = (step + 1) * 0.01
    mean_v = (end_x - start_x) / max(duration, 1e-9)

    print(
        f"ep={ep:02d} steps={step+1:4d} score={score:+.2f} "
        f"dx={end_x-start_x:+.4f} m mean_COM_v={mean_v:+.4f} m/s"
    )

env.close()
