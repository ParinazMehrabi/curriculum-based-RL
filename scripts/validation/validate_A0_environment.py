import gym
import numpy as np
import sconegym
import sconegym_crutch_v3

ENV_ID = "sconewalk_rajagopal_crutch_v3_A0_walk_003-v1"

print("=" * 88)
print("CURRICULUM V3 — A0 ENVIRONMENT SMOKE TEST")
print("=" * 88)

env = gym.make(ENV_ID)
u = env.unwrapped

print("env id              :", ENV_ID)
print("action shape        :", env.action_space.shape)
print("action low/high     :", env.action_space.low[0], env.action_space.high[0])
print("observation shape   :", env.observation_space.shape)
print("DOFs                :", len(u.model.dofs()))
print("actuators           :", len(u.model.actuators()))
print("muscles             :", len(u.model.muscles()))
print("target velocity     :", u.target_vel_v3)
print("initial vx mean/std :", u.initial_forward_velocity, u.initial_forward_velocity_std)
print("trajectory reward   : OFF")
print("crutch reward       : OFF")
print("effort reward       : OFF")
print()

assert env.action_space.shape == (9,)
assert np.allclose(env.action_space.low, -1.0)
assert np.allclose(env.action_space.high, +1.0)
assert len(u.model.dofs()) == 16
assert len(u.model.actuators()) == 9
assert len(u.model.muscles()) == 0

# Confirm diversity without requiring passive stability.
states = []
for seed in range(5):
    obs = env.reset(seed=seed)
    q = np.asarray(u.model.dof_position_array(), dtype=float)
    dq = np.asarray(u.model.dof_velocity_array(), dtype=float)
    states.append(np.concatenate([q, dq]))

    print(
        f"reset {seed}: "
        f"COM vx={u.model.com_vel().x:+.5f}  "
        f"pelvis vx={dq[u._dof_index['pelvis_tx']]:+.5f}  "
        f"pelvis y={q[u._dof_index['pelvis_ty']]:.5f}  "
        f"armR={q[u._dof_index['arm_flex_r']]:+.4f}  "
        f"elbowR={q[u._dof_index['elbow_flex_r']]:+.4f}"
    )

unique = 0
for i in range(1, len(states)):
    if not np.allclose(states[0], states[i]):
        unique += 1
assert unique >= 3, "Reset diversity is unexpectedly low."

obs = env.reset(seed=123)
action = np.zeros(9, dtype=np.float32)
obs2, reward, done, info = env.step(action)

print()
print("first controlled step:")
print("  reward     :", reward)
print("  done       :", done)
print("  COM vx     :", info["com_vx"])
print("  components :", {k: v for k, v in info.items() if k.startswith("r_")})
print()
print("PASS")
print()
print("If this passes, Stage A0 is structurally ready for the first training run.")
env.close()
