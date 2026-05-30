import numpy as np
import collections
import torch
from src.environment import PhysicsSandbox
from src.models_separate_dyn import NonParametricJEPASpatialSeparateDyn
from src.motor import CLTSMotorController

seed = 101
model = NonParametricJEPASpatialSeparateDyn(
    d_max=8, h=3, k=4, cooldown=300, stabilization_period=100,
    pos_encoding='none', primary_objective='sfa',
    sfa_weight=5.0, gdasr_log_only=True,
    dyn_readout='mean', sub_features=1, dyn_source='spatial',
    mask_dyn_sim=True, coord_vicreg=True,
)
model.d_t = 3
model.load_state_dict(torch.load('archive/iter_029/results/checkpoints/b_sfavicreg,_sfa_5.0_seed101.pt', map_location='cpu'))
model.eval()

env = PhysicsSandbox(N=3, seed=seed)
controller = CLTSMotorController(Kp=2.0, Kd=0.5, Kv=0.5)
controller.reset()

history = collections.deque(maxlen=4)
obs = env.render()
history.append(obs)
for _ in range(3):
    obs, info = env.step({'acc': 0.0, 'push': False})
    history.append(obs)

prev_vel = info['velocities'].copy()
collisions = []
for step in range(500):
    x_hist = np.stack(list(history)[:3], axis=0)
    x_target = history[3]
    x_hist_t = torch.from_numpy(x_hist).float().unsqueeze(0)
    x_target_t = torch.from_numpy(x_target).float().unsqueeze(0)
    with torch.no_grad():
        loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(
            x_hist_t, x_target_t, d_t_predict=3)
        z_coord, z_dyn = model.encoder(x_target_t)
    centroids = z_coord[:, :3]
    
    if step < 200:
        controller.token_locus = 0
        controller.attention_cooldown = controller.attention_cooldown_max
    
    action, locus, surprises = controller.get_action(
        model, history[3], info, z_pred_coord, z_target_coord,
        z_pred_dyn, z_target_dyn, 3, centroids)
    
    if step == 100:
        env.masses[0] *= 3.0
        env.pointer_pos = env.positions[0] + (5.0 if env.positions[0] < 64.0 else -5.0)
        action['push'] = True
    
    obs, info = env.step(action)
    history.append(obs)
    
    # Collision detection
    vel_change = np.abs(info['velocities'] - prev_vel)
    changed = np.where(vel_change > 1.0)[0]
    for i in changed:
        pos_i = info['positions'][i]
        r_i = info['radii'][i]
        is_boundary = (pos_i - r_i < 2.0) or (pos_i + r_i > 126.0)
        if not is_boundary:
            collisions.append(step)
            # print(f'  step={step} obj={i} dv={vel_change[i]:.2f} pos={pos_i:.1f} obj_pos={info["positions"]}')
    prev_vel = info['velocities'].copy()

print(f'Total object-object collisions (non-boundary, |dv|>1.0) in 500 steps: {len(collisions)}')
if collisions:
    print(f'Collision steps (first 20): {collisions[:20]}')

# Now test tracking error for just a few steps
print('--- Tracking debug (steps 200-210) ---')
for step in range(195, 210):
    x_hist = np.stack(list(history)[:3], axis=0)
    x_target = history[3]
    x_hist_t = torch.from_numpy(x_hist).float().unsqueeze(0)
    x_target_t = torch.from_numpy(x_target).float().unsqueeze(0)
    with torch.no_grad():
        loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(
            x_hist_t, x_target_t, d_t_predict=3)
        z_coord, z_dyn = model.encoder(x_target_t)
    centroids = z_coord[:, :3]
    
    action, locus, surprises = controller.get_action(
        model, history[3], info, z_pred_coord, z_target_coord,
        z_pred_dyn, z_target_dyn, 3, centroids)
    
    obs, info = env.step(action)
    history.append(obs)
    
    target_pos = centroids[0, locus].item()
    pointer_pos = info['pointer_pos']
    err = abs(pointer_pos - target_pos)
    centroid_vals = centroids[0].cpu().numpy()
    print(f'step={step} locus={locus} centroid_target={target_pos:.1f} pointer={pointer_pos:.1f} err={err:.1f}')
    print(f'  centroids={centroid_vals} obj_pos={info["positions"]}')
