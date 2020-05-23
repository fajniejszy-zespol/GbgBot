

# TODO:  add game as return in the worker. 

def worker(remote, parent_remote, env, worker_id):
    parent_remote.close()

    steps = 0
    tds = 0

    while True:
        command, data = remote.recv()
        if command == 'step':
            steps += 1
            action, dif = data[0], data[1]
            obs, reward, done, info = env.step(action)
            tds_scored = info['touchdowns'] - tds
            tds = info['touchdowns']
            reward_shaped = reward_function(env, info, shaped=True)
            ball_carrier = env.game.get_ball_carrier()
            # PPCG
            if dif < 1.0:
                if ball_carrier and ball_carrier.team == env.game.state.home_team:
                    extra_endzone_squares = int((1.0 - dif) * 25.0)
                    distance_to_endzone = ball_carrier.position.x - 1
                    if distance_to_endzone <= extra_endzone_squares:
                        #reward_shaped += rewards_own[OutcomeType.TOUCHDOWN]
                        env.game.state.stack.push(Touchdown(env.game, ball_carrier))
            if done or steps >= reset_steps:
                # If we  get stuck or something - reset the environment
                if steps >= reset_steps:
                    print("Max. number of steps exceeded! Consider increasing the number.")
                done = True
                obs = env.reset()
                steps = 0
                tds = 0
            remote.send((obs, reward, reward_shaped, tds_scored, done, info))
        elif command == 'reset':
            dif = data
            steps = 0
            tds = 0
            obs = env.reset()
            # set_difficulty(env, dif)
            remote.send(obs)
        elif command == 'render':
            env.render()
        elif command == 'close':
            break

            
class VecEnv():
    def __init__(self, envs):
        """
        envs: list of FFAI environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
                   for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions, difficulty=1.0):
        cumul_rewards = None
        cumul_shaped_rewards = None
        cumul_tds_scored = None
        cumul_dones = None
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', [action, difficulty]))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, rews_shaped, tds, dones, infos = zip(*results)
        if cumul_rewards is None:
            cumul_rewards = np.stack(rews)
            cumul_shaped_rewards = np.stack(rews_shaped)
            cumul_tds_scored = np.stack(tds)
        else:
            cumul_rewards += np.stack(rews)
            cumul_shaped_rewards += np.stack(rews_shaped)
            cumul_tds_scored += np.stack(tds)
        if cumul_dones is None:
            cumul_dones = np.stack(dones)
        else:
            cumul_dones |= np.stack(dones)
        return np.stack(obs), cumul_rewards, cumul_shaped_rewards, cumul_tds_scored, cumul_dones, infos

    def reset(self, difficulty=1.0):
        for remote in self.remotes:
            remote.send(('reset', difficulty))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))
        return

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)

        
        
        
def main():
    es = [make_env(i) for i in range(num_processes)]
    envs = VecEnv([es[i] for i in range(num_processes)])
    
    # Clear log file
    try:
        os.remove(log_filename)
    except OSError:
        pass

    gbg_bot = GbgBotWrapper(es[0], 8 )   
    
    # PPCG
    difficulty = 0.0
    dif_delta = 0.01

    # Reset environments
    obs = envs.reset(difficulty)
    spatial_obs, non_spatial_obs = update_obs(obs)

    # Add obs to memory
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)

    # Variables for storing stats
    all_updates = 0
    all_episodes = 0
    all_steps = 0
    episodes = 0
    proc_rewards = np.zeros(num_processes)
    proc_tds = np.zeros(num_processes)
    episode_rewards = []
    episode_tds = []
    wins = []
    value_losses = []
    policy_losses = []
    log_updates = []
    log_episode = []
    log_steps = []
    log_win_rate = []
    log_td_rate = []
    log_mean_reward = []
    log_difficulty = []

    renderer = ffai.Renderer()

    while all_steps < num_steps:

        for step in range(steps_per_update):
            
            # Calculate action for all processes 
            
            #action_masks = compute_action_masks(obs)
            #action_masks = torch.tensor(action_masks, dtype=torch.bool)

            #values, actions = ac_agent.act(
            #    Variable(memory.spatial_obs[step]),
            #    Variable(memory.non_spatial_obs[step]),
            #    Variable(action_masks))

            # action_objects = []

            # for action in actions:
                # action_type, x, y = compute_action(action.numpy()[0])
                # action_object = {
                    # 'action-type': action_type,
                    # 'x': x,
                    # 'y': y
                # }
                # action_objects.append(action_object)

            obs, env_reward, shaped_reward, tds_scored, done, info = envs.step(action_objects, difficulty=difficulty)
            #envs.render()
            '''
            for j in range(len(obs)):
                ob = obs[j]
                renderer.render(ob, j)
            '''
            # CALCULATE REWARD 
            reward = torch.from_numpy(np.expand_dims(np.stack(env_reward), 1)).float()
            shaped_reward = torch.from_numpy(np.expand_dims(np.stack(shaped_reward), 1)).float()
            r = reward.numpy()
            sr = shaped_reward.numpy()
            for i in range(num_processes):
                proc_rewards[i] += sr[i]
                proc_tds[i] += tds_scored[i]
            
            
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            dones = masks.squeeze()
            episodes += num_processes - int(dones.sum().item())
            for i in range(num_processes):
                if done[i]:
                    if r[i] > 0:
                        wins.append(1)
                        difficulty += dif_delta
                    elif r[i] < 0:
                        wins.append(0)
                        difficulty -= dif_delta
                    else:
                        wins.append(0.5)
                        difficulty -= dif_delta
                    if ppcg:
                        difficulty = min(1.0, max(0, difficulty))
                    else:
                        difficulty = 1
                    episode_rewards.append(proc_rewards[i])
                    episode_tds.append(proc_tds[i])
                    proc_rewards[i] = 0
                    proc_tds[i] = 0

            # Update the observations returned by the environment
            spatial_obs, non_spatial_obs = update_obs(obs)

            # insert the step taken into memory
            memory.insert(step, spatial_obs, non_spatial_obs,
                          actions.data, values.data, shaped_reward, masks, action_masks)

        next_value = ac_agent(Variable(memory.spatial_obs[-1], requires_grad=False), Variable(memory.non_spatial_obs[-1], requires_grad=False))[0].data

        # Compute returns
        memory.compute_returns(next_value, gamma)

        spatial = Variable(memory.spatial_obs[:-1])
        spatial = spatial.view(-1, *spatial_obs_space)
        non_spatial = Variable(memory.non_spatial_obs[:-1])
        non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        actions_mask = Variable(memory.action_masks[:-1])

        # Evaluate the actions taken
        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

        values = values.view(steps_per_update, num_processes, 1)
        action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

        advantages = Variable(memory.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()
        #value_losses.append(value_loss)

        # Compute loss
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        #policy_losses.append(action_loss)

        optimizer.zero_grad()

        total_loss = (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

        optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])

        # Updates
        all_updates += 1
        # Episodes
        all_episodes += episodes
        episodes = 0
        # Steps
        all_steps += num_processes * steps_per_update

        # Logging
        if all_updates % log_interval == 0 and len(episode_rewards) >= num_processes:
            td_rate = np.mean(episode_tds)
            episode_tds.clear()
            mean_reward = np.mean(episode_rewards)
            episode_rewards.clear()
            win_rate = np.mean(wins)
            wins.clear()
            #mean_value_loss = np.mean(value_losses)
            #mean_policy_loss = np.mean(policy_losses)    
            
            log_updates.append(all_updates)
            log_episode.append(all_episodes)
            log_steps.append(all_steps)
            log_win_rate.append(win_rate)
            log_td_rate.append(td_rate)
            log_mean_reward.append(mean_reward)
            log_difficulty.append(difficulty)

            log = "Updates: {}, Episodes: {}, Timesteps: {}, Win rate: {:.2f}, TD rate: {:.2f}, Mean reward: {:.3f}, Difficulty: {:.2f}" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, mean_reward, difficulty)

            log_to_file = "{}, {}, {}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, mean_reward, difficulty)

            print(log)

            # Save to files
            with open(log_filename, "a") as myfile:
                myfile.write(log_to_file)

            # Saving the agent
            torch.save(ac_agent, "models/" + model_name)

            episodes = 0
            value_losses.clear()
            policy_losses.clear()
            
            # plot
            if ppcg:
                fig, axs = plt.subplots(1, 4, figsize=(16, 5))
            else:
                fig, axs = plt.subplots(1, 3, figsize=(12, 5))
            axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[0].plot(log_steps, log_mean_reward)
            axs[0].set_title('Reward')
            #axs[0].set_ylim(bottom=0.0)
            axs[0].set_xlim(left=0)
            axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[1].plot(log_steps, log_td_rate)
            axs[1].set_title('TD/Episode')
            axs[1].set_ylim(bottom=0.0)
            axs[1].set_xlim(left=0)
            axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[2].plot(log_steps, log_win_rate)
            axs[2].set_title('Win rate')            
            axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
            axs[2].set_xlim(left=0)
            if ppcg:
                axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[3].plot(log_steps, log_difficulty)
                axs[3].set_title('Difficulty')
                axs[3].set_yticks(np.arange(0, 1.001, step=0.1))
                axs[3].set_xlim(left=0)
            fig.tight_layout()
            fig.savefig("plots/"+model_name+".png")
            plt.close('all')

            # Save model
            torch.save(ac_agent, "models/" + model_name)

    torch.save(ac_agent, "models/" + model_name)
    envs.close()
