import relocor
from aliases import sde_dict, get_action_param
# from stable_baselines3 import PPO, A2C, TD3, DDPG




def main():

    # Parameters
    sde_name = 'multi_bs'
    action_name = 'ortho'
    AgentClass = relocor.agents.PG
    N_euler = 50
    T = 1.
    EPOCHS = 20
    batch_size = 512
    batch_eval = 512*16
    epoch_eval_freq = 1
    state_idxs_plot = [0]
    action_idxs_plot = [0]


    sde = sde_dict[sde_name]['sde']
    payoff = sde_dict[sde_name]['payoff']
    batch_payoff = sde_dict[sde_name]['batch_payoff']
    dir_name = '{}_0'.format(sde_name)

    action_param, batch_action_param = get_action_param(action_name, sde)

    env = relocor.SDEEnvironment(
        sde = sde,
        T = T,
        N_euler = N_euler,
        test_function = payoff,
        action_param = action_param
    )

    batch_env = relocor.BatchSDEEnvironment(
        sde = sde,
        T = T,
        N_euler = N_euler,
        batch_test_function = batch_payoff,
        batch_action_param = batch_action_param
    )


    experiment = relocor.Experiment(
        env = env, batch_env = batch_env, AgentClass = AgentClass
    )

    print("========== BEGIN EVALUATING BASELINE ==========")
    variance, total_reward, mean = experiment.evaluate(
        nb_episodes=20,
        batch_size=512*16,
        policy_action=experiment.env.action_param.baseline_action)
    print('Mean, Variance with baseline agent: {}, {}'.format(mean, variance))
    print("========== END EVALUATING BASELINE ==========")

    print('\n')


    print("========== BEGIN EVALUATING ANTITHETIC ==========")
    variance, total_reward, mean = experiment.evaluate(
        nb_episodes=20,
        batch_size=512*16,
        policy_action=experiment.env.action_param.antithetic_action)
    print('Mean, Variance with antithetic agent: {}, {}'.format(mean, variance))
    print("========== END EVALUATING ANTITHETIC ==========")

    print('\n')


    print("========== BEGIN TRAINING ==========")
    experiment.train(
        total_timesteps=N_euler*EPOCHS,
        batch_size = batch_size,
        batch_eval = batch_eval,
        epoch_eval_freq = epoch_eval_freq
        )
    print("========== END TRAINING ==========")

    experiment.plot_train_variance()

    print("========== BEGIN EVALUATING TRAINED AGENT ==========")
    variance, total_reward, mean = experiment.evaluate(
        nb_episodes=10,
        batch_size=512*16)
    print('Mean, Variance with trained agent: {}, {}'.format(mean, variance))
    print("========== END EVALUATING TRAINED AGENT ==========")
    

    print('\n')

    experiment.run_trajectory()
    experiment.display_trajectory(state_idxs=state_idxs_plot, action_idxs=action_idxs_plot)

    print("========== SAVING ==========")
    saved = False
    while not saved:
        try:
            experiment.save_experiment(path='./results/{}'.format(dir_name))
            saved = True
        except:
            dir_nb = int(dir_name[dir_name.rindex('_')+1:]) + 1
            dir_name = '{}_{}'.format(sde_name, dir_nb)
    experiment.save_trajectory(path='./results/{}'.format(dir_name))




if __name__ == '__main__':
    main()
