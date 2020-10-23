from DDPG import *
import warnings

warnings.simplefilter("ignore")
# Learning rate for actor-critic models
total_episodes   = 100
ep_reward_list   = [] # To store reward history of each episode
avg_reward_list  = [] # To store average reward history of last few episodes
best_reward      = -10e8

myagent = Agent(environment='Pendulum-v0',loadsavedfile=True,disablenoise=True)

for eps in range(total_episodes):

    state           = myagent.sim.reset()
    episodic_reward = 0

    while True:
        myagent.sim.render()
        action                        = myagent.policy(state.reshape(1,myagent.numberOfStates))
        nextstate, reward, done, info = myagent.sim.step([action])
        myagent.observation           = (state,action,reward,nextstate)
        #myagent.record_buffer()
        episodic_reward               = episodic_reward + reward
       # myagent.learn()
        # End this episode when `done` is True
        if done:
            print('noise variance is %.8f' % (myagent.noisevariance))
            print('noise is %.8f' % (myagent.noise))
            break

        state = nextstate

    ep_reward_list.append(episodic_reward)
    # Mean of last 10 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(eps, avg_reward))
    avg_reward_list.append(avg_reward)
    if avg_reward_list[-1]>best_reward:
        best_reward = avg_reward_list[-1]
        print('saving models')
        #myagent.save()
    print('-----------------------------------------------------------------')
# Plotting graph Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()