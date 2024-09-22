from environment import DynamicSoaringEnv
from agent import DQNAgent
from visualize import DynamicSoaringVisualizer
import numpy as np

# Initialize environment, agent, and visualizer
env = DynamicSoaringEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = DQNAgent(state_size, action_size)
visualizer = DynamicSoaringVisualizer()

# Training parameters
n_episodes = 1000
batch_size = 32

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    for time in range(500):  # max 500 steps per episode
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode: {e}/{n_episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            break
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Save model every 100 episodes
    if e % 100 == 0:
        agent.save(f"dynamic_soaring_model_{e}.h5")

print("Training finished.")

# Test the trained agent
state = env.reset()
state = np.reshape(state, [1, state_size])
total_reward = 0
done = False

while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])
    state = next_state
    total_reward += reward
    env.render()
    visualizer.update(state[0])

print(f"Test episode finished. Total reward: {total_reward:.2f}")

# Show the final trajectory
visualizer.render()
visualizer.show()