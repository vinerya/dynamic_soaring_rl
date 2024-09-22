# Dynamic Soaring Reinforcement Learning Environment

This project implements a reinforcement learning environment for dynamic soaring, inspired by the flight patterns of albatrosses. The environment simulates a realistic version of the wind dynamics and bird flight mechanics involved in dynamic soaring, using OpenAI Gym and a custom DQN agent. It also includes a manual control mode for users to develop intuition about the environment.

## Project Structure

- `environment.py`: Contains the `DynamicSoaringEnv` class, which implements the OpenAI Gym interface for the dynamic soaring environment.
- `agent.py`: Implements a custom DQN (Deep Q-Network) agent for learning the dynamic soaring task.
- `main.py`: The main script to run the training process and visualize the results.
- `visualize.py`: Provides a 3D visualization of the dynamic soaring trajectory.
- `manual_control.py`: Implements a game-like interface for manual control of the dynamic soaring environment.
- `requirements.txt`: Lists all the required Python packages for this project.

## Environment Details

The dynamic soaring environment has been designed to provide a realistic simulation of albatross flight patterns. Key features include:

1. **Complex Wind Model**: The environment implements a multi-layer wind model where wind speed and direction vary with altitude. This creates a more realistic scenario for the agent to learn dynamic soaring techniques.

2. **Accurate Flight Dynamics**: The simulation includes detailed bird flight mechanics, incorporating:
   - Lift and drag forces based on the bird's airspeed and angle of attack
   - Realistic bird parameters (mass, wing area, aspect ratio)
   - Energy expenditure calculations

3. **Sophisticated Reward Function**: The reward is based on the energy gained from altitude changes minus the energy spent working against drag. This encourages the agent to find efficient soaring patterns.

4. **Realistic Termination Conditions**: The episode ends if the bird touches the ground, flies too high (above 3000m), or stalls (airspeed below 5 m/s).

5. **Detailed State Representation**: The state includes the bird's 3D position and velocity, providing a complete picture of its flight condition.

6. **Flexible Action Space**: The agent controls the bird's angle of attack and bank angle, allowing for complex maneuvers.

This environment provides a challenging and realistic platform for reinforcement learning agents to develop dynamic soaring strategies similar to those observed in albatrosses.

## Requirements

- Python 3.7+
- OpenAI Gym
- NumPy
- TensorFlow 2.x
- Matplotlib
- Pygame

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/dynamic_soaring_rl.git
   cd dynamic_soaring_rl
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the agent and visualize the results, run:

```
python main.py
```

This will start the training process using the custom DQN agent. The agent will learn to perform dynamic soaring in the simulated environment. After training, the script will run a test episode and display a 3D visualization of the learned trajectory.

To use the manual control mode and develop intuition about the environment, run:

```
python manual_control.py
```

In the manual control mode:
- Use the UP and DOWN arrow keys to control the angle of attack
- Use the LEFT and RIGHT arrow keys to control the bank angle
- The display shows the bird's position, current wind layers, and relevant flight information
- Try to maintain altitude and speed by utilizing the varying wind layers

## Customization

You can modify the following files to experiment with different aspects of the simulation:

- `environment.py`: Adjust the wind profile, physics simulation, or reward function.
- `agent.py`: Modify the neural network architecture or hyperparameters of the DQN agent.
- `main.py`: Change the number of episodes, maximum steps per episode, or other training parameters.
- `visualize.py`: Customize the visualization settings or add additional plots.
- `manual_control.py`: Adjust the user interface or add more features to the manual control mode.

## Future Improvements

- Implement even more sophisticated wind models, including turbulence and thermals
- Add real-time visualization during training
- Experiment with different RL algorithms (e.g., PPO, SAC) to compare performance
- Incorporate more detailed bird physiology and energy consumption models
- Enhance the manual control mode with more intuitive controls and visual feedback

## License

This project is open-source and available under the MIT License.