import numpy as np
import gym
from gym import spaces

class DynamicSoaringEnv(gym.Env):
    def __init__(self):
        super(DynamicSoaringEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Bird parameters
        self.mass = 8.5  # kg
        self.wing_area = 0.65  # m^2
        self.aspect_ratio = 15.0
        self.cd0 = 0.033  # Zero-lift drag coefficient
        self.k = 1 / (np.pi * self.aspect_ratio * 0.9)  # Induced drag factor

        # Environment parameters
        self.g = 9.81  # m/s^2
        self.rho = 1.225  # kg/m^3 (air density at sea level)
        self.dt = 0.1  # time step in seconds

        # Initialize state
        self.state = None
        self.wind_layers = self._generate_wind_layers()
        
    def _generate_wind_layers(self):
        # Generate more complex wind layers
        layers = []
        for i in range(5):
            speed = 5 + i * 2  # Wind speed increases with altitude
            direction = np.random.uniform(0, 2*np.pi)  # Random wind direction
            height = i * 200  # Layer height
            layers.append((speed, direction, height))
        return layers
    
    def _get_wind_velocity(self, x, y, z):
        # Interpolate wind velocity based on altitude
        for i, (speed, direction, height) in enumerate(self.wind_layers):
            if z < height or i == len(self.wind_layers) - 1:
                if i == 0:
                    factor = z / height
                else:
                    prev_height = self.wind_layers[i-1][2]
                    factor = (z - prev_height) / (height - prev_height)
                
                prev_speed, prev_direction = self.wind_layers[i-1][:2] if i > 0 else (0, 0)
                
                interp_speed = prev_speed + factor * (speed - prev_speed)
                interp_direction = prev_direction + factor * (direction - prev_direction)
                
                wx = interp_speed * np.cos(interp_direction)
                wy = interp_speed * np.sin(interp_direction)
                return np.array([wx, wy, 0])
        
        return np.array([0, 0, 0])  # Fallback if no suitable layer found

    def reset(self):
        # Initialize state: [x, y, z, vx, vy, vz]
        self.state = np.array([0, 0, 100, 10, 0, 0], dtype=np.float32)
        return self.state
    
    def step(self, action):
        assert self.action_space.contains(action), f"{action} is an invalid action"
        
        # Unpack state
        x, y, z, vx, vy, vz = self.state
        
        # Apply action (change in angle of attack and bank angle)
        d_alpha, d_beta = action
        alpha = np.clip(d_alpha, -np.pi/6, np.pi/6)  # Angle of attack
        beta = np.clip(d_beta, -np.pi/4, np.pi/4)  # Bank angle
        
        # Calculate airspeed and lift
        v_bird = np.array([vx, vy, vz])
        v_wind = self._get_wind_velocity(x, y, z)
        v_air = v_bird - v_wind
        airspeed = np.linalg.norm(v_air)
        
        cl = 2 * np.pi * alpha  # Simplified lift coefficient
        lift = 0.5 * self.rho * airspeed**2 * self.wing_area * cl
        
        # Calculate drag
        cd = self.cd0 + self.k * cl**2
        drag = 0.5 * self.rho * airspeed**2 * self.wing_area * cd
        
        # Calculate forces
        F_lift = lift * (np.cross(v_air, [np.sin(beta), np.cos(beta), 0]) / airspeed)
        F_drag = -drag * (v_air / airspeed)
        F_gravity = np.array([0, 0, -self.mass * self.g])
        
        F_total = F_lift + F_drag + F_gravity
        
        # Update velocity
        acceleration = F_total / self.mass
        v_bird += acceleration * self.dt
        
        # Update position
        x += vx * self.dt
        y += vy * self.dt
        z += vz * self.dt
        
        # Ensure bird doesn't go underground
        z = max(z, 0)
        
        # Update state
        self.state = np.array([x, y, z, v_bird[0], v_bird[1], v_bird[2]], dtype=np.float32)
        
        # Calculate reward
        energy_gained = self.mass * self.g * (z - self.state[2])  # Potential energy gained
        energy_spent = np.linalg.norm(F_drag) * airspeed * self.dt  # Work done against drag
        reward = energy_gained - energy_spent
        
        # Check if episode is done
        done = z <= 0 or z > 3000 or airspeed < 5  # End if bird touches ground, goes too high, or stalls
        
        return self.state, reward, done, {}
    
    def render(self):
        x, y, z, vx, vy, vz = self.state
        print(f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")
        print(f"Velocity: ({vx:.2f}, {vy:.2f}, {vz:.2f})")
        print(f"Airspeed: {np.linalg.norm([vx, vy, vz]):.2f}")
        wind = self._get_wind_velocity(x, y, z)
        print(f"Wind: ({wind[0]:.2f}, {wind[1]:.2f}, {wind[2]:.2f})")

    def get_state_description(self):
        x, y, z, vx, vy, vz = self.state
        airspeed = np.linalg.norm([vx, vy, vz])
        wind = self._get_wind_velocity(x, y, z)
        return {
            "position": (x, y, z),
            "velocity": (vx, vy, vz),
            "airspeed": airspeed,
            "wind": tuple(wind),
            "altitude": z
        }

    def get_wind_layers(self):
        return [(speed, direction, height) for speed, direction, height in self.wind_layers]