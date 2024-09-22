import pygame
import sys
import numpy as np
from environment import DynamicSoaringEnv

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dynamic Soaring Simulator")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Font
font = pygame.font.Font(None, 36)

def draw_bird(screen, x, y):
    pygame.draw.circle(screen, RED, (int(x), int(y)), 5)

def draw_wind_layers(screen, wind_layers):
    for i, (speed, direction, height) in enumerate(wind_layers):
        y = HEIGHT - (height / 10)
        pygame.draw.line(screen, BLUE, (0, y), (WIDTH, y), 2)
        text = font.render(f"Wind: {speed:.1f} m/s", True, BLUE)
        screen.blit(text, (10, y - 30))

def main():
    try:
        env = DynamicSoaringEnv()
        state = env.reset()
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Get user input
            keys = pygame.key.get_pressed()
            d_alpha = 0
            d_beta = 0
            if keys[pygame.K_UP]:
                d_alpha += 0.1
            if keys[pygame.K_DOWN]:
                d_alpha -= 0.1
            if keys[pygame.K_LEFT]:
                d_beta -= 0.1
            if keys[pygame.K_RIGHT]:
                d_beta += 0.1

            # Clip action values to be within -1 and 1
            d_alpha = np.clip(d_alpha, -1, 1)
            d_beta = np.clip(d_beta, -1, 1)

            # Step the environment
            action = np.array([d_alpha, d_beta], dtype=np.float32)
            print(f"Action: {action}")  # Debug print
            state, reward, done, _ = env.step(action)
            print(f"State: {state}, Reward: {reward}, Done: {done}")  # Debug print

            # Clear the screen
            screen.fill(WHITE)

            # Draw wind layers
            draw_wind_layers(screen, env.get_wind_layers())

            # Draw the bird
            x, y, z, _, _, _ = state
            draw_bird(screen, x % WIDTH, HEIGHT - (z / 10))

            # Display state information
            state_desc = env.get_state_description()
            info_text = [
                f"Altitude: {state_desc['altitude']:.2f} m",
                f"Airspeed: {state_desc['airspeed']:.2f} m/s",
                f"Wind: ({state_desc['wind'][0]:.2f}, {state_desc['wind'][1]:.2f}, {state_desc['wind'][2]:.2f})",
                f"Reward: {reward:.2f}"
            ]
            for i, text in enumerate(info_text):
                surface = font.render(text, True, BLACK)
                screen.blit(surface, (10, 10 + i * 40))

            pygame.display.flip()
            clock.tick(30)

            if done:
                print("Episode finished")
                state = env.reset()

    except Exception as e:
        print(f"An error occurred: {e}")
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()