import numpy as np
import time
import random
import pygame
import subprocess
import json
from collections import deque
import cv2
import select
import os

class Ball:
    def __init__(self, x, screen_width):
        self.x = x
        self.y = 0  # Start at top of screen
        self.radius = 20
        self.speed = 1.25  # Changed from 5 to 1.25 (4 times slower)
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.caught = False
        self.screen_width = screen_width
        
    def update(self, screen_height):
        if not self.caught:
            self.y += self.speed
        return self.y > screen_height
    
    def draw(self, screen):
        if not self.caught:
            pygame.draw.circle(screen, self.color, (self.x, int(self.y)), self.radius)

class HailoPoseEstimation:
    """Real Hailo pose estimation implementation"""
    def __init__(self):
        print("\n=== Starting HailoPoseEstimation ===")
        try:
            print("Starting rpicam-hello process...")
            # Don't create our own window, we'll use the Hailo "Pose" window
            cmd = ['rpicam-hello', '-t', '0', '--post-process-file', '/usr/share/rpi-camera-assets/hailo_yolov8_pose.json',
                  '--width', '640', '--height', '640', '--verbose']
            print("Command:", ' '.join(cmd))
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            print("Process started successfully")
            print("Waiting for pipeline initialization...")
            time.sleep(3)  # Give more time for pipeline to initialize
            
            # Create transparent overlay - match the Hailo window size
            self.overlay = np.zeros((640, 640, 4), dtype=np.uint8)  # RGBA
            
            print("Pipeline initialization complete!")
                    
        except FileNotFoundError as e:
            print("ERROR: Could not find rpicam-hello command!")
            print("Make sure Hailo software is installed and in system PATH")
            print("Error details:", str(e))
            raise
        except Exception as e:
            print("ERROR: Failed to initialize Hailo pipeline!")
            print("Error details:", str(e))
            raise
        
    def get_keypoints(self):
        try:
            # Check if process is still alive
            if self.process.poll() is not None:
                print("ERROR: Pipeline process has terminated!")
                stdout, stderr = self.process.communicate()
                print("Process stdout:", stdout)
                print("Process stderr:", stderr)
                return None
            
            # Check for any stderr output without blocking
            try:
                rlist, _, _ = select.select([self.process.stderr], [], [], 0.1)  # 0.1 second timeout
                if rlist:
                    stderr_line = self.process.stderr.readline()
                    if stderr_line and not "Viewfinder frame" in stderr_line:  # Filter out viewfinder messages
                        print("Pipeline message:", stderr_line.strip())
            except:
                pass
                
            # Read one line of output from the pipeline with timeout
            try:
                rlist, _, _ = select.select([self.process.stdout], [], [], 0.1)  # 0.1 second timeout
                if not rlist:
                    return None  # No data available
                    
                line = self.process.stdout.readline()
                if line:
                    if "Viewfinder frame" not in line:  # Filter out viewfinder messages
                        print("\nReceived data:", line.strip())
                        try:
                            # Parse the JSON output
                            data = json.loads(line)
                            if 'poses' in data and len(data['poses']) > 0:
                                print(f"Found {len(data['poses'])} poses")
                                # Return the keypoints of the first detected person
                                keypoints = data['poses'][0]['keypoints']
                                print("Original keypoints:", keypoints)
                                # Scale keypoints to match game window size
                                scaled_keypoints = []
                                for kp in keypoints:
                                    # Scale from 640x640 camera resolution to game window size (800x600)
                                    scaled_keypoints.append([
                                        int(kp[0] * 800/640),  # Scale X coordinate
                                        int(kp[1] * 600/640)   # Scale Y coordinate - now using 640 as source height
                                    ])
                                print("Scaled keypoints:", scaled_keypoints)
                                return scaled_keypoints
                            else:
                                print("No poses detected in frame")
                        except json.JSONDecodeError as e:
                            print("ERROR: Failed to parse JSON data!")
                            print("Raw data:", line)
                            print("Error details:", str(e))
                else:
                    print("No data received from pipeline")
            except Exception as e:
                print(f"ERROR reading pipeline output: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Error reading pose data: {e}")
            print("Error type:", type(e).__name__)
            print("Error details:", str(e))
            return None
            
    def cleanup(self):
        """Cleanup resources and close windows"""
        print("\nStarting cleanup...")
        if self.process:
            print("Terminating Hailo pipeline process...")
            try:
                self.process.terminate()
                print("Waiting for process to end...")
                self.process.wait(timeout=5)  # Wait up to 5 seconds
                print("Process terminated successfully")
            except subprocess.TimeoutExpired:
                print("WARNING: Process did not terminate, forcing kill...")
                self.process.kill()
                self.process.wait()
                print("Process killed")
            except Exception as e:
                print("ERROR during process cleanup:", str(e))
            
            # Add a small delay to ensure windows close
            print("Cleaning up windows...")
            time.sleep(0.5)
            # Try to close any remaining windows
            try:
                cv2.destroyAllWindows()
                print("Windows cleaned up successfully")
            except Exception as e:
                print("ERROR cleaning up windows:", str(e))
        print("Cleanup complete!")

class MockPoseEstimation:
    """Mock pose estimation for testing"""
    def __init__(self):
        self.left_hand_x = 400
        self.right_hand_x = 400
        
    def get_keypoints(self):
        """Return simulated keypoints based on mouse position"""
        mouse_x, _ = pygame.mouse.get_pos()
        # Simulate both hands following mouse with some separation
        self.left_hand_x = mouse_x - 50
        self.right_hand_x = mouse_x + 50
        
        # Return 17 keypoints with hands at mouse position
        keypoints = [[0, 0] for _ in range(17)]
        keypoints[9] = [self.left_hand_x, 400]  # Left wrist
        keypoints[10] = [self.right_hand_x, 400]  # Right wrist
        return keypoints
    
    def cleanup(self):
        pass

class BallCatchingGame:
    def __init__(self, use_mock=True):
        # Initialize pygame
        pygame.init()
        
        # Match the Hailo window size for real mode
        if not use_mock:
            self.screen_width = 640
            self.screen_height = 640
        else:
            self.screen_width = 800
            self.screen_height = 600
        
        # Set up display - use a surface for drawing that we'll overlay on the camera feed
        self.screen = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        if use_mock:
            pygame.display.set_caption("Ball Catching Game - Press Q to quit")
        
        self.font = pygame.font.Font(None, 36)
        
        # Game state
        self.score = 0
        self.balls = []
        self.last_ball_time = time.time()
        self.ball_interval = 2.0  # Time between new balls
        self.game_duration = 60  # Game length in seconds
        self.start_time = None
        
        # Initialize pose estimation
        self.use_mock = use_mock
        print("\n=== Starting game in {} mode ===".format("mock" if use_mock else "real"))
        self.pose_estimator = MockPoseEstimation() if use_mock else HailoPoseEstimation()
        
        # Hand positions and hitboxes
        self.hand_width = 40
        self.hand_height = 20
        self.default_left_hand = [self.screen_width//4, self.screen_height*4//5]  # Default position when no pose detected
        self.default_right_hand = [self.screen_width*3//4, self.screen_height*4//5]  # Default position when no pose detected
        self.current_left_hand = self.default_left_hand.copy()
        self.current_right_hand = self.default_right_hand.copy()

    def spawn_ball(self):
        """Create a new ball at a random x position"""
        x = random.randint(self.screen_width // 10, self.screen_width * 9 // 10)
        self.balls.append(Ball(x, self.screen_width))
    
    def check_catches(self, left_hand_pos, right_hand_pos):
        """Check if any balls are caught by the hands"""
        left_hand_rect = pygame.Rect(
            left_hand_pos[0] - self.hand_width//2,
            left_hand_pos[1] - self.hand_height//2,
            self.hand_width,
            self.hand_height
        )
        right_hand_rect = pygame.Rect(
            right_hand_pos[0] - self.hand_width//2,
            right_hand_pos[1] - self.hand_height//2,
            self.hand_width,
            self.hand_height
        )
        
        for ball in self.balls:
            if not ball.caught:
                ball_rect = pygame.Rect(
                    ball.x - ball.radius,
                    ball.y - ball.radius,
                    ball.radius * 2,
                    ball.radius * 2
                )
                if ball_rect.colliderect(left_hand_rect) or ball_rect.colliderect(right_hand_rect):
                    ball.caught = True
                    self.score += 10
    
    def draw_hands(self, left_hand_pos, right_hand_pos):
        """Draw rectangles representing the hands"""
        # Draw left hand
        pygame.draw.rect(self.screen, (255, 200, 200),
                        (left_hand_pos[0] - self.hand_width//2,
                         left_hand_pos[1] - self.hand_height//2,
                         self.hand_width, self.hand_height))
        
        # Draw right hand
        pygame.draw.rect(self.screen, (200, 255, 200),
                        (right_hand_pos[0] - self.hand_width//2,
                         right_hand_pos[1] - self.hand_height//2,
                         self.hand_width, self.hand_height))
    
    def draw_ui(self, time_remaining):
        """Draw score and time"""
        # Create semi-transparent background for text
        bg_surface = pygame.Surface((200, 80), pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, (0, 0, 0, 128), bg_surface.get_rect())  # Semi-transparent black
        self.screen.blit(bg_surface, (5, 5))
        
        # Draw score and time with white text
        score_text = self.font.render('Score: {}'.format(self.score), True, (255, 255, 255))
        time_text = self.font.render('Time: {}s'.format(int(time_remaining)), True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 50))

    def pygame_surface_to_cv2_overlay(self, surface):
        """Convert Pygame surface to CV2 overlay"""
        # Get the pygame surface data as a string
        image_string = pygame.image.tostring(surface, 'RGBA')
        # Convert to numpy array
        image_array = np.frombuffer(image_string, dtype=np.uint8)
        # Reshape to 2D array with RGBA channels
        image_array = image_array.reshape((surface.get_height(), surface.get_width(), 4))
        return image_array

    def run(self):
        """Main game loop"""
        self.start_time = time.time()
        running = True
        clock = pygame.time.Clock()
        last_pose_time = time.time()
        show_help = True
        
        while running:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            time_remaining = max(0, self.game_duration - elapsed_time)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_h:
                        show_help = not show_help
            
            # Clear screen with transparency
            self.screen.fill((0, 0, 0, 0))  # Fully transparent
            
            # Spawn new balls
            if current_time - self.last_ball_time > self.ball_interval:
                self.spawn_ball()
                self.last_ball_time = current_time
                self.ball_interval = max(1.0, 2.0 - elapsed_time/180)
            
            # Get hand positions from pose estimation
            keypoints = self.pose_estimator.get_keypoints()
            if keypoints and len(keypoints) >= 11:
                last_pose_time = current_time
                # Scale keypoints to our game window size
                self.current_left_hand = [
                    int(keypoints[9][0] * self.screen_width/640),
                    int(keypoints[9][1] * self.screen_height/640)
                ]
                self.current_right_hand = [
                    int(keypoints[10][0] * self.screen_width/640),
                    int(keypoints[10][1] * self.screen_height/640)
                ]
            elif current_time - last_pose_time > 5 and show_help:
                msg = self.font.render('Stand in front of camera to play!', True, (255, 255, 255))
                msg_rect = msg.get_rect(center=(self.screen_width/2, 50))
                self.screen.blit(msg, msg_rect)
                
                if not self.use_mock:
                    help_msg = self.font.render('Press H to hide this message', True, (255, 255, 255))
                    help_rect = help_msg.get_rect(center=(self.screen_width/2, 90))
                    self.screen.blit(help_msg, help_rect)
            
            # Update game state with current hand positions
            self.check_catches(self.current_left_hand, self.current_right_hand)
            
            # Draw hands at current positions
            self.draw_hands(self.current_left_hand, self.current_right_hand)
            
            # Update and draw balls
            for ball in self.balls[:]:
                if ball.update(self.screen_height):
                    self.balls.remove(ball)
                else:
                    ball.draw(self.screen)
            
            # Draw UI
            self.draw_ui(time_remaining)
            
            if not self.use_mock:
                # Convert Pygame surface to CV2 overlay
                overlay = self.pygame_surface_to_cv2_overlay(self.screen)
                # Show the overlay in the existing "Pose" window
                cv2.imshow("Pose", overlay)
                cv2.waitKey(1)
            else:
                # In mock mode, use regular Pygame display
                pygame_screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame_screen.blit(self.screen, (0, 0))
                pygame.display.flip()
            
            # Maintain consistent frame rate
            clock.tick(60)
            
            # End game when time is up
            if time_remaining <= 0:
                running = False
        
        # Show final score
        self.screen.fill((0, 0, 0, 128))  # Semi-transparent black
        final_score_text = self.font.render('Final Score: {}'.format(self.score), True, (255, 255, 255))
        text_rect = final_score_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
        self.screen.blit(final_score_text, text_rect)
        
        if not self.use_mock:
            overlay = self.pygame_surface_to_cv2_overlay(self.screen)
            cv2.imshow("Pose", overlay)
            cv2.waitKey(3000)  # Show for 3 seconds
        else:
            pygame_screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame_screen.blit(self.screen, (0, 0))
            pygame.display.flip()
            time.sleep(3)
        
        # Cleanup
        self.pose_estimator.cleanup()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    # Use real mode with game overlay on camera feed
    game = BallCatchingGame(use_mock=False)
    game.run()
