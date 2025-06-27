import cv2
import numpy as np
import time
import random
import pygame
import subprocess
import json
from collections import deque

class HailoPoseEstimation:
    def __init__(self):
        # Start the Hailo pipeline as a subprocess
        self.process = subprocess.Popen(
            ['rpicam-hello', '-t', '0', '--post-process-file', '/usr/share/rpi-camera-assets/hailo_yolov8_pose.json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        time.sleep(2)  # Give time for pipeline to initialize
        
    def get_keypoints(self):
        try:
            # Read one line of output from the pipeline
            line = self.process.stdout.readline()
            if line:
                # Parse the JSON output
                data = json.loads(line)
                if 'poses' in data and len(data['poses']) > 0:
                    # Return the keypoints of the first detected person
                    return data['poses'][0]['keypoints']
            return None
        except Exception as e:
            print(f"Error reading pose data: {e}")
            return None
            
    def cleanup(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

class PoseGame:
    def __init__(self):
        # Initialize game state
        self.score = 0
        self.game_time = 60  # Game duration in seconds
        self.start_time = None
        self.current_pose = None
        self.poses = ['hands_up', 't_pose', 'squat']  # Available poses
        self.pose_history = deque(maxlen=5)  # Store last 5 pose detections for smoothing
        
        # Initialize pygame for UI
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Pose Matching Game")
        self.font = pygame.font.Font(None, 36)
        
        # Initialize Hailo pose estimation
        self.pose_estimator = HailoPoseEstimation()
        
        # Define pose keypoint configurations
        self.pose_configs = {
            'hands_up': self._check_hands_up,
            't_pose': self._check_t_pose,
            'squat': self._check_squat
        }
    
    def _check_hands_up(self, keypoints):
        """Check if person has both hands raised above head"""
        if not keypoints or len(keypoints) < 17:
            return False
            
        # Get relevant keypoint indices
        nose = keypoints[0]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        # Check if both wrists are above nose
        return left_wrist[1] < nose[1] and right_wrist[1] < nose[1]
    
    def _check_t_pose(self, keypoints):
        """Check if person is in T-pose"""
        if not keypoints or len(keypoints) < 17:
            return False
            
        shoulders_y = abs(keypoints[5][1] - keypoints[6][1])  # Should be level
        wrists_y = abs(keypoints[9][1] - keypoints[10][1])    # Should be level
        
        return shoulders_y < 30 and wrists_y < 30  # Threshold for being level
    
    def _check_squat(self, keypoints):
        """Check if person is squatting"""
        if not keypoints or len(keypoints) < 17:
            return False
            
        hip = keypoints[11]  # Left hip
        knee = keypoints[13]  # Left knee
        ankle = keypoints[15] # Left ankle
        
        # Calculate angles between joints
        hip_knee_ankle = self._calculate_angle(hip, knee, ankle)
        return 70 < hip_knee_ankle < 110  # Angle range for squat
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def select_new_pose(self):
        """Select a new random pose for the player to match"""
        available_poses = [p for p in self.poses if p != self.current_pose]
        self.current_pose = random.choice(available_poses)
    
    def check_pose(self, keypoints):
        """Check if current pose matches required pose"""
        if self.current_pose is None or not keypoints:
            return False
            
        is_matching = self.pose_configs[self.current_pose](keypoints)
        self.pose_history.append(is_matching)
        
        # Only count as matched if majority of recent detections match
        return sum(self.pose_history) >= len(self.pose_history) * 0.6
    
    def update_score(self, matched):
        """Update game score based on pose matching"""
        if matched:
            self.score += 10
            self.select_new_pose()
    
    def draw_ui(self):
        """Draw game UI using pygame"""
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Draw current pose to match
        if self.current_pose:
            pose_text = self.font.render(f'Match pose: {self.current_pose}', True, (255, 255, 255))
            self.screen.blit(pose_text, (10, 50))
        
        # Draw time remaining if game is running
        if self.start_time:
            elapsed = time.time() - self.start_time
            remaining = max(0, self.game_time - elapsed)
            time_text = self.font.render(f'Time: {int(remaining)}s', True, (255, 255, 255))
            self.screen.blit(time_text, (10, 90))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        self.start_time = time.time()
        self.select_new_pose()
        
        running = True
        while running:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get pose keypoints from Hailo
            keypoints = self.pose_estimator.get_keypoints()
            
            # Check if pose matches and update score
            if keypoints and self.check_pose(keypoints):
                self.update_score(True)
            
            # Update UI
            self.draw_ui()
            
            # Check if game time is up
            if time.time() - self.start_time > self.game_time:
                running = False
        
        # Cleanup
        self.pose_estimator.cleanup()
        pygame.quit()

if __name__ == "__main__":
    game = PoseGame()
    game.run()
