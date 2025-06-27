# Pose Matching Game with Raspberry Pi and Hailo-8L

A fun interactive game that uses pose estimation to match specific poses and score points!

## Requirements

- Raspberry Pi 5
- Raspberry Pi AI Kit with Hailo-8L
- USB Camera or Raspberry Pi Camera
- Python 3.9+

## Installation

1. Install the Hailo-8L drivers and software:
```bash
sudo apt install hailo-all
sudo reboot
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## How to Play

1. Run the game:
```bash
python main.py
```

2. The game will show you a pose to match (hands up, T-pose, or squat)
3. Try to match the pose shown on screen
4. Hold the pose for a moment to score points
5. You have 60 seconds to score as many points as possible!

## Pose Descriptions

- **Hands Up**: Raise both hands above your head
- **T-Pose**: Stand with arms stretched out horizontally like a T
- **Squat**: Bend your knees into a squat position

## Scoring

- Each successful pose match: 10 points
- Try to get the highest score in 60 seconds!

## Troubleshooting

1. Make sure the Hailo-8L is properly connected:
```bash
lspci | grep Hailo
```

2. Verify Hailo software installation:
```bash
hailortcli fw-control identify
```

## Contributing

Feel free to contribute to this project by submitting issues or pull requests! 
