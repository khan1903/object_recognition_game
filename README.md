# Object Recognition Game

This is an interactive object recognition game where the user needs to show certain objects (like a book, cup, or bottle) in front of the camera to earn points.

## Setup

1. Clone this repository.
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv3 model weights and config from [YOLOv3 website](https://pjreddie.com/darknet/yolo/) and place them under `models/` folder.

4. Download the `coco.names` class file and place it under the `data/` folder.

5. Run the game:

    ```bash
    python src/game.py
    ```

## Game Rules

- The game runs for 30 seconds.
- The user needs to show specific objects (book, cup, bottle) to earn points.
- The score and time left are displayed in the game window.
