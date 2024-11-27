from PySudoku import play
from pygame_screen_record import ScreenRecorder
import pygame
import time
import SudokuSquare
from GameResources import *


def visualize_assignments(assignments):
    """
    Visualizes the set of assignments created by the Sudoku AI and records a video dynamically.
    """
    # Filter assignments to include only updated states
    last_assignment = None
    filtered_assignments = []

    for i in range(len(assignments)):
        if last_assignment:
            last_assignment_items = [item for item in last_assignment.items() if len(item[1]) == 1]
            current_assignment_items = [item for item in assignments[i].items() if len(item[1]) == 1]
            shared_items = set(last_assignment_items) & set(current_assignment_items)
            if len(shared_items) < len(current_assignment_items):
                filtered_assignments.append(assignments[i])
        last_assignment = assignments[i]

    # Initialize Pygame and ScreenRecorder
    pygame.init()
    size = (700, 700)  # Match the display size used in your PySudoku play function
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Sudoku Solver Visualization")
    background_image = pygame.image.load("./images/sudoku-board-bare.jpg").convert()

    # Create a clock to control the frame rate
    clock = pygame.time.Clock()

    # Initialize the ScreenRecorder
    recorder = ScreenRecorder(10)  # 30 FPS recording
    recorder.start_rec()

    try:
        for values in filtered_assignments:
            # Event processing to prevent unresponsiveness
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # Draw background and clear screen
            screen.blit(background_image, (0, 0))

            # Draw the current state of the board
            theSquares = []
            digits = '123456789'
            rows = 'ABCDEFGHI'
            for y in range(9):
                for x in range(9):
                    if x in (0, 1, 2):
                        startX = (x * 57) + 38
                    if x in (3, 4, 5):
                        startX = (x * 57) + 99
                    if x in (6, 7, 8):
                        startX = (x * 57) + 159

                    if y in (0, 1, 2):
                        startY = (y * 57) + 35
                    if y in (3, 4, 5):
                        startY = (y * 57) + 100
                    if y in (6, 7, 8):
                        startY = (y * 57) + 165

                    col = digits[x]
                    row = rows[y]
                    string_number = values[row + col]
                    if len(string_number) > 1 or string_number == '' or string_number == '.':
                        number = None
                    else:
                        number = int(string_number)
                    theSquares.append(SudokuSquare.SudokuSquare(number, startX, startY, "N", x, y))

            # Draw each square
            for num in theSquares:
                num.draw()

            # Update the display (double buffering prevents flickering)
            pygame.display.update()

            # Control the frame rate (reduce excessive redraws)
            clock.tick(30)

            # Pause for a short duration to visualize updates
            time.sleep(0.2)  # Adjust delay for smoother or faster updates
    finally:
        recorder.stop_rec()
        recorder.save_recording("sudoku_solving.avi")
        print("Recording saved as sudoku_solving.avi")

        # Quit Pygame
        pygame.quit()

