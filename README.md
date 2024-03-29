# Infinite Maze Game

## Overview
Infinite Maze is a basic 'maze-style' game, in which participants attempt
to get as far through the maze as possible while keeping a steady pace. The
maze is randomly generated with each game, which adds to the difficulty of
attempting to get as far as possible. 

What makes this maze infinite is exactly that - there is no end to the maze. 
As long as participants are able to keep up with the pace of the maze, the 
game will never end.

## Setup
The following commands in this section (**bolded**) should be run in a terminal window after navigating to the root directory of the application.

1. To install the requirements needed to run the game, type **make**.
2. To run the setup and initialization of the game, type **python setup.py install**.

### Execution
To compile and execute the game, type **python infinite_maze**

## Game Definitions
|  Term      |  Definition  |
| ---------- | ------------ |
| **Player** | Represented by the 'dot' in game. |
| **Pace**   | A game mechanic in which the game tries to catch up to the Player. When the Pace successfully catches up to the Player, the game is over. |
| **Wall**   | Inhibits the movement of the Player. |
| **Point**  | Awarded when the Player based on movement. |

## Game Rules
- Objective is to get as far right through the maze as possible without being caught by the pace.
- Points are awarded based on each movement the player performs to the **right**.
- Points are taken away based on each movement the player performs to the **left**.
- Pace will start at the 30 second mark, and become incrementally quicker every 30 seconds onwards.

## Controls
|  Key           |  Action                 |
| -------------- | ----------------------- |
| **w, a, s, d** | up, left, down, right   |
| **space**      | pause                   |
| **esc, q**     | end game                |
| **y, n**       | yes, no (when prompted) |


## Specifications
The only software needed to run this program is Python. Infinite Maze Game
was developed and tested using Python (v3.10.4).

## Special Features
Infinite Maze Game makes use of the Pygame library, which is a simple and easy way to
create basic games, import images, and draw shapes. More information on Pygame can be found
[**here**](https://www.pygame.org/docs/).

## For Developers
### Code Structure
|  Directory           |  Description  |
| -------------------- | ------------- |
| **./**               | All setup and information files. |
| **./img/**           | Image resources used in the game. |
| **./infinite_maze/** | All Python code, and complied files. | 
