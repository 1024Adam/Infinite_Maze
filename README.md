# Infinite Maze Game

## Overview
Infinite Maze is a basic 'maze-style' game, in which participants attempt
to get as far through the maze as possible while keeping a steady pace. The
maze is randomly generated with each game, which adds to the difficulty of
attempting to get as far as possible. 

What makes this maze infinite is exactly that - there is no end to the maze. 
As long as participants are able to keep up with the pace of the maze, the 
game will never end.

## Execution
From the root directory: *python infinite_maze*

## Game Definitions
|  Term  |  Definition  |
| ------ | ------------ |
| Player | Represented by the 'dot' in game. |
| Pace   | A game mechanic in which the game tries to catch up to the player. When the pace successfully catches up, the game is over. |
| Wall   | Inhibits the movement of the Player. |
| Point  | Awarded when the Player based on movement. |

## Game Rules
- Objective is to get as far right through the maze as possible without being caught by the pace.
- Points are awarded based on each movement the player performs to the *right*.
- Points are taken away based on each movement the player performs to the *left*.
- Pace will start at the 30 second mark, and become incrementally quicker every 30 seconds onwards.

## Assets
Infinite Maze Game makes use of the Pygame library, which is a simple and easy way to
create basic games, import images, and draw shapes.

## Specifications
The only software needed to run this program is Python. Infinite Maze Game
was developed and tested using Python (v2.7.11).


