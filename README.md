# Oracle Chess Engine

## Introduction

Oracle is the first chess engine that can play like a human, from amateur to super GM, at any time control. Oracle can almost instantly play like a 2800-rated player in classical or an 800-rated player in blitz.

Oracle uses GPT-3.5 turbo instruct and Stockfish to deliver human-like chess moves. Users need an [OpenAI API key](https://platform.openai.com/api-keys) and a version of Stockfish, which can be downloaded [here](https://stockfishchess.org/download/).

## Features

- **Human-like Play:** Oracle adapts her predictions to any rating level from amateur to super GM.
- **Time Control Flexibility:** Oracle adapts her predictions to 4 different time controls: bullet, blitz, rapid, or classical.
- **Expected Score Evaluations:** Instead of using centipawns, Oracle evaluates positions by giving White's expected score out of 100 games. This expected score takes into account the position, the likeliest next moves, the rating of both players and the time control.
- **Two Modes:**
  - **Oracle_one_move:** Takes a PGN game as input and predicts the likeliest next moves.
  - **Oracle_pgn_file:** Takes a PGN file with multiple games and predicts every move of every game, useful for creating data to test Oracle.

## Usage

- **Oracle_one_move:** Set your openAI API key and the path to your Stockfish at the top of the file then run the file. Past the PGN up to the move you want to predict into the console, type END and press Enter
- **Oracle_pgn_file:** Set your openAI API key, the path to your Stockfish, your input pgn file, and your output csv file, and then run the file. Oracle will write her predictions for every move of every game of the PGN into the csv file.

## Author's Note

I am a FIDE Master and Woman International Master with no previous coding experience, so the code might contains mistakes or improper formulations. 

## License

This project is licensed under the MIT License. See the MIT License file for details. 

