import openai
import math
import chess
import chess.engine
import chess.pgn
import csv
import concurrent.futures
import re
import time
import logging
import signal
import sys


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key here
openai.api_key = '' 

# Set your Stockfish path here
engine_path = ''

# Set your input pgn file path here
pgn_file_path = '.pgn'

#Set your output csv file path here
output_file_path = '.csv'

total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0
best_move_notation = "" 

def extract_moves_from_pgn(pgn_file_path):
    all_moves = []
    with open(pgn_file_path, 'r', encoding='utf-8-sig') as file:
        games = file.read().split('\n\n\n')
        
    for game in games:
        lines = game.split('\n')
        for line in lines:
            if not line.startswith('['):
                line = re.sub(r'\{[^}]*\}|\([^)]*\)', '', line)
                moves = line.split(' ')
                for move in moves:
                    if not re.match(r'\d+\.', move) and move != '' and move not in {"1-0", "0-1", "1/2-1/2", "*"}:
                        all_moves.append(move)
    return all_moves

def process_pgn(pgn_file_path):
    prompts = []
    game_index = 0

    with open(pgn_file_path, 'r', encoding='utf-8-sig') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_index += 1

            header = []
            for key in game.headers.keys():
                if key in ['Event', 'Site', 'Round', 'White', 'Black', 'WhiteElo', 'BlackElo', 'TimeControl']:
                    header.append(f'[{key} "{game.headers[key]}"]')
            header_str = '\n'.join(header)
            
            node = game
            moves = []
            while not node.is_end():
                next_node = node.variation(0)
                moves.append(node.board().san(next_node.move))
                node = next_node
            
            moves_str = ' '.join(moves).strip()
            move_list = moves_str.split(' ')
            move_list = [move for move in move_list if move and not move[0].isdigit()]

            if move_list:
                prompts.append((game_index, 1, f"{header_str}\n\n1."))
                current_moves = "1."
                move_number = 1
                for i in range(len(move_list)-1):
                    if i % 2 == 0:
                        current_moves += f" {move_list[i]}"
                        prompts.append((game_index, move_number, f"{header_str}\n\n{current_moves}"))
                    else:
                        current_moves += f" {move_list[i]}"
                        if i + 1 < len(move_list):
                            move_number += 1
                            current_moves += f" {move_number}."
                        prompts.append((game_index, move_number, f"{header_str}\n\n{current_moves}"))

    print(f"Number of prompts generated: {len(prompts)}")
    return prompts

def get_legal_moves(board):
    return [board.san(move) for move in board.legal_moves]

def analyze_moves(engine, board, n, time_limit=1.3, depth=20, threads=8, hash_size=512):
    engine.configure({"Threads": threads, "Hash": hash_size})

    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit, depth=depth), multipv=n)
    except Exception as e:
        logging.error(f"Engine analysis failed: {e}")
        return []

    evals_with_mate = []
    for i in range(n):
        try:
            move_info = info[i]
            move = board.san(chess.Move.from_uci(move_info['pv'][0].uci()))
            eval_score = move_info['score'].relative

            if eval_score.is_mate():
                eval_score = f"mate:{eval_score.mate()}"
            else:
                eval_score = clamp_score(eval_score.score())
                if board.turn == chess.BLACK:
                    eval_score = -eval_score

            evals_with_mate.append((move, eval_score))
        except Exception as e:
            logging.error(f"Failed to process move {i}: {e}")
            evals_with_mate.append((None, None))

    return evals_with_mate
def clamp_score(score):
    return max(-2000, min(2000, score))

def make_api_call_with_backoff(prompt, retries, model):
    global total_input_tokens, total_output_tokens, total_cost
    attempt = 0
    while attempt < retries:
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=1,
                logprobs=10
            )

            input_tokens = response['usage']['prompt_tokens']
            output_tokens = response['usage']['completion_tokens']
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
                
            cost_input = input_tokens * 1.50 / 1_000_000
            cost_output = output_tokens * 2.00 / 1_000_000
            total_cost += cost_input + cost_output
            return response
        except openai.error.RateLimitError as e:
            logging.error(f"Rate limit error: {e}")
            sleep_time = min(60, (2 ** attempt))
            logging.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            attempt += 1
        except Exception as e:
            logging.error(f"API call error: {e}")
            time.sleep(1)
            attempt += 1
    return None

def get_best_move_notation(best_win_percentage, new_norm_prob, board_turn, current_win_percentage):
    if (board_turn == chess.WHITE and best_win_percentage > 20 and best_win_percentage - current_win_percentage > 5) or \
       (board_turn == chess.BLACK and best_win_percentage < 80 and current_win_percentage - best_win_percentage > 5):
        if new_norm_prob < 50:
            return "!!"
        elif 50 <= new_norm_prob <= 80:
            return "!"
        else:
            return ""
    return ""

def find_best_move_index(moves, turn):
    if turn == chess.WHITE:
        best_eval = max
    else:
        best_eval = min

    best_move = best_eval(moves, key=lambda x: x[1])
    best_move_idx = moves.index(best_move)
    return best_move_idx, best_move[1]

def get_notation(percentage_loss, is_best_move, new_norm_prob, eval_score, board_turn):
    global best_move_notation  
    
    if is_best_move:
        return best_move_notation  
    
    if percentage_loss == 0:
        if best_move_notation in ["!!", "!"]:
            return "!!" if best_move_notation == "!!" else "!" 
        else:
            return ""  
    
    if 0 < percentage_loss <= 0.5:
        if best_move_notation in ["!!", "!"]:
            return "!!" if best_move_notation == "!!" else "!" 
        else:
            return ""      
    elif 0.5 < percentage_loss <= 2.5:
        if best_move_notation == "!!":
            return "!" 
        else:
            return ""
    elif 2.5 < percentage_loss <= 5:
        return ""
    elif 5 < percentage_loss <= 10:
        return "?!"  
    elif 10 < percentage_loss <= 20:
        return "?"  
    else:
        return "??"  


def get_top_sequences(prompt, legal_moves, model="gpt-3.5-turbo-instruct", depth=5, retries=3, prob_threshold=0.001):
    sequences = []
    request_count = 0

    if depth == 0:
        return [("", 0)]

    def is_valid_continuation(token, partial_sequence, legal_moves):
        potential_sequence = partial_sequence + token
        return any(legal_move.startswith(potential_sequence) for legal_move in legal_moves)

    def expand_sequence(prompt, seq, seq_logprob, legal_moves, depth, retries, prob_threshold):
        if depth == 0:
            return

        expanded_prompt = prompt + " " + seq
        response = make_api_call_with_backoff(expanded_prompt, retries, model)

        if not response or not response.choices or 'logprobs' not in response.choices[0] or not response.choices[0]['logprobs']:
            return

        top_logprobs = response.choices[0]['logprobs'].get('top_logprobs')
        if not top_logprobs or len(top_logprobs) == 0:
            return

        top_tokens = top_logprobs[0]
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for next_token, next_logprob in top_tokens.items():
                next_token = next_token.strip()
                combined_sequence = seq + next_token
                combined_logprob = seq_logprob + next_logprob

                combined_prob = math.exp(combined_logprob)
                next_token_prob = math.exp(next_logprob)

                if next_token_prob < prob_threshold or combined_prob < prob_threshold:
                    continue

                if not is_valid_continuation(next_token, seq, legal_moves):
                    continue

                possible_moves = [move for move in legal_moves if move.startswith(combined_sequence)]
                if len(possible_moves) == 1:
                    complete_sequence = possible_moves[0]
                    sequences.append((complete_sequence, combined_logprob))
                    legal_moves.remove(complete_sequence)
                else:
                    
                    if "O-O" in legal_moves and "O-O-O" in legal_moves:
                        if combined_sequence == "O-O":
                            
                            response_after_OO = make_api_call_with_backoff(prompt + " " + combined_sequence, retries, model)
                            if response_after_OO:
                                top_logprobs_after_OO = response_after_OO.choices[0]['logprobs'].get('top_logprobs')
                                if top_logprobs_after_OO:
                                    top_tokens_after_OO = top_logprobs_after_OO[0]
                                    next_token_logprob = top_tokens_after_OO.get("-O", None)

                                    if next_token_logprob is not None:
                                        combined_logprob_OOO = combined_logprob + next_token_logprob
                                        combined_prob_OOO = math.exp(combined_logprob_OOO)
                                        combined_prob_OO = combined_prob - combined_prob_OOO
                                        combined_logprob_OO = math.log(combined_prob_OO) if combined_prob_OO > 0 else float('-inf')

                                        if combined_prob_OOO >= prob_threshold:
                                            sequences.append(("O-O-O", combined_logprob_OOO))
                                            legal_moves.remove("O-O-O")

                                        if combined_logprob_OO != float('-inf') and combined_prob_OO >= prob_threshold:
                                            sequences.append((combined_sequence, combined_logprob_OO))
                                            legal_moves.remove(combined_sequence)
                                    else:
                                        if combined_prob >= prob_threshold:
                                            sequences.append((combined_sequence, combined_logprob))
                                            legal_moves.remove(combined_sequence)
                            else:
                                if combined_prob >= prob_threshold:
                                    sequences.append((combined_sequence, combined_logprob))
                                    legal_moves.remove(combined_sequence)
                        else:
                            futures.append(executor.submit(expand_sequence, prompt, combined_sequence, combined_logprob, legal_moves, depth - 1, retries, prob_threshold))
                    else:
                        futures.append(executor.submit(expand_sequence, prompt, combined_sequence, combined_logprob, legal_moves, depth - 1, retries, prob_threshold))

            done, not_done = concurrent.futures.wait(futures, timeout=300)

            if not_done:
                logging.warning(f"Some tasks did not complete within the timeout period: {not_done}")

    response = make_api_call_with_backoff(prompt, retries, model)
    if not response:
        return []

    if not response.choices or 'logprobs' not in response.choices[0] or not response.choices[0]['logprobs']:
        return []

    try:
        top_logprobs = response.choices[0]['logprobs'].get('top_logprobs')
        if not top_logprobs or len(top_logprobs) == 0:
            return []
        top_tokens = top_logprobs[0]
    except IndexError:
        return []

    initial_sequences = []

    for token, logprob in top_tokens.items():
        token = token.strip()
        if not token:
            continue

        token_prob = math.exp(logprob)
        if token_prob < prob_threshold:
            continue

        if not is_valid_continuation(token, "", legal_moves):
            continue

        initial_sequences.append((token, logprob, prompt + token))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(expand_sequence, prompt, token, logprob, legal_moves.copy(), depth - 1, retries, prob_threshold) for token, logprob, _ in initial_sequences]
        done, not_done = concurrent.futures.wait(futures, timeout=1000)

        if not_done:
            logging.warning(f"Some tasks did not complete within the timeout period: {not_done}")

    if not sequences and prob_threshold > 0.01:
        return get_top_sequences(prompt, legal_moves, model, depth, retries, prob_threshold / 10)

    return sequences

def calculate_win_percentage(rating, centipawns):
    
    coefficient = rating * -0.00000274 + 0.00048
    
    exponent = coefficient * centipawns

    win_percentage = 100 * (0.5 + (0.5 * (2 / (1 + math.exp(exponent)) - 1)))
    
    return win_percentage

def parse_time_control(time_control):
    phases = time_control.split(':')
    total_time = 0
    average_moves = 40 
    increments = []

    for phase in phases:
        if '+' in phase:
            base, increment = phase.split('+')
            increments.append(int(increment))
        else:
            increments.append(0)

    for i, phase in enumerate(phases):
        if '/' in phase:
            moves, base_increment = phase.split('/')
            base_time = int(base_increment.split('+')[0])
            moves = int(moves)
            total_time += base_time + (moves * increments[i])
        else:
            base_time = int(phase.split('+')[0])
            if i == len(phases) - 1:
                total_time += base_time + (average_moves * increments[i])
            else:
                total_time += base_time

    return total_time

def determine_game_type(time_control):
    if time_control == '-':
        return "Unknown"
    
    total_time = parse_time_control(time_control)

    if total_time < 180:
        return "Bullet"
    elif total_time < 600:
        return "Blitz"
    elif total_time < 3600:
        return "Rapid"
    else:
        return "Classical"


def process_pgn_and_analyze(pgn_file_path, engine_path, model="gpt-3.5-turbo-instruct", depth=5, prob_threshold=0.001, output_file_path='analysis_results.csv', default_rating=2000, default_game_type="classical"):
    all_moves = extract_moves_from_pgn(pgn_file_path)
    global best_move_notation
    prompts = process_pgn(pgn_file_path)

    results = []
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    start_time = time.time()

    def save_result(result):
        with open(output_file_path, 'a', newline='') as csvfile:
            fieldnames = [
                'game_index', 'movenumber', 'white_or_black', 'prediction', 'notation', 'raw_prob', 
                'new_norm_prob', 'win_percentage', 'eval_centipawns', 'eval_with_mate', 'is_played',
                'best_move_importance', 'rating', 'game_type'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)

    def signal_handler(sig, frame):
        logging.warning('Interrupt signal received. Saving results...')
        engine.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = [
            'game_index', 'movenumber', 'white_or_black', 'prediction', 'notation', 'raw_prob', 
            'new_norm_prob', 'win_percentage', 'eval_centipawns', 'eval_with_mate', 'is_played',
            'best_move_importance', 'rating', 'game_type'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    try:
        move_index = 0
        current_game_index = None
        current_move_number = 1
        
        for prompt_index, (game_index, move_number, prompt) in enumerate(prompts):
            if game_index != current_game_index:
                current_game_index = game_index
                current_move_number = 1

            logging.info(f"Prompt to GPT:\n{prompt}")
            try:
                pgn_content = "\n".join(prompt.split("\n\n")[:-1])
                white_elo = re.search(r'\[WhiteElo\s+"(\d+)"\]', pgn_content)
                black_elo = re.search(r'\[BlackElo\s+"(\d+)"\]', pgn_content)
                time_control_match = re.search(r'\[TimeControl\s+"([^"]+)"\]', pgn_content)
                
                if time_control_match:
                    time_control = time_control_match.group(1)
                    game_type = determine_game_type(time_control)
                    game_type = game_type.lower()
                else:
                    game_type = default_game_type

                if white_elo and black_elo:
                    try:
                        white_elo = int(white_elo.group(1))
                        black_elo = int(black_elo.group(1))
                    except ValueError:
                        white_elo = black_elo = default_rating
                else:
                    white_elo = black_elo = default_rating

                def adjust_rating(rating, game_type):
                    adjustments = {"bullet": 0, "blitz": 200, "rapid": 700, "classical": 1200}
                    rating += adjustments.get(game_type, 0)
                    rating = max(1000, min(4100, rating))
                    return rating       
                white_elo = adjust_rating(white_elo, game_type)
                black_elo = adjust_rating(black_elo, game_type)         


                board = chess.Board()

                moves = prompt.split("\n\n")[1].strip().split(" ")
                for move in moves:
                    if re.match(r'\d+\.', move):
                        continue 
                    try:
                        board.push_san(move)
                    except ValueError:
                        logging.warning(f"Invalid move detected and skipped: {move}")
                        continue
                
                rating = white_elo if board.turn == chess.WHITE else black_elo

                legal_moves = get_legal_moves(board)
                logging.info(f"Legal moves before Stockfish: {legal_moves}")
                
                top_sequences = get_top_sequences(prompt, legal_moves, model, depth, retries=3, prob_threshold=prob_threshold)
                
                logging.info(f"Top sequences from GPT: {top_sequences}")
                
                evals_with_mate = analyze_moves(engine, board, len(legal_moves))
                evals = []
                eval_with_mate_dict = {}
                for move, eval_score in evals_with_mate:
                    eval_with_mate_dict[move] = eval_score
                    if isinstance(eval_score, str) and eval_score.startswith('mate:'):
                        mate_value = int(eval_score.split(':')[1])
                        if board.turn == chess.WHITE:
                            eval_score = 10000 if mate_value > 0 else -10000
                        else:
                            eval_score = -10000 if mate_value > 0 else 10000
                    evals.append((move, eval_score))

                best_move_idx, best_eval_value = find_best_move_index(evals, board.turn)

                best_move = evals[best_move_idx][0]
                
                actual_move = all_moves[move_index]

                move_probabilities = {}
                for seq, logprob in top_sequences:
                    probability = math.exp(logprob) * 100
                    if seq in move_probabilities:
                        move_probabilities[seq] += probability
                    else:
                        move_probabilities[seq] = probability

                if best_move not in move_probabilities:
                    move_probabilities[best_move] = prob_threshold * 100

                total_probability = sum(move_probabilities.values())
                raw_probabilities = {move: (prob / total_probability) * 100 for move, prob in move_probabilities.items()}
                normalized_moves_initial = raw_probabilities.copy()

                win_percentages_1500 = {move: calculate_win_percentage(1500, eval_score) for move, eval_score in evals}
                win_percentages_rating = {move: calculate_win_percentage(rating, eval_score) for move, eval_score in evals}
                
                def find_win_percentages(win_percentages):
                    sorted_percentages = sorted(win_percentages.values())
                    highest = sorted_percentages[-1]
                    lowest = sorted_percentages[0]
                    second_highest = sorted_percentages[-2] if len(sorted_percentages) > 1 else None
                    second_lowest = sorted_percentages[1] if len(sorted_percentages) > 1 else None
                    return highest, lowest, second_highest, second_lowest

                highest_1500, lowest_1500, second_highest_1500, second_lowest_1500 = find_win_percentages(win_percentages_1500)
                highest_rating, lowest_rating, second_highest_rating, second_lowest_rating = find_win_percentages(win_percentages_rating)

                diff_high_1500 = (highest_1500 - second_highest_1500) if second_highest_1500 is not None else 0
                diff_high_rating = (highest_rating - second_highest_rating) if second_highest_rating is not None else 0
                diff_low_1500 = (second_lowest_1500 - lowest_1500) if second_lowest_1500 is not None else 0
                diff_low_rating = (second_lowest_rating - lowest_rating) if second_lowest_rating is not None else 0


                if board.turn == chess.WHITE:
                    best_move_importance = max(diff_high_1500, diff_high_rating)
                else:
                    best_move_importance = max(diff_low_1500, diff_low_rating)

                intercept_best_move = {
                    'classical': (( (min(rating, 4100)) / 4100) + (20 * (best_move_importance / 100) ** 0.5)) * (min(rating, 4100)) / 4100,
                    'rapid': (( min(rating, 3700)) / 3700 + (14 * (best_move_importance / 100)) ** 0.5) * (min(rating, 3700)) / 3700,
                    'blitz': (( min(rating, 3600)) / 3600 + (6 * (best_move_importance / 100) ** 0.5)) * (min(rating, 3600)) / 3600,
                    'bullet': ((min(rating, 3400)) / 3400 + (2 * (best_move_importance / 100) ** 0.5)) * (min(rating, 3400)) / 3400
                }.get(game_type, (( (min(rating, 4100)) / 4100) + (20 * (best_move_importance / 100) ** 0.5)) * (min(rating, 4100)) / 4100)

                slope_best_move = (100 - intercept_best_move) / 100

                normalized_moves_initial[best_move] = intercept_best_move + slope_best_move * normalized_moves_initial[best_move]               

                total_probability_modified = sum(normalized_moves_initial.values())
                normalized_moves_best_move_adjusted = {move: (prob / total_probability_modified) * 100 for move, prob in normalized_moves_initial.items()}

                mate_in_dict = {}
                for move, eval_score in evals_with_mate:
                    if isinstance(eval_score, str) and eval_score.startswith('mate:'):
                        mate_in_dict[move] = abs(int(eval_score.split(':')[1]))

                multiplier = {
                    'classical': 150,
                    'rapid': 500,
                    'blitz': 1000,
                    'bullet': 4000
                }.get(game_type, 150) 

                percentage_losses = {}
                for move in normalized_moves_best_move_adjusted:
                    win_percentage_1500 = win_percentages_1500[move]
                    win_percentage_rating = win_percentages_rating[move]
                    
                    if board.turn == chess.WHITE:
                        percentage_loss_1500 = highest_1500 - win_percentage_1500
                        percentage_loss_rating = highest_rating - win_percentage_rating
                    else:
                        percentage_loss_1500 = win_percentages_1500[move] - lowest_1500
                        percentage_loss_rating = win_percentages_rating[move] - lowest_rating

                    percentage_loss = max(percentage_loss_1500, percentage_loss_rating)
                    percentage_losses[move] = percentage_loss

                    mate_in = mate_in_dict.get(move)
                    if mate_in is not None and mate_in > 0:
                        if percentage_losses[move] == 0:
                            normalized_moves_best_move_adjusted[move] *= (1 + (rating / (multiplier * mate_in)))
                        elif percentage_losses[move] > 0:
                            normalized_moves_best_move_adjusted[move] /= (1 + (rating / (multiplier * mate_in)))
                    if percentage_loss > 0:
                        normalized_moves_best_move_adjusted[move] /= (1 + (percentage_loss / ((-19 * rating) / 600 + 131.67)))

                total_probability_modified = sum(normalized_moves_best_move_adjusted.values())
                normalized_moves_final = {move: (prob / total_probability_modified) * 100 for move, prob in normalized_moves_best_move_adjusted.items()}

                sorted_moves = sorted(normalized_moves_final.items(), key=lambda x: x[1], reverse=True)

                if actual_move not in normalized_moves_final:
                    sorted_moves.append((actual_move, 0))
                    raw_probabilities[actual_move] = 0
                    
                win_percentages = {}
                for move, norm_prob in sorted_moves:
                    eval_score = next((eval_score for m, eval_score in evals if m == move), None)
                    if eval_score is not None:
                        win_percentages[move] = calculate_win_percentage(rating, eval_score)

                current_win_percentage = sum(win_percentages[move] * norm_prob / 100 for move, norm_prob in sorted_moves if move in win_percentages)

                best_move_notation = get_best_move_notation(
                    calculate_win_percentage(rating, best_eval_value), 
                    normalized_moves_final[best_move], 
                    board.turn, 
                    current_win_percentage
                )


                for move, norm_prob in sorted_moves:
                    eval_score = next((eval_score for m, eval_score in evals if m == move), None)
                    if eval_score is None:
                        continue

                    win_percentage = win_percentages.get(move, 0)

                    raw_prob = raw_probabilities[move]
                    is_played = 1 if move == actual_move else 0

                    notation = get_notation(percentage_losses.get(move, 0), move == best_move, norm_prob, eval_score, board.turn)


                    result = {
                        'game_index': game_index,
                        'movenumber': current_move_number,
                        'white_or_black': "." if (board.turn == chess.WHITE) else "...",
                        'prediction': move,
                        'notation': notation,
                        'raw_prob': raw_prob,
                        'new_norm_prob': norm_prob,
                        'win_percentage': win_percentage,
                        'eval_centipawns': eval_score,
                        'eval_with_mate': eval_with_mate_dict.get(move),
                        'is_played': is_played,
                        'best_move_importance': best_move_importance, 
                        'rating': rating,  
                        'game_type': game_type  
                    }
                    save_result(result)
                
                move_index += 1
                if board.turn == chess.BLACK:
                    current_move_number += 1  

                if move_index >= len(all_moves):
                    break
            except Exception as e:
                logging.error(f"An error occurred while processing game {game_index}, move {move_number}: {e}")
                continue

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        engine.quit()
        end_time = time.time()
        print(f"Final results saved. Total moves processed: {move_index}")
        print(f"Total Time taken: {end_time - start_time:.2f} seconds")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")


if __name__ == "__main__": 
    process_pgn_and_analyze(pgn_file_path, engine_path, output_file_path=output_file_path)
    logging.info(f"Results saved to {output_file_path}")
