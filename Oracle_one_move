import openai
import math
import chess
import chess.engine
import chess.pgn
import io
import time
from tabulate import tabulate
import re
import concurrent.futures

# Set your OpenAI API key here
openai.api_key = ''

# Set your Stockfish path here
engine_path = ''

total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0
best_move_notation = "" 

def clean_pgn(pgn):
    lines = pgn.strip().split('\n')
    cleaned_lines = []
    headers = []
    moves = []
    header_present = False
    moves_started = False
    
    required_headers = ['[Event ', '[Site ', '[Round ', '[White ', '[Black ', '[WhiteElo ', '[BlackElo ', '[TimeControl ']
    for line in lines:
        if line.startswith('['):
            if any(line.startswith(header) for header in required_headers):
                headers.append(line.strip()) 
                header_present = True
        else:
            if line.strip(): 
                moves_started = True
            if moves_started:
                moves.append(line.strip())

    if not header_present:
        default_header = [
            '[White "?"]',
            '[Black "?"]',
            '[WhiteElo "?"]',
            '[BlackElo "?"]'
        ]
        headers = default_header 

    if headers and moves:
        cleaned_lines = headers + [''] + moves
    else:
        cleaned_lines = headers + moves

    moves_str = ' '.join(moves)

    if moves_str.endswith(('1-0', '0-1', '1/2-1/2', '*')):
        moves_str = moves_str.rsplit(' ', 1)[0]

    moves_str = re.sub(r'\s?\{[^}]*\}', '', moves_str) 
    moves_str = re.sub(r'\s?\$\d{1,2}', '', moves_str) 
    moves_str = re.sub(r'\s?\d+\.\.\.', '', moves_str) 
    moves_str = re.sub(r'[?!]', '', moves_str) 

    # Remove nested parentheses
    while re.search(r'\([^()]*\)', moves_str):
        moves_str = re.sub(r'\([^()]*\)', '', moves_str)

    moves_str = re.sub(r'\s+', ' ', moves_str).strip()
    if moves_str[-1] != '.':
        last_space_index = moves_str.rfind(' ')
        

        if moves_str[last_space_index - 1] == '.':
            pass
        else:
            last_dot_index = moves_str.rfind('.')
            
            move_number_start = moves_str.rfind(' ', 0, last_dot_index) + 1
            move_number = int(moves_str[move_number_start:last_dot_index]) + 1
            moves_str += f" {move_number}."

    moves = moves_str.split(' ')

    cleaned_lines = headers + [''] + [' '.join(moves)]

    result = '\n'.join(cleaned_lines)
    return result

def get_legal_moves(board):
    return [board.san(move) for move in board.legal_moves]

def get_top_sequences(prompt, legal_moves, model, depth, retries=3, prob_threshold=0.001):
    sequences = []

    if depth == 0:
        return [("", 0)]

    def make_api_call(prompt, retries):
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
            except Exception as e:
                attempt += 1
                time.sleep(1)
        return None

    def is_valid_continuation(token, partial_sequence, legal_moves):
        potential_sequence = partial_sequence + token
        return any(legal_move.startswith(potential_sequence) for legal_move in legal_moves)

    def expand_sequence(prompt, seq, seq_logprob, legal_moves, depth, retries, prob_threshold):
        if depth == 0:
            return

        expanded_prompt = prompt + " " + seq
        response = make_api_call(expanded_prompt, retries)

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
                            response_after_OO = make_api_call(prompt + " " + combined_sequence, retries)
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
            concurrent.futures.wait(futures)

    response = make_api_call(prompt, retries)
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
        concurrent.futures.wait(futures)

    if not sequences and prob_threshold > 0.01:
        return get_top_sequences(prompt, legal_moves, model, depth, retries, prob_threshold / 10)

    return sequences

def clamp_score(score):
    return max(-2000, min(2000, score))

def calculate_win_percentage(rating, centipawns):
    coefficient = rating * -0.00000274 + 0.00048
    
    exponent = coefficient * centipawns

    win_percentage = 100 * (0.5 + (0.5 * (2 / (1 + math.exp(exponent)) - 1)))
    
    return win_percentage

def analyze_moves(engine, board, n, time_limit=1.3, depth=20, threads=8, hash_size=512):
    engine.configure({"Threads": threads, "Hash": hash_size})
    
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit, depth=depth), multipv=n)
        evals = []
        for i in range(n):
            move_info = info[i]
            move = board.san(chess.Move.from_uci(move_info['pv'][0].uci()))
            eval_score = move_info['score'].relative
            
            if eval_score.is_mate():
                eval_score = f"mate:{eval_score.mate()}"
            else:
                eval_score = clamp_score(eval_score.score())
                if board.turn == chess.BLACK:
                    eval_score = -eval_score

            evals.append((move, eval_score))
        return evals
    except Exception as e:
        return []
    finally:
        engine.quit()
        
def get_best_move_notation(best_win_percentage, new_norm_prob, board_turn, current_win_percentage):
    global best_move_notation  
    best_move_notation = ""  
    
    if (board_turn == chess.WHITE and best_win_percentage > 20 and best_win_percentage - current_win_percentage > 5) or \
       (board_turn == chess.BLACK and best_win_percentage < 80 and current_win_percentage - best_win_percentage > 5):
        if new_norm_prob < 40:
            best_move_notation = "!!"
        elif 40 <= new_norm_prob <= 80:
            best_move_notation = "!"
        else:
            best_move_notation = ""
    return best_move_notation

def find_best_move_index(moves, turn):
    if turn == chess.WHITE:
        best_eval = max
    else:
        best_eval = min

    best_move = best_eval(moves, key=lambda x: x[1])
    best_move_idx = moves.index(best_move)
    return best_move_idx, best_move[1]

def get_color_and_notation(percentage_loss, is_best_move):
    global best_move_notation  
    
    if is_best_move:
        return "\033[96m", best_move_notation  
    
    if percentage_loss == 0:
        if best_move_notation in ["!!", "!"]:
            return "\033[96m", "!!" if best_move_notation == "!!" else "!"  
        else:
            return "\033[96m", ""  
    
    if 0 < percentage_loss <= 0.5:
        if best_move_notation in ["!!", "!"]:
            return "\033[92m", "!!" if best_move_notation == "!!" else "!"  
        else:
            return "\033[92m", ""         
    elif 0.5 < percentage_loss <= 2.5:
        if best_move_notation == "!!":
            return "\033[92m", "!"  
        else:
            return "\033[92m", ""  
    elif 2.5 < percentage_loss <= 5:
         return "\033[92m", ""  
    elif 5 < percentage_loss <= 10:
        return "\033[93m", "?!"  
    elif 10 < percentage_loss <= 20:
        return "\033[33m", "?"  
    else:
        return "\033[91m", "??"  

    return "\033[0m", ""  

def highlight_move(move, color):
    color_end = "\033[0m"
    return f"{color}{move}{color_end}"

def process_pgn_and_rating(pgn_content, white_elo, black_elo, engine_path, model="gpt-3.5-turbo-instruct", depth=5, prob_threshold=0.001):
    global best_move_notation
    start_time = time.time()
    pgn_content = clean_pgn(pgn_content)
    prompt = pgn_content.strip()

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    game = chess.pgn.read_game(io.StringIO(pgn_content))
    board = game.board()

    rating = white_elo if board.turn == chess.WHITE else black_elo
    high_rating= max(white_elo, black_elo)

    node = game
    while not node.is_end():
        next_node = node.variation(0)
        board.push(next_node.move)
        node = next_node

    legal_moves = get_legal_moves(board)

    n = len(legal_moves)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_analyze_moves = executor.submit(analyze_moves, engine, board, n)
        future_top_sequences = executor.submit(get_top_sequences, prompt, legal_moves, model, depth, 3, prob_threshold)

    top_sequences = future_top_sequences.result()

    all_evals_with_mate = future_analyze_moves.result()


    all_evals = []
    for move, eval_score in all_evals_with_mate:
        if isinstance(eval_score, str) and eval_score.startswith('mate:'):
            mate_value = int(eval_score.split(':')[1])
            if board.turn == chess.WHITE:
                eval_score = 10000 if mate_value > 0 else -10000
            else:
                eval_score = -10000 if mate_value > 0 else 10000
        all_evals.append((move, eval_score))

    move_probabilities = {}
    for seq, logprob in top_sequences:
        probability = math.exp(logprob) * 100
        if seq in move_probabilities:
            move_probabilities[seq] += probability
        else:
            move_probabilities[seq] = probability

    best_move_idx, best_eval_value = find_best_move_index(all_evals, board.turn)

    best_move = all_evals[best_move_idx][0]

    if best_move not in move_probabilities:
        move_probabilities[best_move] = prob_threshold * 100

    total_probability = sum(move_probabilities.values())
    normalized_moves_initial = {move: (prob / total_probability) * 100 for move, prob in move_probabilities.items()}

    win_percentages_1500 = {move: calculate_win_percentage(1500, eval_score) for move, eval_score in all_evals}
    win_percentages_rating = {move: calculate_win_percentage(rating, eval_score) for move, eval_score in all_evals}

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

    total_probability = sum(normalized_moves_initial.values())
    normalized_moves_best_adjusted = {move: (prob / total_probability) * 100 for move, prob in normalized_moves_initial.items()}

    win_percentages_1500 = {move: calculate_win_percentage(1500, eval_score) for move, eval_score in all_evals}
    win_percentages_rating = {move: calculate_win_percentage(rating, eval_score) for move, eval_score in all_evals}

    mate_in_dict = {}
    for move, eval_score in all_evals_with_mate:
        if isinstance(eval_score, str) and eval_score.startswith('mate:'):
            mate_in_dict[move] = abs(int(eval_score.split(':')[1]))

    multiplier = {
        'classical': 150,
        'rapid': 500,
        'blitz': 1000,
        'bullet': 4000
    }.get(game_type, 150) 
    
    percentage_losses = {}
    for move in normalized_moves_best_adjusted:
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

        elo = white_elo if board.turn == chess.WHITE else black_elo

        mate_in = mate_in_dict.get(move)
        if mate_in is not None and mate_in > 0:
            if percentage_losses[move] == 0:
                normalized_moves_best_adjusted[move] *= (1 + (elo / (multiplier * mate_in)))
            elif percentage_losses[move] > 0:
                normalized_moves_best_adjusted[move] /= (1 + (elo / (multiplier * mate_in)))
        if percentage_loss > 0:
            normalized_moves_best_adjusted[move] /= (1 + (percentage_loss / ((-19 * elo) / 600 + 131.67)))


    total_probability_modified = sum(normalized_moves_best_adjusted.values())
    normalized_moves_final = {move: (prob / total_probability_modified) * 100 for move, prob in normalized_moves_best_adjusted.items()}


    sorted_moves = sorted(normalized_moves_final.items(), key=lambda x: x[1], reverse=True)

    valid_moves = []
    final_board = None

    for move, norm_prob in sorted_moves:
        eval_score = next((eval_score for m, eval_score in all_evals if m == move), None)
        if eval_score is None:
            continue
        valid_moves.append((move, norm_prob, eval_score))
        final_board = board 

    valid_total_probability = sum(prob for move, prob, eval_score in valid_moves)
    new_normalized_moves = [(move, prob, (prob / valid_total_probability) * 100, eval_score) for move, prob, eval_score in valid_moves]

    win_percentages = []
    current_win_percentage = 0
    for i, (move, raw_prob, new_norm_prob, eval_score) in enumerate(new_normalized_moves):
        win_percentage = calculate_win_percentage(high_rating, eval_score)
        win_percentages.append(win_percentage)          
        current_win_percentage += win_percentage * new_norm_prob / 100


    best_move_notation = get_best_move_notation(
        calculate_win_percentage(rating, best_eval_value), 
        normalized_moves_final[best_move], 
        board.turn, 
        current_win_percentage
    )

    valid_total_probability = sum(prob for move, prob, _, eval_score in new_normalized_moves)
    new_normalized_moves = [(move, prob, (prob / valid_total_probability) * 100 if valid_total_probability > 0 else 0.0, eval_score) for move, prob, _, eval_score in new_normalized_moves]

    rows_to_print = list(range(min(3, len(new_normalized_moves))))
    best_move_idx_in_normalized = next(i for i, (move, _, _, _) in enumerate(new_normalized_moves) if move == best_move)
    if best_move_idx_in_normalized not in rows_to_print:
        rows_to_print = list(range(min(3, len(new_normalized_moves))))
        rows_to_print.append(best_move_idx_in_normalized)

    table_data = []
    for i in rows_to_print:
        move, raw_prob, new_norm_prob, eval_score = new_normalized_moves[i]
        win_percentage = win_percentages[i]
        percentage_loss = percentage_losses[move] if move in percentage_losses else 0
        is_best_move = (move == best_move)
        best_move_notation = get_best_move_notation(win_percentage, new_norm_prob, board.turn, current_win_percentage)
        color, notation = get_color_and_notation(percentage_loss, is_best_move)
        colored_move = highlight_move(f"{move}{notation}", color)
        row = [f"#{i+1}", colored_move, f"{new_norm_prob:.2f}%" if raw_prob > 0 else "0.00%", f"{win_percentage:.2f}%"]
        table_data.append(row)

    headers = ["#", "Move", "Likelihood", "Evaluation"]
    table = tabulate(table_data, headers=headers, tablefmt="pretty")

    end_time = time.time()
    print(f"Total Time taken: {end_time - start_time:.2f} seconds")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"\n        Current Evaluation: {current_win_percentage:.2f}%")
    print(table)
    

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
        return "bullet"
    elif total_time < 600:
        return "blitz"
    elif total_time < 3600:
        return "rapid"
    else:
        return "classical"

if __name__ == "__main__":
    print('Enter the PGN (type "END" on a new line to finish):')
    pgn_content = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        pgn_content.append(line)

    pgn_content = "\n".join(pgn_content)

    time_control_match = re.search(r'\[TimeControl\s+"([^"]+)"\]', pgn_content)
    if time_control_match:
        time_control = time_control_match.group(1)
        game_type = determine_game_type(time_control)
    else:
        game_type = input("Enter the game type (bullet, blitz, rapid, classical): ").strip().lower()   
    game_type = game_type.lower() 

    white_elo = re.search(r'\[WhiteElo\s+"(\d+)"\]', pgn_content)
    black_elo = re.search(r'\[BlackElo\s+"(\d+)"\]', pgn_content)

    def get_elo(elo, prompt):
        return int(elo.group(1)) if elo else int(input(prompt))

    def adjust_rating(rating, game_type):
        adjustments = {"bullet": 0, "blitz": 200, "rapid": 700, "classical": 1200}
        rating += adjustments.get(game_type, 0)
        rating = max(1000, min(4100, rating))
        return rating

    white_elo = get_elo(white_elo, "Input White Elo: ")
    black_elo = get_elo(black_elo, "Input Black Elo: ")

    white_elo = adjust_rating(white_elo, game_type)
    black_elo = adjust_rating(black_elo, game_type)
  
    process_pgn_and_rating(pgn_content, white_elo, black_elo, engine_path)
