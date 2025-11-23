import argparse
import os
import random
import numpy as np
import torch
import signal
import sys
import glob
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done, td_error=None):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        # Set priority based on TD error or max priority for new experiences
        self.priorities[self.pos] = max_prio if td_error is None else abs(td_error) + 1e-6
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # default hidden sizes (can be changed by CLI)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    def save_checkpoint(self, filename=None):
        """Save current model state"""
        if filename is None:
            filename = f'model_interrupted_game_{self.n_games}.pth'
        self.model.save(filename)
        print(f'Model saved as {filename}')
        
    def load_latest_checkpoint(self, pattern='model_checkpoint_game_*.pth'):
        """Load the latest checkpoint matching pattern"""
        model_dir = './model'
        if not os.path.exists(model_dir):
            return None
            
        files = glob.glob(os.path.join(model_dir, pattern))
        if not files:
            return None
            
        # Sort by game number (extract number from filename)
        def extract_game_num(filename):
            try:
                return int(filename.split('_game_')[1].split('.pth')[0])
            except:
                return 0
                
        latest_file = max(files, key=extract_game_num)
        try:
            self.model.load(os.path.basename(latest_file))
            game_num = extract_game_num(latest_file)
            self.n_games = game_num
            print(f'Loaded latest checkpoint: {latest_file} (game {game_num})')
            return latest_file
        except Exception as e:
            print(f'Failed to load {latest_file}: {e}')
            return None
        
    def save_checkpoint(self, filename=None):
        """Save current model state"""
        if filename is None:
            filename = f'model_interrupted_game_{self.n_games}.pth'
        self.model.save(filename)
        print(f'Model saved as {filename}')
        
    def load_latest_checkpoint(self, pattern='model_checkpoint_game_*.pth'):
        """Load the latest checkpoint matching pattern"""
        model_dir = './model'
        if not os.path.exists(model_dir):
            return None
            
        files = glob.glob(os.path.join(model_dir, pattern))
        if not files:
            return None
            
        # Sort by game number (extract number from filename)
        def extract_game_num(filename):
            try:
                return int(filename.split('_game_')[1].split('.pth')[0])
            except:
                return 0
                
        latest_file = max(files, key=extract_game_num)
        try:
            self.model.load(os.path.basename(latest_file))
            game_num = extract_game_num(latest_file)
            self.n_games = game_num
            print(f'Loaded latest checkpoint: {latest_file} (game {game_num})')
            return latest_file
        except Exception as e:
            print(f'Failed to load {latest_file}: {e}')
            return None

    def get_state(self, game):
        """Enhanced state representation with more informative features."""
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Enhanced state with additional features
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            
            # Additional features for better learning
            len(game.snake),  # snake length (normalized)
            abs(game.food.x - game.head.x) / game.w,  # normalized distance to food x
            abs(game.food.y - game.head.y) / game.h,  # normalized distance to food y
        ]
        
        # Normalize snake length
        state[10] = (state[10] - 3) / 20  # Assuming max reasonable length ~23

        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in prioritized replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_long_memory(self):
        """Train using prioritized experience replay."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample from prioritized replay buffer
        batch, indices, weights = self.memory.sample(BATCH_SIZE, beta=0.4 + (self.n_games * 0.01))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.float)
        weights = torch.tensor(weights, dtype=torch.float)
        
        # Compute current Q values
        current_q = self.model(states)
        action_indices = torch.argmax(actions, dim=1)
        predicted_q = current_q.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using Double DQN
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1)
            next_q_values = self.trainer.target_model(next_states)
            max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute TD errors for priority update
        td_errors = (predicted_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Weighted loss
        loss = (weights * (predicted_q - target_q).pow(2)).mean()
        
        # Optimize
        self.trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.trainer.optimizer.step()
        
        self.total_training_steps += 1

    def train_short_memory(self, state, action, reward, next_state, done):
        """Quick single-step training."""
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        return loss

    def update_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_action(self, state):
        """Epsilon-greedy action selection with improved exploration."""
        final_move = [0, 0, 0]
        
        if random.random() < self.epsilon:
            # Random exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Greedy exploitation
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def get_action_eval(self, state):
        """Deterministic action selection for evaluation (no epsilon)."""
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0,0,0]
        final_move[move] = 1
        return final_move


# Global agent reference for signal handler
agent_global = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by saving model"""
    print('\nInterrupted! Saving model before exit...')
    if agent_global:
        agent_global.save_checkpoint()
    sys.exit(0)

def train():
    global agent_global
    
    parser = argparse.ArgumentParser(description='Train or evaluate Snake RL agent')
    parser.add_argument('--mode', choices=['train','eval','human'], default='train')
    parser.add_argument('--render', action='store_true', help='Render game windows during training/eval')
    parser.add_argument('--load', type=str, default=None, help='Load model file from model/ path (or "latest" for latest checkpoint)')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256], help='Hidden layer sizes')
    parser.add_argument('--save-every', type=int, default=50, help='Save model every N games')
    parser.add_argument('--max-games', type=int, default=0, help='Stop after N games (0 = infinite)')
    parser.add_argument('--speed', type=int, default=40, help='Game speed (frames per second)')
    parser.add_argument('--fast', action='store_true', help='Fast training (disable render)')
    parser.add_argument('--best-at-10-threshold', type=int, default=3, help='If best score >= this at 10 games, save a special checkpoint')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent(lr=LR, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
    agent_global = agent  # Set global reference for signal handler
    
    # Update model architecture to match new input size
    if args.hidden != [256]:  # If custom hidden sizes specified
        agent.model = DuelingDQN(14, args.hidden, 3)
        agent.trainer = DoubleDQNTrainer(agent.model, lr=LR, gamma=agent.gamma)

    # Handle model loading
    if args.resume:
        # Resume from latest checkpoint
        loaded = agent.load_latest_checkpoint()
        if not loaded:
            print('No checkpoint found to resume from, starting fresh')
    elif args.load:
        if args.load == 'latest':
            # Load latest checkpoint
            loaded = agent.load_latest_checkpoint()
            if not loaded:
                print('No checkpoint found, starting fresh')
        elif args.load == 'best':
            # Load best model
            loaded = agent.load_latest_checkpoint('model_best_game_*.pth')
            if not loaded:
                print('No best model found, starting fresh')
        else:
            # Load specific file
            try:
                agent.model.load(args.load, map_location='cpu')
                print(f'Loaded model from {args.load}')
            except Exception as e:
                print('Could not load model:', e)

    if args.mode == 'human':
        # run human playable game
        from snake_game_human import SnakeGame
        game_h = SnakeGame()
        while True:
            game_over, score = game_h.play_step()
            if game_over:
                break
        print('Final Score', score)
        return

    render_flag = args.render and not args.fast
    game = SnakeGameAI(render=render_flag, speed=args.speed)

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.update_epsilon()  # Decay exploration
            agent.train_long_memory()

            if score > record:
                record = score
                # always save the new record
                agent.model.save(file_name=f'dueling_dqn_best_game_{agent.n_games}.pth')

            # special checkpoint: if by 10 games we already have a good model
            if agent.n_games == 10 and record >= args.best_at_10_threshold:
                agent.model.save(file_name=f'dueling_dqn_best_at_10_score_{record}.pth')
                print(f'Checkpoint: saved best_at_10 (score={record})')    
                # quick evaluation of the current model (deterministic actions)
                try:
                    from game import SnakeGameAI as EvalGame
                    eval_games = 5
                    eval_scores = []
                    eval_env = EvalGame(render=False, speed=args.speed)
                    for _ in range(eval_games):
                        # run one episode
                        eval_env.reset()
                        done_eval = False
                        while not done_eval:
                            s = agent.get_state(eval_env)
                            a = agent.get_action_eval(s)
                            _, done_eval, sc = eval_env.play_step(a)
                        eval_scores.append(sc)
                    avg_eval = sum(eval_scores)/len(eval_scores)
                    print(f'Evaluation after 10 games: avg score over {eval_games} runs = {avg_eval}')
                except Exception as e:
                    print('Evaluation failed:', e)

            # periodic save
            if args.save_every>0 and agent.n_games % args.save_every == 0:
                agent.model.save(file_name=f'dueling_dqn_checkpoint_game_{agent.n_games}.pth')

            # Enhanced logging with training statistics
            print(f'Game {agent.n_games}, Score {score}, Record: {record}, '
                  f'Epsilon: {agent.epsilon:.3f}, Memory: {len(agent.memory)}')            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            if args.max_games and agent.n_games >= args.max_games:
                print('Reached max games limit; exiting.')
                break

    # final model save
    agent.save_checkpoint('model_final.pth')
    print(f'Training completed. Final model saved after {agent.n_games} games.')


if __name__ == '__main__':
    train()