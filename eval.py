import argparse
import os
import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet


def get_state_from_game(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),

        dir_l,
        dir_r,
        dir_u,
        dir_d,

        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y,

        len(game.snake),
        abs(game.food.x - game.head.x) / game.w,
        abs(game.food.y - game.head.y) / game.h,
    ]

    # normalize length similar to agent
    state[10] = (state[10] - 3) / 20
    return np.array(state, dtype=np.float32)


def get_state_v11(game):
    # older 11-feature state used in earlier checkpoints
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

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
        game.food.y > game.head.y  # food down
    ]

    return np.array(state, dtype=np.float32)


def load_model(path, input_size, hidden, output_size=3, map_location='cpu'):
    # Resolve actual file path (absolute or inside ./model)
    if os.path.isabs(path) and os.path.exists(path):
        file_path = path
    elif os.path.exists(path):
        file_path = path
    else:
        candidate = os.path.join('model', path)
        if os.path.exists(candidate):
            file_path = candidate
        else:
            candidate2 = os.path.join('model', os.path.basename(path))
            if os.path.exists(candidate2):
                file_path = candidate2
            else:
                raise FileNotFoundError(f'Cannot find model file: {path}')

    # Load the state_dict to inspect keys and support legacy formats
    # Older checkpoints may be full pickled objects; allow non-weights-only loading (trusted local file)
    try:
        state = torch.load(file_path, map_location=map_location)
    except Exception:
        # torch.load in newer PyTorch versions may require explicit weights_only flag
        # This is safe for local, trusted checkpoints.
        state = torch.load(file_path, map_location=map_location, weights_only=False)

    # If this looks like the older sequential net (keys start with 'net.'), reconstruct legacy model
    if any(k.startswith('net.') for k in state.keys()):
        # infer linear layer sizes from net.* weights
        net_weight_keys = [k for k in state.keys() if k.startswith('net.') and k.endswith('.weight')]
        indices = sorted(int(k.split('.')[1]) for k in net_weight_keys)
        # For each weight key, get its shape (out_dim, in_dim)
        dims = []
        for idx in indices:
            w = state.get(f'net.{idx}.weight')
            if w is None:
                raise RuntimeError('Unexpected legacy model state dict structure')
            dims.append((w.shape[1], w.shape[0]))  # (in_dim, out_dim)

        # dims is list of (in,out) for each linear; chain them
        # Build sequential: Linear(in0,out0), ReLU, Linear(out0,out1), ReLU, ..., Linear(outN-1,outN)
        layers = []
        for i, (in_dim, out_dim) in enumerate(dims):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            # add activation except after last linear
            if i < len(dims) - 1:
                layers.append(torch.nn.ReLU())

        class LegacyNet(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.net = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

            def save(self, file_name='model.pth'):
                model_folder_path = './model'
                if not os.path.exists(model_folder_path):
                    os.makedirs(model_folder_path)
                file_name = os.path.join(model_folder_path, file_name)
                torch.save(self.state_dict(), file_name)

        model = LegacyNet(layers)
        model.load_state_dict(state)
        model.eval()
        return model

    # Otherwise assume the saved state matches the current DuelingDQN keys
    model = Linear_QNet(input_size, hidden, output_size)
    # state may be a state_dict directly
    if isinstance(state, dict) and not any(k.startswith('module.') for k in state.keys()):
        model.load_state_dict(state)
    else:
        # Some checkpoints wrap the state_dict under a key like 'state_dict' or 'model'
        candidate_keys = ['state_dict', 'model_state', 'model']
        found = False
        for k in candidate_keys:
            if k in state and isinstance(state[k], dict):
                model.load_state_dict(state[k])
                found = True
                break
        if not found:
            # fallback: try loading full object (may be saved as entire model)
            try:
                model = state
            except Exception:
                raise RuntimeError('Unable to load model state dict: unknown format')

    model.eval()
    return model


def run_eval(model_path, episodes=10, render=False, hidden=[256], speed=40):
    # Load model (we'll detect required input size from the model itself)
    # Use input_size placeholder; load_model will reconstruct legacy nets if necessary
    placeholder_input = 14
    model = load_model(model_path, placeholder_input, hidden)

    # determine expected input size by inspecting the first Linear layer
    expected_input = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            expected_input = m.in_features
            break
    if expected_input is None:
        expected_input = 14

    if expected_input == 11:
        state_fn = get_state_v11
    else:
        state_fn = get_state_from_game

    scores = []
    for ep in range(episodes):
        env = SnakeGameAI(render=render, speed=speed)
        env.reset()
        done = False
        while not done:
            state = state_fn(env)
            state_t = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                pred = model(state_t)
                action_idx = int(torch.argmax(pred).item())
            action = [0, 0, 0]
            action[action_idx] = 1
            _, done, score = env.play_step(action)
        scores.append(score)
        print(f'Episode {ep+1}: score={score}')
    print('---')
    print(f'Average score over {episodes} episodes: {np.mean(scores):.3f} ± {np.std(scores):.3f}')
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/model_final.pth', help='Path to model file (relative or absolute)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render the game during evaluation')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256], help='Hidden layer sizes used by the saved model')
    parser.add_argument('--speed', type=int, default=40, help='Render speed (FPS)')
    args = parser.parse_args()

    if not os.path.exists(args.model) and not os.path.exists(os.path.join('model', args.model)):
        print('Model file not found:', args.model)
        print('Available files in ./model:')
        if os.path.exists('model'):
            for f in os.listdir('model'):
                print('  ', f)
        raise SystemExit(1)

    run_eval(args.model, episodes=args.episodes, render=args.render, hidden=args.hidden, speed=args.speed)
