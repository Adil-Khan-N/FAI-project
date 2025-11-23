import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """Dueling DQN architecture with separate value and advantage streams.
        
        This architecture learns V(s) and A(s,a) separately, then combines them:
        Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        This helps with faster convergence and better generalization.
        """
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        
        # Shared feature layers
        self.features = nn.ModuleList()
        in_size = input_size
        for hs in hidden_sizes[:-1] if len(hidden_sizes) > 1 else hidden_sizes:
            self.features.append(nn.Linear(in_size, hs))
            self.features.append(nn.ReLU())
            self.features.append(nn.Dropout(0.1))  # Small dropout for regularization
            in_size = hs
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(in_size, hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_size, hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, output_size)
        )
        
        # Initialize weights with Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Shared features
        features = x
        for layer in self.features:
            features = layer(features)
        
        # Separate value and advantage computation
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth', map_location=None):
        file_path = os.path.join('./model', file_name) if not os.path.isabs(file_name) else file_name
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        state = torch.load(file_path, map_location=map_location)
        self.load_state_dict(state)

# Backward compatibility alias
Linear_QNet = DuelingDQN


class DoubleDQNTrainer:
    def __init__(self, model, lr, gamma, target_update_freq=100):
        self.lr = lr
        self.gamma = gamma
        self.model = model  # Main network
        self.target_model = type(model)(model.features[0].in_features if hasattr(model, 'features') else 11, 
                                       256, 3)  # Target network
        self.target_model.load_state_dict(model.state_dict())
        self.target_update_freq = target_update_freq
        self.training_step = 0
        
        # Use AdamW optimizer with weight decay for better generalization
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Use Huber loss instead of MSE for more stable training
        self.criterion = nn.SmoothL1Loss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train_step(self, state, action, reward, next_state, done):
        # Convert lists of numpy arrays to single numpy arrays first for efficiency
        if isinstance(state, (list, tuple)):
            try:
                state = np.array(state)
            except Exception:
                state = np.vstack(state)
        if isinstance(next_state, (list, tuple)):
            try:
                next_state = np.array(next_state)
            except Exception:
                next_state = np.vstack(next_state)

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Current Q values from main network
        current_q = self.model(state)
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using main network
            next_actions = self.model(next_state).argmax(dim=1)
            # Evaluate these actions using target network
            next_q_values = self.target_model(next_state)
            max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values
            target_q = reward + self.gamma * max_next_q * (1 - torch.tensor(done, dtype=torch.float))
        
        # Get Q values for taken actions
        action_indices = torch.argmax(action, dim=1)
        predicted_q = current_q.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute loss
        loss = self.criterion(predicted_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()

# Backward compatibility alias
QTrainer = DoubleDQNTrainer



