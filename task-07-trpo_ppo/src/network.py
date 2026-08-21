import torch
import torch.nn as nn
from torch.distributions import Categorical



class Actor(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
    ):
        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                state_dim,
                hidden_dim
            ),

            nn.Tanh(),


            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),


            nn.Linear(
                hidden_dim,
                action_dim
            )

        )


    def forward(self, state):

        logits = self.net(state)

        return Categorical(
            logits=logits
        )



class Critic(nn.Module):

    def __init__(
        self,
        state_dim,
        hidden_dim=64,
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                state_dim,
                hidden_dim
            ),

            nn.Tanh(),


            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),


            nn.Linear(
                hidden_dim,
                1
            )

        )


    def forward(self,state):

        value = self.net(state)

        return value.squeeze(-1)