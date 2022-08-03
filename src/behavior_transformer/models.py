from types import SimpleNamespace
from typing import Optional
import logging

import math
import torch
import torch.nn as nn


class BehaviorEmbeddings(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super().__init__()
        self.obs_shape = input_shape - args.n_actions
        self.act_shape = args.n_actions
        self.embedding_dim = args.token_embed_dim
        self.batch_size = args.batch_size
        self.n_agents = args.n_agents

        self.embed_obs = nn.Linear(self.obs_shape, args.token_embed_dim)
        self.embed_act = nn.Linear(self.act_shape, args.token_embed_dim)
        self.embed_timestep = nn.Embedding(
            args.episode_limit + 1, args.token_embed_dim
        )  # Will have to customize run() to pass in env_info["episode_limit"]
        self.embed_ln = nn.LayerNorm(args.token_embed_dim)

    def split_inputs(self, inputs):
        """
        shape(inputs) = [b, n, e]:
            b = batch size
            n = number of agents
            e = experience dim which is just the observation shape (o) plus action shape (a)

        return observations, actions
        shape(observations) = [b, n, o]:
        shape(actions) = [b. n. a]
        """
        observations = inputs[:, :, : self.obs_shape]
        actions = inputs[:, :, -self.act_shape :]
        logging.debug(f"observations shape = {observations.shape}")
        logging.debug(f"actions shape = {actions.shape}")
        return observations, actions

    def prepare_input_sequences(self, observations, actions, timesteps):
        """
        format sequence of n agent observation action pairs to a sequence
        """
        observations = observations.view(self.batch_size, -1, self.obs_shape)
        actions = actions.view(self.batch_size, -1, self.act_shape)
        timesteps = timesteps.repeat(1, 1, self.n_agents)
        timesteps = timesteps.permute(0, 2, 1).reshape((self.batch_size, -1))
        logging.debug(f"observation sequence shape = {observations.shape}")
        logging.debug(f"action sequence shape = {actions.shape}")
        logging.debug(f"timestep sequence shape = {timesteps.shape}")
        return observations, actions, timesteps

    def prepare_output_sequences(self, observation_embeddings, action_embeddings):
        _, seq_length, _ = observation_embeddings.shape
        stacked_sequence = (
            torch.stack((observation_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(self.batch_size, 2 * seq_length, self.embedding_dim)
        )
        return stacked_sequence.permute(1, 0, 2)

    def forward(self, inputs, timesteps):
        """Tokenize each modality and augment with positional embeddings from
        sequence timesteps.

        NOTE: Need to customize MAC to pass in timesteps

        Args:
            observations (Tensor): A (B, O) shape tensor containing a batch of
                local observations from all agents.
            actions (Tensor): A (B*N, A) shape tensor containing a batch of agent
                onehote actions.
            timesteps (Tensor): A (B*N,) shape tensor containing a batch of
                episode timesteps repeated for all agents.
                E.g. for n = 3 agents, t = [1, 1, 1, 2, 2, 2, ...]

        Returns:
            Tuple[Tensor, Tensor]: The token embeddings for all agent
                observations and actions, both of shape (B*N, E).
        """
        observations, actions = self.split_inputs(inputs)
        observations, actions, timesteps = self.prepare_input_sequences(
            observations, actions, timesteps
        )

        embed_o = self.embed_obs(observations)
        embed_a = self.embed_act(actions)
        embed_t = self.embed_timestep(timesteps)
        logging.debug(f"embed obs shape = {embed_o.shape}")
        logging.debug(f"embed act shape = {embed_a.shape}")
        logging.debug(f"embed time shape = {embed_t.shape}")

        embed_o = embed_o + embed_t
        embed_a = embed_a + embed_t

        return self.prepare_output_sequences(embed_o, embed_a)


class BehaviorAttention(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super().__init__()

        self.query = nn.Linear(args.token_embed_dim, args.token_embed_dim, bias=True)
        self.key = nn.Linear(args.token_embed_dim, args.token_embed_dim, bias=True)
        self.value = nn.Linear(args.token_embed_dim, args.token_embed_dim, bias=True)

        self.softmax = nn.Softmax(dim=1)

        self.output = nn.Linear(args.token_embed_dim, args.token_embed_dim)

        self.dropout = nn.Dropout(0.2)

        self.scale = 1 / math.sqrt(args.token_embed_dim)

        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        Computes the attention scores between each sample
        shape(query, key) = [s, b, d]:
            s = sequence length
            b = batch size
            d = token embedding dim

        ith batch in the sequence
        jth batch in the sequence
        computes scalar value attention scores between each token in a sequence
        shape(scores) = [s, s, b]
        """
        return torch.einsum("ibd, jbd->ijb", query, key)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Query, key, and value tensors are copies of the sequence of tokens and have shape:
        shape(query, key) = [s, b, d]:
            s = sequence length
            b = batch size
            d = token embedding dim
        """
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores)
        attn = self.dropout(attn)

        x = torch.einsum(
            "ijb, jbd->ibd", attn, value
        )  # Calculates outer product: softmax(QK^T) * V

        self.attn = attn.detach()

        return self.output(x)


class FeedForward(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super().__init__()
        self.layer1 = nn.Linear(args.token_embed_dim, args.hidden_ff, bias=True)
        self.layer2 = nn.Linear(args.hidden_ff, args.token_embed_dim, bias=True)

        self.dropout = nn.Dropout(args.dropout)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)

        return self.layer2(x)


class BehaviorEncoder(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super().__init__()
        self.attn = BehaviorAttention(input_shape, args)
        self.ff = FeedForward(input_shape, args)

    def forward(self, z):
        attn = self.attn(query=z, key=z, value=z)
        logging.debug(f"attention shape = {attn.shape}")
        ff = self.ff(attn)
        logging.debug(f"feed forward shape = {ff.shape}")
        return ff


class BehaviorTransformer(nn.Module):
    def __init__(self, input_shape: int, args: SimpleNamespace):
        super(BehaviorTransformer, self).__init__()
        self.args = args

        self.embedding = BehaviorEmbeddings(input_shape, args)
        self.encoder = BehaviorEncoder(input_shape, args)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        z = self.embedding(x, t)
        logging.debug(f"Token embeddings shape = {z.shape}")

        encodings = self.encoder(z)
        logging.debug(f"Encoding shape = {encodings.shape}")

        return encodings


class BehaviorCloningClassifier(nn.Module):
    def __init__(self, input_shape: int, args: SimpleNamespace):
        super(BehaviorCloningClassifier, self).__init__()
        self.embedding_dim = args.token_embed_dim
        self.n_actions = args.n_actions

        self.transformer = BehaviorTransformer(input_shape, args)
        self.bc_head = nn.Linear(self.embedding_dim, self.n_actions)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        encodings = self.transformer(x, t)

        # separate sequence of observations and actions
        encodings = encodings.reshape(batch_size, -1, 2, self.embedding_dim)
        action_encodings = encodings[:, :, 1]

        return self.bc_head(action_encodings)
