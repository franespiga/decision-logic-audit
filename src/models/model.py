from transformers import (
    DecisionTransformerPreTrainedModel,
    DecisionTransformerGPT2Model)

from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput

from .config import DecisionTransformerConfig

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from collections import deque
import numpy as np
import scipy.stats

class DecisionTransformerModel(DecisionTransformerPreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        leverage = self.config.max_leverage if self.config.allow_leveraged_positions else 1
        if self.config.allow_short_positions:
            #Short & Long positions with leverage, weights in (-Inf, Inf)
            self.predict_action = leverage* nn.Sequential(
                *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()]))
            )
        else:
            if self.config.allow_leveraged_positions:
                #Long only with leverage, weights in [0,Inf)
                self.predict_action = leverage* nn.Sequential(
                    *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Sigmoid()]))
                )
            else:
                # Long only without leverage, weights in [0,1]
                self.predict_action = nn.Sequential(
                    nn.Linear(config.hidden_size, config.act_dim),
                    nn.Softmax(dim=2)
                )


        self.predict_return = torch.nn.Linear(self.config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

        self.knowledge = {
            'actions':[],
            'states':[],
            'rtgs':[]
        }
        self.action_tracker = torch.zeros(1, config.act_dim)

    def forward(
            self,
            states: Optional[torch.FloatTensor] = None,
            actions: Optional[torch.FloatTensor] = None,
            rewards: Optional[torch.FloatTensor] = None,
            returns_to_go: Optional[torch.FloatTensor] = None,
            timesteps: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Control for NAN inputs and replace with agent knowledge
        if torch.isnan(states).any():
            state_knowledge = torch.mean(torch.cat(self.knowledge['states']), dim = 0)
            states = torch.where(torch.isnan(states), state_knowledge, states)
        if torch.isnan(actions).any():
            action_knowledge = torch.mean(torch.cat(self.knowledge['actions']), dim = 0)
            actions = torch.where(torch.isnan(actions), action_knowledge, actions)
        if torch.isnan(returns_to_go).any():
            rtg_knowledge = torch.mean(torch.cat(self.knowledge['rtgs']), dim = 0)
            returns_to_go = torch.where(torch.isnan(returns_to_go), rtg_knowledge, returns_to_go)

        self.knowledge['states'].append(states)
        self.knowledge['actions'].append(actions)
        self.knowledge['rtgs'].append(returns_to_go)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        if torch.isnan(stacked_inputs).any():
            # <FE-assumptions>: Replace stacked_inputs with nan values either with 0 or running known mean for imputation.
            # I quite like that as it is based on the agent knowledge
            a = 3
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        if torch.isnan(x).any():
            # <FE-assumptions>: something should be clearly done, but I dont know what.
            pass

    # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        if torch.isnan(action_preds).any():
            # If any action is nan, return the previous state of the agent (portfolio)
            action_preds = self.action_tracker
        else:
            self.action_tracker = action_preds

        if not return_dict:
            return (state_preds, action_preds, return_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class TrainableDT(DecisionTransformerModel):
    def __init__(self, config=None):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_targets) ** 2)

        if loss == 0.0:
            a = 1

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

class DecisionTitan(DecisionTransformerModel):
    def __init__(self, config, memory_size=1000, working_size=10, memory_dim=None, n_heads=None):
        super().__init__(config)
        self.memory_size = memory_size
        self.working_size = working_size
        self.memory_dim = config.hidden_size if memory_dim is None else memory_dim
        self.n_heads = config.n_head if n_heads is None else n_heads

        # Episodic memory: tensor of past states
        self.episodic_memory = torch.zeros(memory_size, config.state_dim)
        self.episodic_ptr = 0
        # Semantic memory: tensor of (state, action) pairs
        self.semantic_memory = torch.zeros(memory_size,config.state_dim + config.act_dim)
        self.semantic_ptr = 0
        # Working memory: tensor of recent (action, reward) pairs
        self.working_memory = torch.zeros(working_size, config.act_dim+1)
        self.working_ptr = 0

        # Attention modules
        self.episodic_attention = nn.MultiheadAttention(config.state_dim, self.n_heads,
                                                        batch_first=True)
        self.semantic_attention = nn.MultiheadAttention(config.state_dim + config.act_dim, self.n_heads, batch_first=True)
        self.working_attention = nn.MultiheadAttention(config.act_dim+1, self.n_heads, batch_first=True)


        # For surprise calculation
        self.reward_history = []
        self.surprise_alpha = 1.0  # Tunable parameter

    def change_device(self, device):
        self.episodic_attention = self.episodic_attention.to(device)
        self.semantic_attention = self.semantic_attention.to(device)
        self.working_attention = self.working_attention.to(device)

        self.episodic_memory = self.episodic_memory.to(device)
        self.semantic_memory = self.semantic_memory.to(device)
        self.working_memory = self.working_memory.to(device)


    def store_episodic(self, state):
        """
        # state: [batch, seq, dim] or [batch, dim]
        state = state.detach().mean(dim=1) if state.dim() == 3 else state.detach()
        n = state.shape[0]
        idx = (self.episodic_ptr + torch.arange(n)) % self.memory_size
        self.episodic_memory[idx] = state.cpu()
        self.episodic_ptr = (self.episodic_ptr + n) % self.memory_size
        """

        # state: [batch, seq, dim] or [batch, dim]
        state = state.mean(dim=1) if state.dim() == 3 else state
        n = state.shape[0]
        idx = (self.episodic_ptr + torch.arange(n)) % self.memory_size
        self.episodic_memory[idx] = state
        self.episodic_ptr = (self.episodic_ptr + n) % self.memory_size



    def retrieve_episodic(self, query):
        # If query is [63, 20, 408], let's first average over the sequence dimension
        if query.dim() == 3:
            query = query.mean(dim=1)  # Now [63, 408]

        batch_size = query.shape[0]
        # Reshape tensors to correct dimensions for attention
        q = query.unsqueeze(1)  # [batch, 1, dim]
        mem = self.episodic_memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, mem, dim]

        # Ensure all inputs are float tensors
        q = q.float()
        mem = mem.float()

        attn_out, _ = self.episodic_attention(q, mem, mem)
        return attn_out.squeeze(1)  # [batch, dim]

    def store_semantic(self, state, action):
        """
        # state, action: [batch, seq, dim] or [batch, dim]
        state = state.detach().mean(dim=1) if state.dim() == 3 else state.detach()
        action = action.detach().mean(dim=1) if action.dim() == 3 else action.detach()
        pair = torch.cat([state, action], dim=-1)
        n = pair.shape[0]
        idx = (self.semantic_ptr + torch.arange(n)) % self.memory_size
        self.semantic_memory[idx] = pair.cpu()
        self.semantic_ptr = (self.semantic_ptr + n) % self.memory_size
        """
        # state, action: [batch, seq, dim] or [batch, dim]
        state = state.mean(dim=1) if state.dim() == 3 else state
        action = action.mean(dim=1) if action.dim() == 3 else action
        pair = torch.cat([state, action], dim=-1)
        n = pair.shape[0]
        idx = (self.semantic_ptr + torch.arange(n)) % self.memory_size
        self.semantic_memory[idx] = pair
        self.semantic_ptr = (self.semantic_ptr + n) % self.memory_size

    def retrieve_semantic(self, query):
        # If query is [63, 20, 408], let's first average over the sequence dimension
        if query.dim() == 3:
            query = query.mean(dim=1)  # Now [63, 408]

        batch_size = query.shape[0]
        # Reshape tensors to correct dimensions for attention
        q = query.unsqueeze(1)  # [batch, 1, dim]
        mem = self.semantic_memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, mem, dim]

        # Ensure all inputs are float tensors
        q = q.float()
        mem = mem.float()

        attn_out, _ = self.semantic_attention(q, mem, mem)
        return attn_out.squeeze(1)  # [batch, 2*dim]

    def store_working(self, action, reward):
        """
                # action: [batch, seq, dim] or [batch, dim], reward: [batch, seq, 1] or [batch, 1]
        action = action.detach().mean(dim=1) if action.dim() == 3 else action.detach()
        reward = reward.detach().mean(dim=1) if reward.dim() == 3 else reward.detach()
        pair = torch.cat([action, reward], dim=-1)
        n = pair.shape[0]
        idx = (self.working_ptr + torch.arange(n)) % self.working_size
        self.working_memory[idx] = pair.cpu()
        self.working_ptr = (self.working_ptr + n) % self.working_size
        """
        # action: [batch, seq, dim] or [batch, dim], reward: [batch, seq, 1] or [batch, 1]
        action = action.mean(dim=1) if action.dim() == 3 else action
        reward = reward.mean(dim=1) if reward.dim() == 3 else reward
        pair = torch.cat([action, reward], dim=-1)
        n = pair.shape[0]
        idx = (self.working_ptr + torch.arange(n)) % self.working_size
        self.working_memory[idx] = pair
        self.working_ptr = (self.working_ptr + n) % self.working_size

    def retrieve_working(self, query):
        # If query is [63, 20, 408], let's first average over the sequence dimension
        if query.dim() == 3:
            query = query.mean(dim=1)  # Now [63, 408]

        batch_size = query.shape[0]
        # Reshape tensors to correct dimensions for attention
        q = query.unsqueeze(1)  # [batch, 1, dim]
        mem = self.working_memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, mem, dim]

        # Ensure all inputs are float tensors
        q = q.float()
        mem = mem.float()

        attn_out, _ = self.working_attention(q, mem, mem)
        return attn_out.squeeze(1)  # [batch, dim+1]

    def forward(self, states=None, actions=None, rewards=None, returns_to_go=None, timesteps=None, attention_mask=None, output_hidden_states=None, output_attentions=None, return_dict=None):
        # Store to memories
        if states is not None:
            self.store_episodic(states)
        if states is not None and actions is not None:
            self.store_semantic(states, actions)
        if actions is not None and rewards is not None:
            self.store_working(actions, rewards)

        # Retrieve contexts from different memory types
        episodic_context = None
        semantic_context = None
        working_context = None

        # Retrieve from episodic memory if states are provided
        if states is not None:
            episodic_context = self.retrieve_episodic(states) ##

        # Retrieve from semantic memory if both states and actions are provided
        if states is not None and actions is not None:
            combined_semantic = torch.cat([states.mean(dim=1), actions.mean(dim=1)], dim=-1)
            semantic_context = self.retrieve_semantic(combined_semantic) ##

        # Retrieve from working memory if both actions and rewards are provided
        if actions is not None and rewards is not None:
            combined_working = torch.cat([actions.mean(dim=1), rewards.mean(dim=1)], dim=-1)
            working_context = self.retrieve_working(combined_working)

        # If contexts are retrieved, you can use them to enhance or modify the input states/actions
        if episodic_context is not None:
            states = states + episodic_context.unsqueeze(1)  # Adjust dimensionality as needed

        if semantic_context is not None:
            # Enhance states and actions with semantic memory
            states = states + semantic_context[:, :states.shape[-1]].unsqueeze(1)
            actions = actions + semantic_context[:, states.shape[-1]:].unsqueeze(1)

        if working_context is not None:
            # Enhance actions with working memory context
            actions = actions + working_context[:, :actions.shape[-1]].unsqueeze(1)
            rewards = rewards + working_context[:, actions.shape[-1]:].unsqueeze(1)

        # Call parent forward
        return super().forward(states=states, actions=actions, rewards=rewards, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)

    def get_reward_skewness(self, new_rewards):
        if isinstance(new_rewards, torch.Tensor):
            new_rewards = new_rewards.detach().cpu().numpy().flatten()
        self.reward_history.extend(new_rewards.tolist())
        if len(self.reward_history) < 3:
            return 0.0
        return float(np.abs(scipy.stats.skew(self.reward_history)))


class TrainableTitan(DecisionTitan):
    def __init__(self, config=None):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # Surprise amplifier based on reward skewness
        rewards = kwargs.get("rewards", None)
        if rewards is not None:
            skewness = self.get_reward_skewness(rewards)
            amplifier = 1 + self.surprise_alpha * skewness
        else:
            amplifier = 1.0
        loss = amplifier * torch.mean((action_preds - action_targets) ** 2)
        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)