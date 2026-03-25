# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from typing import List, Union
from vocab import AAS, SCORE_DIM, ESM2_VOCAB, ESM2_PATH
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM


def one_hot_to_string(one_hot: Union[List[List[int]], np.ndarray], alphabet: str) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.
    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height, score_dim=SCORE_DIM):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv1d(20, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv1d(128, 80, kernel_size=3, padding=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv1d(128, 20, kernel_size=3, padding=1)
        self.val_fc1 = nn.Linear(board_width*board_height, 128)
        self.val_fc2 = nn.Linear(128, score_dim)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        # x_act = F.relu(self.act_conv1(x))
        # x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        # x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        # FIXME 20240627: the output is not in (-1, 1)
        # x_val = F.tanh(self.val_fc2(x_val))
        x_val = self.val_fc2(x_val)
        return x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_file=None, use_gpu=True, score_dim=SCORE_DIM):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.score_dim = score_dim

        self.device = torch.device("cuda:0")

        # the policy value net module
        if self.use_gpu:
            self.value_net = Net(board_width, board_height, self.score_dim).to(self.device)
        else:
            self.value_net = Net(board_width, board_height, self.score_dim)
        self.optimizer = optim.Adam(self.value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.value_net.load_state_dict(net_params)
        # use plm model for policy net
        # model_path = "./esm2_models/to_dong/checkpoints/E4B/below0/checkpoint-11823"  # our finetuned model
        self.plm_tokenizer = AutoTokenizer.from_pretrained(ESM2_PATH)
        self.plm_model = EsmForMaskedLM.from_pretrained(ESM2_PATH).to(self.device)
        self.plm_dict = {}

    def value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            # state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).to(self.device))
            value = self.value_net(state_batch)
            return value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)))
            value = self.value_net(state_batch)
            return value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        # print("legal positions {}".format(len(board.availables)))
        legal_positions = board.availables
        current_state_0 = np.expand_dims(board.current_state(), axis=0)
        current_state = np.ascontiguousarray(current_state_0)
        # plm_logits = plm_decode(plm_encode(one_hot_to_string(board._state, AAS), model=self.plm_model), model=self.plm_model)
        # plm_probs = np.exp(torch.log_softmax(torch.from_numpy(plm_logits.flatten()), dim=-1)).numpy()
        cur_seq = one_hot_to_string(board._state, AAS)
        if cur_seq not in self.plm_dict:
            plm_logits = self.plm_model(**self.plm_tokenizer([cur_seq], return_tensors="pt").to(self.device)).logits[0, 1:-1]
            valid_idx = [idx for idx, tok in enumerate(ESM2_VOCAB)]
            plm_logits = plm_logits[:, valid_idx]
            evo_idx = [ESM2_VOCAB.index(aa) for aa in AAS]
            evo_logits = plm_logits[:, evo_idx]
            # for finetune, finetune+dpo: sharp the possibility
            plm_probs = np.exp(torch.log_softmax(evo_logits.detach().cpu(), dim=-1)).flatten().numpy()
            # for pretrain, smooth the possibility
            # plm_probs = np.exp(torch.log_softmax(evo_logits.flatten().detach(), dim=-1)).numpy()
            # print(sum(plm_probs), max(plm_probs))
            self.plm_dict[cur_seq] = plm_probs
        else:
            plm_probs = self.plm_dict[cur_seq]

        if self.use_gpu:
            value = self.value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = plm_probs
        else:
            value = self.value_net(Variable(torch.from_numpy(current_state)).float())
            act_probs = plm_probs
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0].cpu().numpy()
        return act_probs, value

    def train_step(self, state_batch, reward_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).to(self.device))
            # mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)).to(self.device))
            reward_batch = Variable(torch.FloatTensor(np.array(reward_batch)).to(self.device))
        else:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)))
            # mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)))
            reward_batch = Variable(torch.FloatTensor(np.array(reward_batch)))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        value = self.value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1, self.score_dim), reward_batch, reduction='sum')
        loss = value_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        # entropy = -torch.mean(torch.sum(torch.log(mcts_probs + 1E-3) * mcts_probs, 1))
        return loss.item()  # , entropy.item()

    def get_value_param(self):
        net_params = self.value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_value_param()  # get model params
        torch.save(net_params, model_file)
