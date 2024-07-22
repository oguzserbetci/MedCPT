__author__ = "qiao"

"""
The BioCPT model class
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

import adapters
def set_requires_grad(m, requires_grad):
	for param in m.parameters():
		param.requires_grad_(requires_grad)


class Biencoder(nn.Module):
	def __init__(self, args):
		super(Biencoder, self).__init__()
		self.args = args

		q_path = args.bert_q_path
		d_path = args.bert_d_path

		self.config_q = AutoConfig.from_pretrained(q_path)
		self.bert_q = AutoModel.from_pretrained(q_path)

		if q_path == d_path:
			self.config_d = self.config_q
			self.bert_d = self.bert_q
		else:
			self.config_d = AutoConfig.from_pretrained(d_path)
			self.bert_d = AutoModel.from_pretrained(d_path)

		if self.args.detached:
			set_requires_grad(self.bert_q, False)
			set_requires_grad(self.bert_d, False)
			in_features = self.bert_d.config.hidden_size
			self.ff = nn.Sequential(
				nn.LayerNorm(in_features),
				nn.Linear(in_features, in_features * 2),
				nn.GELU(),
				nn.Linear(in_features * 2, in_features),
			)
		elif self.args.adapter:
			adapters.init(self.bert_q)
			self.bert_q.add_adapter(self.args.adapter, config="seq_bn")
			self.bert_q.set_active_adapters(self.args.adapter)
			if q_path != d_path:
				adapters.init(self.bert_d)
				self.bert_d.add_adapter(self.args.adapter, config="seq_bn")
				self.bert_d.set_active_adapters(self.args.adapter)
		elif self.args.adapter_paths:
			adapters.init(self.bert_q)
			loaded_adapters = []
			for adapter in self.args.adapter_paths:
				a = self.bert_q.load_adapter(adapter, with_head=False)
				loaded_adapters.append(a)
			if self.args.fuse_adapter:
				self.adapter_setup = adapters.Fuse(*loaded_adapters)
				self.bert_q.add_adapter_fusion(self.adapter_setup, self.args.fuse_adapter)
				self.bert_q.train_adapter_fusion(self.adapter_setup)
			

	def save_pretrained(self, path):
		if self.args.adapter:
			self.bert_q.save_adapter(os.path.join(path, f'query_adapter'), self.args.adapter)
		elif self.args.adapter_paths:
			self.bert_q.save_all_adapters(os.path.join(path, f'query_adapter'))
			if self.args.fuse_adapter:
				self.bert_q.save_adapter_fusion(os.path.join(path, f'query_adapter'), self.adapter_setup)
		else:
			self.config_q.save_pretrained(os.path.join(path, "query_encoder"))
			self.bert_q.save_pretrained(os.path.join(path, "query_encoder"))

		if self.bert_q != self.bert_d:
			if self.args.adapter:
				self.bert_d.save_adapter(os.path.join(path, f'doc_adapter'), self.args.adapter)
			elif self.args.adapter_paths:
				self.bert_d.save_all_adapters(os.path.join(path, f'doc_adapter'))
				if self.args.fuse_adapter:
					self.bert_d.save_adapter_fusion(os.path.join(path, f'doc_adapter'), self.adapter_setup)
			else:
				self.config_d.save_pretrained(os.path.join(path, "doc_encoder"))
				self.bert_d.save_pretrained(os.path.join(path, "doc_encoder"))


	def forward(
		self,
		q_input_ids,
		q_token_type_ids,
		q_attention_mask,
		d_input_ids,
		d_token_type_ids,
		d_attention_mask,
		weights,
	):

		if q_token_type_ids is not None:
			embed_q = self.bert_q(
				input_ids=q_input_ids,
				attention_mask=q_attention_mask,
				token_type_ids=q_token_type_ids,
			).last_hidden_state[
				:, 0, :
			]  # B x D
		else:
			embed_q = self.bert_q(
				input_ids=q_input_ids,
				attention_mask=q_attention_mask,
			).last_hidden_state[
				:, 0, :
			]  # B x D

		if d_token_type_ids is not None:
			embed_d = self.bert_d(
				input_ids=d_input_ids,
				attention_mask=d_attention_mask,
				token_type_ids=d_token_type_ids,
			).last_hidden_state[
				:, 0, :
			]  # B x D
		else:
			embed_d = self.bert_d(
				input_ids=d_input_ids,
				attention_mask=d_attention_mask,
			).last_hidden_state[
				:, 0, :
			]  # B x D

		if self.args.detached:
			embed_q = embed_q.detach()
			embed_d = embed_d.detach()
			embed_q = embed_q + self.ff(embed_q)
			embed_d = embed_d + self.ff(embed_d)

		B = embed_q.size(dim=0)
		qd_scores = torch.matmul(embed_q, torch.transpose(embed_d, 1, 0)) # B x B

		# q to d softmax
		q2d_softmax = F.log_softmax(qd_scores, dim=1)

		# d to q softmax
		d2q_softmax = F.log_softmax(qd_scores, dim=0)
		
		# positive indices (diagonal)
		pos_inds = torch.tensor(list(range(B)), dtype=torch.long).to(self.args.device)
		
		q2d_loss = F.nll_loss(q2d_softmax,
			pos_inds,
			weight=weights,
			reduction="mean"
		)

		d2q_loss = F.nll_loss(d2q_softmax,
			pos_inds,
			weight=weights,
			reduction="mean"
		)

		loss = self.args.alpha * q2d_loss + (1 - self.args.alpha) * d2q_loss

		return loss
