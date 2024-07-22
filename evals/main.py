__author__ = 'qiao'

"""
reference:
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=1G6hT73KOzfd
"""

from pathlib import Path
import re
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking import Rerank
import polars as pl

from models import DenseRetriever, CrossEncoder # DIY models
import torch

import argparse
import json
import os
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
					datefmt='%Y-%m-%d %H:%M:%S',
					level=logging.INFO,
					handlers=[LoggingHandler()])

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
			"--name",
			type=str,
			default="scifact",
			help="The evaluation dataset."
	)
	parser.add_argument(
			"--dataset",
			type=str,
			default="scifact",
			help="The evaluation dataset."
	)
	parser.add_argument(
		'--corpus_path',
		type=str,
		help="The corpus path."
	)
	parser.add_argument(
		'--queries_path',
		type=str,
		help="The corpus path."
	)
	parser.add_argument(
		'--qrels_path',
		type=str,
		help="The corpus path."
	)
	parser.add_argument(
		"--max_query_length",
		default=512,
		type=int,
		help="Maximum length of query."
	)
	parser.add_argument(
		"--max_doc_length",
		default=512,
		type=int,
		help="Maximum length of documents."
	)
	parser.add_argument(
			"--query_enc_path", 
			type=str, 
			default="malteos/PubMedNCL",
			help="Path to the query encoder."
	)
	parser.add_argument(
			"--doc_enc_path", 
			type=str,
			default="malteos/PubMedNCL",
			help="Path to the document encoder."
	)
	parser.add_argument(
			"--retriever_tokenizer_path", 
			type=str,
			default="malteos/PubMedNCL",
			help="Path to the retriever tokenizer."
	)
	parser.add_argument(
			"--reranking", action='store_true', 
			help="Whether doing re-ranking."
	)
	parser.add_argument(
			"--cross_enc_path",
			type=str,
			default="malteos/PubMedNCL",
			help="Path to the cross encoder."
	)
	parser.add_argument(
			"--reranker_tokenizer_path",
			type=str,
			default="malteos/PubMedNCL",
			help="Path to the cross encoder tokenizer."
	)
	parser.add_argument(
			"--adapter_paths",
			type=str,
			nargs='+',
			default=None,
			help="The number of top documents to re-rank."
	)
	parser.add_argument(
		"--top_k",
		type=int,
		default="100",
		help="The number of top documents to re-rank.",
	)
	parser.add_argument(
		"--dev",
		action="store_true",
		default=False,
		help="Dev mode.",
	)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info(f'EVAL {args.name}')
	logging.info(f'DEVICE: {device}')

	bi_encoder = DRES(DenseRetriever(args.query_enc_path, args.doc_enc_path, args.adapter_paths, args.retriever_tokenizer_path, device, args.max_query_length, args.max_doc_length), batch_size=32)
	retriever = EvaluateRetrieval(bi_encoder, score_function="dot")

	if args.corpus_path:
		corpus, queries, qrels = GenericDataLoader(
			corpus_file=args.corpus_path, 
			query_file=args.queries_path,
			qrels_file=args.qrels_path
		).load_custom()
	else:
		url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/data/{}.zip".format(args.dataset)
		out_dir = os.path.join(os.getcwd(), "data")
		data_path = util.download_and_unzip(url, out_dir)
		print("Dataset downloaded here: {}".format(data_path))

		data_path = f'data/{args.dataset}'

		corpus, queries, qrels = GenericDataLoader(data_path).load(split="test") # or split = "train" or "dev

	if args.dev:
		corpus = dict(list(corpus.items())[:4])
		qrels = dict(list(qrels.items())[:4])
		queries = {qid: queries[qid] for qid in qrels}

	results = retriever.retrieve(corpus, queries)
	metrics = EvaluateRetrieval.evaluate(qrels, results, retriever.k_values)
	mrr = EvaluateRetrieval.evaluate_custom(qrels, results, retriever.k_values + [len(corpus)], metric='mrr')
	output = {'retrieval': list(metrics)}
	output['retrieval'].append(mrr)
	if step := re.search(r'checkpoint-(\d+)', args.query_enc_path):
		step = step[1]
	else:
		step = None
	all_metrics = {'name': args.name, 'task': 'retrieval', 'step': step}
	for metric in output['retrieval']:
		all_metrics |= metric
	df = pl.DataFrame(all_metrics)

	if args.reranking:
		cross_encoder = CrossEncoder(args.cross_enc_path, args.reranker_tokenizer_path, device, args.max_doc_length)
		reranker = Rerank(cross_encoder, batch_size=32)
		rerank_results = reranker.rerank(corpus, queries, results, top_k=args.top_k)
		output['reranking'] = list(EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values))
		output['reranking'].append(EvaluateRetrieval.evaluate_custom(qrels, results, retriever.k_values + [len(corpus)], metric='mrr'))
		if step := re.search(r'checkpoint-(\d+)', args.cross_enc_path):
			step = step[1]
		else:
			step = None
		reranking_metrics = {'name': args.name, 'task': 'reranker', 'step': step}
		for metric in output['reranking']:
			reranking_metrics |= metric
		df = pl.concat([df, pl.DataFrame(reranking_metrics)])
	
	with open(f'results/{args.name}_results.json', 'w') as f:
		json.dump(output, f, indent=4)

	out_path = Path('results/results.csv')
	out_path_exists = out_path.exists()
	with open(out_path, 'a') as f:
		if step := re.search(r'checkpoint-(\d+)', args.query_enc_path):
			step = step[1]
		else:
			step = None
		df.write_csv(f, include_header=not out_path_exists)

	out_path = Path('results/summary_results.csv')
	out_path_exists = out_path.exists()
	with open(out_path, 'a') as f:
		df.select(['name', 'task', 'step', f'MRR@{len(corpus)}', 'P@10', 'NDCG@10', 'Recall@1000']).write_csv(f, include_header=not out_path_exists)
