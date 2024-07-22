python main.py \
	--output_dir ff_medcpt_ppar \
	--train_dataset datasets/ppar_train.jsonl \
	--pmid2info_path datasets/ppar_pmid2info.json \
	--qid2info_path datasets/ppar_qid2info.json \
	--bert_q_path ncbi/MedCPT-Article-Encoder \
	--bert_d_path ncbi/MedCPT-Article-Encoder \
	--tokenizer_path ncbi/MedCPT-Article-Encoder \
	--detached