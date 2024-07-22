python main.py \
	--output_dir finetune_medcpt_ppr \
	--train_dataset datasets/ppr_train.jsonl \
	--pmid2info_path datasets/ppr_pmid2info.json \
	--qid2info_path datasets/ppr_qid2info.json \
	--bert_q_path ncbi/MedCPT-Article-Encoder \
	--bert_d_path ncbi/MedCPT-Article-Encoder \
	--tokenizer_path ncbi/MedCPT-Article-Encoder