# python main.py \
#   --name medcpt_ppr \
#   --corpus_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/corpus.jsonl \
#   --queries_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/queries/dev_queries.jsonl \
#   --qrels_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/qrels_dev.tsv \
#   --query_enc_path ncbi/MedCPT-Query-Encoder \
#   --doc_enc_path ncbi/MedCPT-Article-Encoder \
#   --retriever_tokenizer_path ncbi/MedCPT-Article-Encoder
# python main.py \
#   --name finetune_medcpt_ppr \
#   --corpus_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/corpus.jsonl \
#   --queries_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/queries/dev_queries.jsonl \
#   --qrels_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/qrels_dev.tsv \
#   --query_enc_path ../retriever/finetune_medcpt/checkpoint-165000/query_encoder \
#   --doc_enc_path ../retriever/finetune_medcpt/checkpoint-165000/doc_encoder \
#   --retriever_tokenizer_path ncbi/MedCPT-Article-Encoder
# python main.py \
#   --name finetune_medcpt_ppr \
#   --corpus_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/corpus.jsonl \
#   --queries_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/queries/dev_queries.jsonl \
#   --qrels_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/qrels_dev.tsv \
#   --query_enc_path ../retriever/finetune_medcpt_ppr/query_encoder \
#   --doc_enc_path ../retriever/finetune_medcpt_ppr/doc_encoder \
#   --retriever_tokenizer_path ncbi/MedCPT-Article-Encoder
python main.py \
  --name adapter_bge-m3_ppr \
  --corpus_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/corpus.jsonl \
  --queries_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/queries/dev_queries.jsonl \
  --qrels_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/qrels_dev.tsv \
  --query_enc_path ../retriever/results/adapter_medcpt_ppr/checkpoint-2500/query_encoder \
  --doc_enc_path ../retriever/results/adapter_medcpt_ppr/checkpoint-2500/query_encoder \
  --max_query_length 1024 \
  --max_doc_length 1024 \
  --adapter_paths ../retriever/results/adapter_medcpt_ppr/checkpoint-2500/adapter_ppr \
  --retriever_tokenizer_path BAAI/bge-m3
# python main.py \
#   --name adapter_medcpt_ppr \
#   --corpus_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/corpus.jsonl \
#   --queries_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/queries/dev_queries.jsonl \
#   --qrels_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/qrels_dev.tsv \
#   --query_enc_path ../retriever/results/adapter_medcpt_correct_ppr/checkpoint-2500/query_encoder \
#   --doc_enc_path ../retriever/results/adapter_medcpt_correct_ppr/checkpoint-2500/query_encoder \
#   --max_query_length 512 \
#   --max_doc_length 512 \
#   --adapter_paths ../retriever/results/adapter_medcpt_correct_ppr/checkpoint-2500/adapter_ppr \
#   --retriever_tokenizer_path ncbi/MedCPT-Article-Encoder
python main.py \
  --name finetune_medcpt_ppr \
  --corpus_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/corpus.jsonl \
  --queries_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/queries/dev_queries.jsonl \
  --qrels_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/qrels_dev.tsv \
  --query_enc_path ../retriever/results/finetune_medcpt_ppr/checkpoint-2500/query_encoder \
  --doc_enc_path ../retriever/results/finetune_medcpt_ppr/checkpoint-2500/query_encoder \
  --max_query_length 512 \
  --max_doc_length 512 \
  --retriever_tokenizer_path ncbi/MedCPT-Article-Encoder
# python main.py \
#   --name medcpt_ppr \
#   --corpus_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/corpus.jsonl \
#   --queries_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/queries/dev_queries.jsonl \
#   --qrels_path /vol/tmp/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/qrels_dev.tsv \
#   --query_enc_path ncbi/MedCPT-Article-Encoder \
#   --doc_enc_path ncbi/MedCPT-Article-Encoder \
#   --max_query_length 512 \
#   --max_doc_length 512 \
#   --retriever_tokenizer_path ncbi/MedCPT-Article-Encoder