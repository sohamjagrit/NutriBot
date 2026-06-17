# RAG Pipeline Testing Notebooks

This directory contains a comprehensive suite of Jupyter notebooks for testing and optimizing each stage of the nutrition RAG pipeline.

## Notebook Series

### 00. Data Exploration (`00_data_exploration.ipynb`)
**Purpose:** Understand the nutrition knowledge base

- Load sample nutrition documents
- Compute statistics: word count, character count, average lengths
- Topic keyword frequency analysis
- Save processed data to CSV for downstream notebooks

**Key outputs:**
- `data/processed/nutrition_docs.csv` — Processed document dataset
- Statistics on document diversity and coverage

**Best for:**
- Baseline data understanding
- Identifying data quality issues
- Planning chunking strategies

---

### 01. Embedding Models (`01_embedding_models.ipynb`)
**Purpose:** Compare different embedding models

- Load and test MiniLM embedder (384-dim)
- Load and test BGE embedder (1024-dim)
- Measure performance: load time, embedding time, quality
- Compare results on same test queries

**Test queries:**
- "What are the benefits of protein?"
- "How much water should I drink daily?"
- "Best foods for weight loss?"
- "How to balance macronutrients?"

**Key metrics:**
- Model load time (seconds)
- Document embedding time
- Per-document embedding latency
- Embedding dimension

**Key outputs:**
- Performance comparison table
- Top-k results for each model
- Quality differences on domain-specific queries

**Best for:**
- Choosing which embedder to use
- Understanding embedding quality trade-offs (speed vs. quality)
- Determining if domain-specific embedder (BGE) is worth the extra compute

**Expected findings:**
- MiniLM: Fast, reasonable quality, lightweight
- BGE: Slower, higher quality, better for nutrition domain queries

---

### 02. Retrieval Strategies (`02_retrieval_strategies.ipynb`)
**Purpose:** Compare semantic vs. hybrid search methods

- Initialize hybrid retriever (BM25 + semantic)
- Test retrieval on multiple queries
- Analyze score distribution: BM25 scores vs. semantic scores
- Compare combined scores with different weight combinations

**Test queries:**
- "What are the benefits of protein?"
- "How much water should I drink?"
- "weight loss tips"

**Key metrics:**
- Combined scores (weighted combination)
- Individual BM25 scores
- Individual semantic scores
- Score variance and distribution

**Key outputs:**
- Top-k results with component scores
- Score distribution analysis
- Insight into how BM25 vs. semantic complement each other

**Best for:**
- Tuning hybrid search weights
- Understanding when keyword search (BM25) vs. semantic search matters
- Identifying queries where one method outperforms the other

**Configuration to experiment with:**
- Adjust `bm25_weight` and `semantic_weight` in retriever config
- Test different `similarity_threshold` values

---

### 03. Prompt Engineering (`03_prompt_engineering.ipynb`)
**Purpose:** Test different system prompts and few-shot examples

- Examine three built-in system prompts:
  - `standard`: Professional nutrition advice
  - `conversational`: Friendly, approachable tone
  - `detailed`: Comprehensive with biochemistry
- View few-shot examples used for in-context learning
- Build complete RAG prompts with context

**Key outputs:**
- Side-by-side comparison of system prompts
- Few-shot example demonstrations
- Complete RAG prompt structure

**Best for:**
- Understanding prompt template variations
- Choosing appropriate tone for your use case
- Inspecting how context is injected into prompts

**Experiments to try:**
- Register custom system prompts
- Test few-shot vs. zero-shot performance
- Modify example questions/answers for domain

---

### 04. RAG Pipeline Evaluation (`04_rag_pipeline_eval.ipynb`)
**Purpose:** End-to-end evaluation of the complete RAG system

- Initialize full NutritionRAG pipeline
- Run complete query flow: retrieve + generate
- Evaluate retrieval quality with ground truth
- Evaluate answer quality with multiple dimensions

**Retrieval metrics (with ground truth):**
- MRR (Mean Reciprocal Rank)
- NDCG@5, NDCG@10 (ranking quality)
- Precision@5, Precision@10 (fraction relevant)
- Recall@5, Recall@10 (coverage of relevant docs)

**Answer quality metrics:**
- Answer Relevance: Does answer address the question?
- Faithfulness: Is answer grounded in retrieved context?
- Context Precision: Are retrieved docs relevant to answer?

**Test queries with ground truth:**
- "How much protein do I need daily?" → relevant: doc[0]
- "What are good sources of healthy fats?" → relevant: doc[2]
- "How much water should I drink?" → relevant: doc[4]

**Key outputs:**
- Per-query evaluation results
- Summary statistics across all queries
- Detailed breakdown by metric

**Best for:**
- Measuring end-to-end system performance
- Identifying bottlenecks: retrieval vs. generation
- Comparing different configurations
- Establishing baseline metrics before optimization

**Next steps after running:**
- A/B test different embedders using these metrics
- Try different hybrid search weights
- Experiment with different system prompts
- Measure impact of fine-tuning

---

### 05. Fine-Tuning Preparation (`05_finetuning_prep.ipynb`)
**Purpose:** Prepare data and analysis for model fine-tuning

**Data collection:**
- Create query-document training pairs
- Format for embedding model fine-tuning (JSONL/CSV)
- Hard negative mining: identify docs retrieved high but not relevant

**Analysis:**
- Query statistics: length, difficulty, score variance
- Document frequency and coverage
- Query-document pair statistics

**Hard negative mining:**
- Find documents with high retrieval scores but not marked relevant
- These are ideal for contrastive learning in fine-tuning

**Outputs:**
- `data/finetune/embedding_pairs.jsonl` — Training pairs in JSONL format
- `data/finetune/embedding_pairs.csv` — Same in CSV format
- Statistics on hard negatives

**Fine-tuning recommendations:**
1. **Embedding Model Fine-Tuning (BGE):**
   - Method: Contrastive learning with triplet/in-batch negatives
   - Loss: MultipleNegativesRankingLoss (Sentence Transformers)
   - Data: Query-document pairs with hard negatives
   - Expected improvement: 5-15% on domain-specific queries

2. **LLM Fine-Tuning (Response Quality):**
   - Collect Q&A pairs with highly-rated responses
   - Use successful examples as demonstrations
   - Methods: LoRA, QLoRA, or full fine-tuning

3. **Data Requirements:**
   - Minimum: 100-500 pairs for embeddings
   - Better: 1000+ pairs for significant improvements
   - Essential: Diverse query types and document topics

**Best for:**
- Planning fine-tuning initiatives
- Understanding data quality and balance
- Identifying easy vs. hard queries
- Creating labeled datasets for supervised learning

---

## Running the Notebooks

### Prerequisites
```bash
pip install -r requirements-dev.txt
```

### Start Jupyter
```bash
jupyter notebook
```

### Execution order (recommended)
1. Start with `00_data_exploration.ipynb` to understand your data
2. Run `01_embedding_models.ipynb` to choose embedders
3. Run `02_retrieval_strategies.ipynb` to tune retrieval
4. Run `03_prompt_engineering.ipynb` to refine prompts
5. Run `04_rag_pipeline_eval.ipynb` to measure performance
6. Run `05_finetuning_prep.ipynb` to prepare for optimization

### Quick start
If you just want to see the pipeline work:
```bash
jupyter notebook 04_rag_pipeline_eval.ipynb
```

## Key Experiments to Try

### Embedding Selection
```
Run 01_embedding_models.ipynb:
1. Compare MiniLM vs BGE on your queries
2. Check if BGE's higher dimensionality (1024 vs 384) justifies latency
3. Measure on-device performance for your deployment target
```

### Hybrid Search Tuning
```
Run 02_retrieval_strategies.ipynb:
1. Start with equal weights: bm25_weight=0.5, semantic_weight=0.5
2. Adjust weights based on query type
3. Try: bm25_weight=0.3, semantic_weight=0.7 (emphasize semantic)
4. Try: bm25_weight=0.7, semantic_weight=0.3 (emphasize keyword)
5. Measure MRR@5 for each configuration
```

### Prompt Optimization
```
Run 03_prompt_engineering.ipynb:
1. Test each prompt type on your queries
2. Register a custom prompt with domain-specific instructions
3. Test few-shot with/without examples
4. Measure answer quality in 04_rag_pipeline_eval.ipynb
```

### Evaluation Benchmarking
```
Run 04_rag_pipeline_eval.ipynb:
1. Create a test set of 10-20 queries with ground truth
2. Measure baseline metrics (MRR, NDCG, etc.)
3. Change embedder to BGE → measure improvement
4. Adjust hybrid weights → measure improvement
5. Change prompt type → measure improvement
6. Document all results for decision-making
```

### Fine-Tuning ROI
```
Run 05_finetuning_prep.ipynb:
1. Calculate how many training pairs you have
2. Estimate fine-tuning effort (time/cost/compute)
3. Based on hard negatives, assess potential gains
4. Decide: fine-tune vs. use pre-trained models
```

## Metric Reference

### Retrieval Metrics
- **MRR (Mean Reciprocal Rank):** Position of first relevant doc (1/rank). Higher is better. Range: [0, 1]
- **NDCG@k:** Ranking quality considering position. Ideal: 1.0, Poor: 0.0
- **Precision@k:** Fraction of top-k results that are relevant. Range: [0, 1]
- **Recall@k:** Fraction of all relevant docs found in top-k. Range: [0, 1]

### Answer Quality Metrics
- **Answer Relevance:** Does answer address question? Range: [0, 1]
- **Faithfulness:** Is answer grounded in context? Range: [0, 1]
- **Context Precision:** Fraction of retrieved docs used in answer. Range: [0, 1]
- **RAGAS Score:** Average of all dimensions. Range: [0, 1]

## Troubleshooting

### Out of Memory
- Reduce batch size in embedding notebooks
- Use MiniLM instead of BGE
- Process documents in chunks

### Slow Embedding
- Use MiniLM instead of BGE for development
- Use CPU for quick iteration, GPU for final runs
- Consider batching

### Poor Retrieval Quality
- Check: Are documents relevant to queries?
- Try: Adjust hybrid weights (emphasize semantic)
- Try: Use BGE embedder for better quality
- Consider: Fine-tune embedder on domain data

### Poor Generation Quality
- Try: Different system prompt type
- Try: Add more/better few-shot examples
- Try: Retrieve more documents (increase top_k)
- Consider: Fine-tune LLM on domain data

## Resources

- **Sentence Transformers:** https://www.sbert.net/
- **BGE Embeddings:** https://huggingface.co/BAAI/bge-large-en-v1.5
- **Evaluation Metrics:** See `src/evaluation/evaluator.py`
- **RAG Best Practices:** https://huggingface.co/docs/transformers/tasks/retrieval_augmented_generation
