# Retrieval Strategy for Relation-Conditioned Issue-Link Prediction

## 1. What is implemented and what is proposed

The current code follows the general **retrieve-then-rerank** design used in multi-stage information retrieval, but it does not implement CoRT, CUR matrix factorization, BM25, a dense encoder, or approximate nearest-neighbour search.

The implemented components are:

1. A deterministic temporal lexical retriever in `src/ilr_post_training/data/sft.py`.
2. Frozen candidate pools exported to the SFT JSONL datasets.
3. A pointwise LLM that judges one `(head, relation, tail)` candidate.
4. A set-retrieval LLM that jointly selects tail identifiers from a supplied pool.

The SFT trainers do not search the issue collection. They read the already-created candidate pools. Therefore, a paper describing the current `v1_full` experiments should call the first stage a **temporal IDF-overlap retriever**, not BM25, CoRT, dense retrieval, or hybrid retrieval.

The final v2 study should add and compare BM25, dense, and hybrid first-stage retrieval. That is a proposed extension and must not be described as implemented until its index-building and retrieval code has been completed and evaluated.

## 2. Why candidate retrieval is necessary

For a new issue `s`, relation `r`, and collection of earlier issues, exhaustive LLM scoring would require evaluating every eligible triple:

\[
\mathcal H(s,r)=\{(s,r,o):o\in\mathcal E(s)\},
\]

where the temporally eligible collection is:

\[
\mathcal E(s)=\{o\in\mathcal E:o\neq s\land t_o<t_s\}.
\]

This collection can contain tens or hundreds of thousands of issues. A generative LLM cannot receive the complete collection in one context, and independent LLM scoring of every tail would be expensive. The system therefore uses a fast first-stage method to produce a smaller pool:

\[
C_M(s,r)=\operatorname{TopM}_{o\in\mathcal E(s)}g(s,r,o),
\]

where `g` is the first-stage retrieval score and `M` is the candidate-pool size.

The LLM operates only on `C_M(s,r)`. Consequently, its end-to-end recall is upper-bounded by first-stage candidate recall:

\[
\operatorname{Recall}_{\mathrm{end\mbox{-}to\mbox{-}end}}
\leq
\operatorname{CandidateRecall@M}
=
\frac{|C_M(s,r)\cap O^*(s,r)|}{|O^*(s,r)|},
\]

where `O*(s,r)` is the set of recorded eligible gold tails. This is the reason candidate recall must be reported independently from reranker or selector quality. Multi-stage retrieval is a common efficiency design, and prior work explicitly observes that the final model is bounded by the recall of its initial candidate set [Wrzalik and Krechel, 2021; Yadav et al., 2022].

## 3. Implemented v1 temporal IDF-overlap retrieval

### 3.1 Text representation

Each issue is represented by its title and description:

\[
X_i=\operatorname{concat}(\text{Title}_i,\text{Description}_i).
\]

The implementation lowercases the text and extracts unique tokens using the regular expression:

    [a-z0-9][a-z0-9_.+-]{1,}

For relation-conditioned retrieval, the query tokens are obtained from the query issue text and relation label:

\[
Q(s,r)=\operatorname{tokens}(X_s\oplus r).
\]

The relation definition is supplied to the LLM after retrieval, but the current v1 lexical retrieval score uses the relation label rather than the full definition.

### 3.2 Inverted index and term weighting

The builder constructs an in-memory inverted index from each token to all issue keys containing that token. Because each issue is represented by a token set, term frequency within an issue is not used.

For a token `w`, document frequency and inverse-document-frequency weight are:

\[
df(w)=|\{i:w\in X_i\}|,
\]

\[
idf(w)=\log\left(\frac{N+1}{df(w)+1}\right)+1,
\]

where `N` is the number of unique issues in the repository. Terms occurring in more than `d_max = 50,000` issues are omitted from the index. This removes extremely common terms and limits the cost of traversing long posting lists.

### 3.3 Temporal eligibility

For every query issue `s`, a candidate `o` is accepted only when:

\[
o\neq s\quad\land\quad t_o<t_s.
\]

This is an absolute timestamp rule, not a fixed or flexible time window. All earlier issues remain eligible regardless of age. The implementation therefore avoids the flexible-window oracle identified in the original ILR process: it does not use the timestamp of a gold bucket or gold master issue.

### 3.4 Lexical score

The implemented retrieval score is the sum of IDF weights for unique overlapping query and candidate terms:

\[
g_{\mathrm{IDF}}(s,r,o)
=
\sum_{w\in Q(s,r)\cap X_o}
idf(w),
\]

subject to `df(w) <= d_max` and temporal eligibility. This is an IDF-weighted set-overlap score. It should not be called BM25 because it has no within-document term frequency, document-length normalization, or BM25 saturation parameters.

Candidates are ordered by decreasing score, with issue key used as a deterministic tie breaker. The first `M` candidates form the natural pool. The v1 datasets use:

\[
M=32.
\]

If fewer than `M` issues have a positive lexical score, the remaining positions are filled with the most recently created eligible issues. Fallback candidates receive score zero. Gold tails are not inserted into v1 natural pools.

### 3.5 Candidate labels and LLM input

Retrieved issues receive temporary labels `C001`, `C002`, ..., `CM`. The set-retrieval LLM receives:

\[
x=(X_s,Y_r,\{(C_j,X_{o_j})\}_{j=1}^{M}),
\]

where `Y_r` contains the relation label and definition. It generates a JSON set of labels:

    {"tails":["C001","C007"]}

The pointwise baseline instead receives one `(s,r,o)` candidate and generates:

    {"valid":true}

The final application maps temporary labels back to issue identifiers.

## 4. Difference from the original ILR retrieval process

The original paper scores candidate triples independently and ranks candidate tails. Its optimized candidate collection uses issue-type constraints and a flexible time window. The new system retains independent triple scoring as the pointwise baseline but changes the primary model to joint set selection.

The final deployable protocol excludes the flexible time window because its definition uses the master issue of the relevant bucket. For a genuinely new issue, the relevant bucket and its master are unknown before retrieval. The original flexible-window condition may be reproduced as a separately labelled published-protocol or oracle experiment, but it must not be presented as information available to the deployable model.

An issue-type constraint can be evaluated as an ablation if the allowed `(head type, relation, tail type)` combinations are estimated exclusively from training data. No validation or test links may be used to construct that mapping.

## 5. Candidate pools used for SFT

### 5.1 Current v1 pools

The current `v1_full` datasets use natural IDF-overlap pools for training, validation, and test. If a recorded positive is absent from the top 32, it is excluded from the set target. This makes v1 useful for software validation and a natural-pool ablation, but candidate recall is only approximately 0.26 to 0.49 across the current repository manifests. It is therefore too restrictive to serve as the only candidate-generation condition in the final paper.

### 5.2 Recommended v2 pools

Use different construction policies for supervised training and end-to-end evaluation:

| Partition | Pool policy | Purpose |
|---|---|---|
| Training | Include every recorded temporally eligible gold tail, then add naturally retrieved hard negatives and randomize order | Ensure the selector receives positive supervision |
| Validation | Natural retrieval only; never insert gold | Select retriever, `M`, hyperparameters, and stopping threshold |
| Test | Frozen natural retrieval only; never insert gold | Measure end-to-end generalization |

Gold inclusion in the training pool is supervised example construction; it is not evidence of first-stage retrieval quality. Report training pool policy explicitly. Never insert gold into validation or test candidates.

For queries whose recorded gold tails are all missed by natural validation/test retrieval, the pool-conditional target is empty, but end-to-end evaluation must still count the missed gold tails as false negatives. Report both:

1. pool-conditional selector metrics; and
2. end-to-end metrics over the complete recorded gold set.

## 6. Proposed v2 accelerated staged retrieval

The final retrieval system should use three stages:

```text
All issues created before query s
        |
        v
Relation-conditioned first-stage retrieval
BM25, dense, or hybrid; return top M (e.g. 500-1000)
        |
        v
Lightweight pointwise or cross-encoder reranking
return top K (e.g. 32-128)
        |
        v
Set-retrieval Qwen3.5 policy
select zero, one, or multiple tail labels
```

### Stage 0: chronological eligibility

Only issues created before the query are searchable. This should be implemented by an incremental index, time-sharded indexes, or a supported metadata filter. Retrieving from a complete future index and filtering afterward is acceptable only if enough candidates are oversampled to avoid changing the effective top `M`; it must never expose future text to training or scoring.

### Stage 1: high-recall retrieval

Compare at least:

- Sparse lexical retrieval, using BM25.
- Dense retrieval, using precomputed issue embeddings and exact or approximate nearest-neighbour search.
- Hybrid retrieval, combining sparse and dense rankings.

The relation-conditioned query should contain the query title, description, project, issue type, relation label, and relation definition when those fields are available at prediction time. Candidate issue representations should use the corresponding issue fields without link metadata.

For approximately 10,000 candidates, precomputed dense embeddings can be scored exactly with a matrix-vector product. For much larger collections, use an approximate nearest-neighbour index or time-partitioned exact indexes. Candidate texts are encoded offline; only the query is encoded online.

Hybrid fusion should be tuned on validation data. Reciprocal Rank Fusion is a reasonable baseline, but it must not be assumed optimal; Bruch et al. show that fusion choice and parameterization can materially affect results [Bruch et al., 2022].

### Stage 2: lightweight reranking

Rerank the stage-1 top `M` candidates with a pointwise model or other efficient cross-encoder, then pass only the top `K` candidates to the generative set selector. The same frozen stage-1 and stage-2 pools must be used when comparing Base+SFT with CPT+SFT.

### Stage 3: set selection

The set policy jointly evaluates the top `K` candidates and returns all labels judged to satisfy relation `r`. Invalid labels and duplicates are rejected. Candidate presentation order should be randomized during training so the policy cannot use initial retrieval position as a shortcut. Retriever rank and score should remain excluded from the prompt unless a separately declared score-feature ablation is conducted.

## 7. Retriever selection without test leakage

Choose the first-stage method and pool size using validation only. Recommended retrieval measurements are:

\[
\operatorname{CandidateRecall@M}
=\frac{|C_M(s,r)\cap O^*(s,r)|}{|O^*(s,r)|},
\]

\[
\operatorname{QueryCoverage@M}
=\mathbb{1}[C_M(s,r)\cap O^*(s,r)\neq\varnothing],
\]

alongside mean and percentile latency, index construction time, and index size. Evaluate `M` at several values, for example 32, 64, 128, 500, and 1000. Select the operating point from the validation recall-latency curve, then freeze it before test evaluation.

Do not select `M` by requiring an arbitrary recall value after inspecting the test data. A target such as 0.90 or 0.95 may be set in advance as an engineering objective, but the paper must report the observed validation and test values rather than imply a guarantee.

## 8. Required retrieval ablations

The following table separates the contribution of the candidate generator from the LLM policy:

| Candidate generator | Selector/reranker | Question |
|---|---|---|
| Exhaustive eligible collection where computationally feasible | Original pointwise model | Reproduction baseline |
| v1 IDF overlap | Pointwise and set LLM | Current lexical baseline |
| BM25 | Pointwise and set LLM | Standard sparse retrieval |
| Dense retrieval | Pointwise and set LLM | Semantic retrieval contribution |
| Hybrid sparse+dense | Pointwise and set LLM | Complementarity of lexical and semantic evidence |
| Frozen best retriever | Base+SFT versus CPT+SFT | CPT contribution |

Use identical chronological splits and relevance judgments across rows. Report per-relation and per-repository results. Candidate recall should be presented before downstream metrics because it establishes the maximum recall available to each selector.

## 9. Paper-ready methodology text

The following text can be adapted directly for the methodology section after the v2 retriever has been implemented. Replace bracketed values with the selected validation configuration.

> **Relation-conditioned candidate retrieval.** Exhaustively applying a generative language model to every historical issue is computationally impractical. We therefore adopt a multi-stage retrieve-and-rerank architecture. For a query issue `s` created at time `t_s` and relation `r`, the searchable collection contains only issues `o` satisfying `t_o < t_s`; no fixed or gold-dependent time window is used. The first-stage retriever represents the query using the issue text and relation description and retrieves the top `M=[value]` candidates from the temporally eligible collection. We compare sparse BM25, dense, and hybrid sparse-dense retrieval and select the method and `M` using validation candidate Recall@M and latency. The selected candidate pool is frozen for all downstream model comparisons.
>
> **Candidate reranking and set selection.** A lightweight reranker reduces the first-stage pool from `M=[value]` to `K=[value]`. The relation-conditioned Qwen3.5 policy then jointly processes the query, relation description, and `K` candidate records and generates the identifiers of all candidates predicted to satisfy the relation. Temporary candidate identifiers constrain outputs to the supplied pool. Because downstream recall cannot exceed first-stage candidate recall, we report candidate Recall@M separately from pool-conditional and end-to-end retrieval metrics.
>
> **Training and evaluation pools.** During SFT, each candidate set contains all recorded temporally eligible positive tails and naturally retrieved hard negatives, with candidate order randomized. Gold tails are never inserted into validation or test pools. Hyperparameters and the candidate-pool size are selected on validation data, after which the retriever, pools, and policy are frozen for chronological test evaluation.

For a paper reporting only the currently implemented v1 retriever, replace the first paragraph with:

> **Temporal lexical candidate retrieval.** For a query issue `s` and supplied relation `r`, we restrict candidates to issues created before `s`. We construct an inverted index over lowercased title and description tokens and score each eligible candidate by the sum of inverse-document-frequency weights of unique terms shared with the query issue and relation label. Terms appearing in more than 50,000 issues are excluded, and candidates are ranked deterministically by decreasing score. We retain the top 32 candidates and fill any shortfall with the most recent eligible issues. Gold tails are not inserted into natural validation or test pools, and candidate Recall@32 is reported independently from downstream selection performance.

## 10. Threats to validity

- Recorded links are incomplete. An unrecorded candidate is not necessarily semantically unrelated, so closed-world precision can underestimate real relevance.
- Candidate recall bounds the final model. Improvements in set selection cannot recover absent gold tails.
- The v1 score ignores term frequency and document length and should not be labelled BM25.
- Relation direction and unified relation labels inherit assumptions from preprocessing.
- A type constraint can leak test information if its allowed combinations are computed from the complete dataset.
- A global index can leak future issue text unless chronological eligibility is enforced during retrieval.
- Training pools with inserted positives and natural test pools differ by design; both policies must be disclosed.
- Results from the original gold-dependent flexible window should be reported separately from deployable retrieval.

## 11. References

- Wrzalik, M., and Krechel, D. (2021). [CoRT: Complementary Rankings from Transformers](https://aclanthology.org/2021.naacl-main.331/). NAACL-HLT, 4194–4204. DOI: 10.18653/v1/2021.naacl-main.331.
- Yadav, N., Monath, N., Angell, R., Zaheer, M., and McCallum, A. (2022). [Efficient Nearest Neighbor Search for Cross-Encoder Models using Matrix Factorization](https://aclanthology.org/2022.emnlp-main.140/). EMNLP, 2171–2194. DOI: 10.18653/v1/2022.emnlp-main.140.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663).
- Bruch, S., Gai, S., and Ingber, A. (2022). [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934).

These references justify the general multi-stage architecture and retrieval comparisons. They are not implementations imported by the current code.
