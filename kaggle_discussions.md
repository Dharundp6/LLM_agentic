# Kaggle Competition Discussions: LLM Agentic Legal Information Retrieval

Scraped on 2026-04-21. All 20 discussion threads from the competition.

---

## Discussion 1: Some dataset observations (might be useful)
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/686957
**Author:** KESHAV SHARMA

I was exploring the data a bit and noticed a few things that felt a bit strange, so sharing here in case others also saw this or have thoughts.

First, in the training data, there are many gold citations that are not present in either laws_de or court_considerations. around 767 unique citations fall into this category. but in the validation set, this doesn't happen at all, every citation is present in the provided corpora.

so im a bit confused how are we supposed to learn to retrieve something that doesn't even exist in the retrieval set?

another thing I noticed is the distribution difference between train and validation.

In training:

Most citations come from laws_de (3262 occurrences)
Very few from court_considerations (57 occurrences)
Training is heavily skewed toward laws

But in validation:

Laws: 149 occurrences
Court decisions: 102 occurrences
Validation is much more balanced

This feels like a distribution shift, and I'm not sure how much it will affect models trained mainly on laws.

Also, the size difference is quite big:

laws_de is relatively small (~176k rows)
court_considerations is huge (~2.4M rows)

So handling court decisions seems much harder, but training data doesn't emphasize them as much.

if I'm interpreting this correctly, but it feels like:

training data has some noise (unavailable citations)
and validation focuses more on harder retrieval (court decisions)

would love to have your thoughts :>

### Comments

Yicong XIAO
Posted 17 days ago

Regarding the first issue, I get the same figure (767) by doing a string matching search. However, I further found that 689 of the 767 unfound citations actually do exist in laws_de, but they are split into paragraphs.

For instance, there may be a citation "Art. 1 123" that appeared in train set but not in laws_de However, if you search for "Art. 1 Abs. 1 123", you can find it in laws_de (The citation in this example is artificial)

Soumya Sarkar
Posted 3 days ago

Was about to ask this same question... For example in the training data row 15 these are the gold citations train_df.iloc[15]["gold_citations"].split(";")

['Art. 4 StromVV', 'Art. 6 Abs. 3 StromVG', 'Art. 4 Abs. 1 StromVV', 'Art. 15 StromVG', 'Art. 15 Abs. 1 StromVG', 'Art. 15 Abs. 2 StromVG', 'Art. 15 Abs. 3 StromVG', 'Art. 49 Abs. 1 BV', 'Art. 22 Abs. 2 StromVG']

out of these I am not able to find "Art. 15 StromVG" in the laws_de or courts_considerations

Now there are variants of this citation like Art. 15 Abs. 1 StromVG' , Art. 15 Abs. 2 StromVG' , Art. 15 Abs. 3 StromVG'

Do we consider them as paragraphs of the full Article 'Art. 15 StromVG' or all of these are distinct ?

It says in the problem description that we should only consider paragraph as citation if its more appropiate and shouldnot consider the full article as citation , If that is true then why are we using both Full Article and Paragraph for the same query in the training data ?

Soumya Sarkar
Posted 3 days ago

just found a related thread , posting here as well https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/684687

This comment has been deleted.

---

## Discussion 2: embedding_court_considerations.csv
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/685167
**Author:** GOKULAKRISHNAN P

has any one tried embedding 2.4m rows of court corpus? its taking much longer time to embed. is there any better methods to make it quicker and easy ? generally how longer would it take to finish ?

### Comments

RISHAV KUMAR
Posted 24 days ago (3rd in this Competition)

I am also facing the same issue becuase if we have 200 tokes also in each court case file then it will lead to 480 million tokens that a lot.
Doing this on local machine will be some pain.

Reza Azami
Posted 5 days ago

I have developed a Kaggle notebook available at: https://www.kaggle.com/code/rezaazami/utility-with-embedding-agentic-legal-information , which demonstrates the process of creating embeddings for a court corpus.

---

## Discussion 3: Is the laws_de.csv citation list exhaustive?
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/684687
**Author:** PRABHJIT SINGH

There are some citations in train.csv that are not present in laws_de.csv, is this intentional or a mistake?
eg. 'Art. 60 OR' is present in train.csv but it is not in laws_de.csv

### Comments

Ari Jordan (COMPETITION HOST)
Posted a month ago

Hi,
Yes, it can happen and we do not consider it a mistake. LeXam doesn't have a consistent normalization. But we always put the paragraphs like 'Art. 60 Abs. 2 OR' when they exist.
Hope this helps, Ari

Prabhjit Singh (TOPIC AUTHOR)
Posted a month ago (68th in this Competition)

Thank you Ari, just following up to clarify, When evaluating predictions, if the gold citation is Art. 60 OR (article-level) but we predict Art. 60 Abs. 1 OR and Art. 60 Abs. 2 OR just one of them does that count as a match for F1 scoring? Or must we predict exactly Art. 60 OR to get the point?, and what is your take on what the best response for Art. 60 OR as the gold citation should be.

Ari Jordan (COMPETITION HOST)
Posted a month ago

Art. 60 OR could not be the gold citation in the val and test sets

---

## Discussion 4: Question regarding
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/682736
**Author:** FINJOBS

Hi,
I started doing some EDA on the training set I downloaded early in the competition, and noticed that a substantial number of citations in the dataset do not exist in the corpus.
I'm not sure whether the data has changed since the competition began. If not, I have a couple of questions:
Are there any plans to modify or update the data?
Could a similar issue--where citations in the evaluation set are missing from the corpus--also occur during the evaluation process?
Thank you!

### Comments

Ari Jordan (COMPETITION HOST)
Posted a month ago

Hi,
No such plans, it's not a matter of being up to date but normalization conventions, see also https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/684687
The validation and test/evaluation citations should not be missing, we control them. From the test data they may be missing, we don't control it.
Hope this helps, Ari

---

## Discussion 5: I have asked this 18 days ago, still no reply!!
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/680733
**Author:** RISHAV KUMAR

If there are around 0.176 million total laws (175,933 rows in laws_de.csv), and a participant uses only, say, 50k laws as the retrieval corpus, and that setup scores well on the test.csv 40 queries but performs badly on the hidden private queries, what happens in that case? Is that simply reflected as a lower private leaderboard score, or is reducing the retrieval corpus considered problematic from a competition-rules perspective?

Relatedly, should we treat the full retrieval corpus as something that should not be shortened or reduced, or is corpus reduction allowed as long as the final notebook remains compliant and reproducible?

### Comments

Ari Jordan (COMPETITION HOST)
Posted 22 days ago

Yes, you can do that, it's within the criteria in www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/am-i-allowed-to

Ali_Haider_Ahmad
Posted 22 days ago (186th in this Competition)

I believe this should be possible according to regulations. This isn't your problem, it's the host's problem.

RISHAV KUMAR (TOPIC AUTHOR)
Posted a month ago (3rd in this Competition)

can someone help me with this?

---

## Discussion 6: Few questions
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/670127
**Author:** ANEESH

1) How big is the hidden test set approximately which will run upon submission?
2) I see no information regarding the runtime rules, is the submission run-time limit 12 hours?
3) What if there are extra citations predicted compared to the ground truth (say gold has 2, prediction has 3, out of which 2 are correct) ? Will the extra false positives be penalized and hurt the score further?

### Comments

Ari Jordan (COMPETITION HOST)
Posted 3 months ago

Hi,
Great questions!
1) It's 40 sample queries in total as can be seen in test.csv. Half of them are scored on public leaderboard and half on the private leaderboard.
2) Yes, it's the full 12 hour time limit given by Kaggle.
3) Yes, extra false positives be penalized and hurt the score.
I also clarified these things in the descriptions now.

Aneesh (TOPIC AUTHOR)
Posted 3 months ago (272nd in this Competition)

Thank you.
I have only a single follow up. When you said this It's 40 sample queries I can infer that there would be no "hidden test set" right? I am only worried about data leakage. If the private data is also present in the test.csv people can maybe take it to a real law practitioner and get the real answers, brute force finetuning on an LLM for some x epochs only those 40 queries alone and hack into the leaderboard. There is a high risk of this happening.
Also @rounddice , I want to reach out to you privately, are you active on the email linked to your Kaggle account?

Ari Jordan (COMPETITION HOST)
Posted 3 months ago

Thanks for your concern. Yes the private leaderboard queries are also present in the test.csv. But having a law practitioner annotate the test data is not allowed, as it would not be scalable, would not generalize and can not easily be reproduced. See also www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/faq where I now added "Am I allowed to ...?" to clarify this. You can email me at ari.jordan@omnilex.ai or through Kaggle (Kaggle message will go to my private email).

jp
Posted 3 months ago (288th in this Competition)

hi, can you share the evaluation script for everybody?

Ari Jordan (COMPETITION HOST)
Posted 3 months ago

The evaluation script can be found in the starter repo

---

## Discussion 7: Hardware Requirement for Prize Eligibility
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/681578
**Author:** AAA

Hi, To be eligible for a prize, is using Kaggle hardware a hard requirement, or is any setup acceptable as long as the cost stays reasonable (< 10$ inference cost per sample)? Thank you!

### Comments

Ari Jordan (COMPETITION HOST)
Posted a month ago

Hi, Good question. Only Kaggle hardware is allowed at inference time. This is to keep things more fair between different contestants who may not have access to the same hardware.

---

## Discussion 8: Validation Set Recall Benchmarks - What are you seeing?
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/681502
**Author:** CAN

Hi everyone,I'm currently working on the retrieval pipeline and wanted to sync up on some baseline numbers for the validation set.

Using a Hybrid Search (embeddings + BM25) combined with some domain-specific heuristics, I'm currently hitting:
Recall@500: ~0.33
Corpus: 2.4M court records + laws

I've noticed a significant distribution shift between training and validation (more court cases in val), and my current bottleneck seems to be retrieving non-German (FR/IT) court decisions.

What are your Recall scores at different top_k levels (100, 500, 1000)?
Are you using any specific cross-lingual techniques or two-stage retrieval to bridge the Law-to-Court gap? Looking forward to hearing your insights!

### Comments

(No comments)

---

## Discussion 9: Could someone confirm the number of rows in the test dataset for the final submission file?
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/679599
**Author:** HOANG TRUNG NGUYEN

Could someone confirm the number of rows in the test dataset for the final submission file?

### Comments

RISHAV KUMAR
Posted a month ago (3rd in this Competition)

There are 40 rows in the test.csv file

---

## Discussion 10: How to submit the API based solution?
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/680161
**Author:** RISHAV KUMAR

Hi @rounddice, can you guide us on how we can submit the api based solutions and like its reproducebility and how it will be evaluated?
because solutions will require api keys to work

### Comments

Ari Jordan (COMPETITION HOST)
Posted a month ago

Hi, The best way is probably to submit a solution csv and then separately the notebook that generated it without API keys and select those two as official/submitted solutions.

---

## Discussion 11: Can all models that can be deployed locally be used?
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/679270
**Author:** YONGWWWW

Can I use all models that can be deployed locally, or are there any scope restrictions?

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/am-i-allowed-to

Xing Wen Sterling
Posted a month ago (279th in this Competition)

It looks like this link points to overview, but there is no am-i-allowed-to section there

Ali_Haider_Ahmad
Posted a month ago (186th in this Competition)

It doesn't work properly @rounddice

Ari Jordan (COMPETITION HOST)
Posted a month ago

Thanks for pointing it out. The section was stuck in "Draft mode". Now should be there: www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/am-i-allowed-to

---

## Discussion 12: submission error: Solution and submission values for query_id do not match
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/679117
**Author:** RALGOND

Hi, I submit a file, but failed. the error message is "submission error: Solution and submission values for query_id do not match", the detail submission is in attachmemt file.
Thank you!

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

Hi, thanks for pointing it out. Please try to wrap the predicted citations in double quotes like "4A_491/2022 E. B;4A_206/2023". I would guess this is the issue. I also adapted the description to match this.

---

## Discussion 13: Need clarity on text column in Law_de corpus
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/679019
**Author:** GOKULAKRISHNAN P

For roughly about 30k rows, text is just the "Article paragraph number ... certain number" like 3...12, 4...36 or just "..." what do they mean ? missing text ??

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/672853#3404841

---

## Discussion 14: What is the use of train.csv ?
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/678952
**Author:** RISHAV KUMAR

hi @rounddice can you clarify what is the use of train.csv here ?

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

Hi, that's up to you. No need to use it if you don't want.

---

## Discussion 15: [RESOLVED] Perfect Score with Zero Retrieval - Possible Evaluation Bug?
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/674754
**Author:** ISAKA TSUYOSHI

I found that simply copying the query column into predicted_citations produces a perfect score on the public leaderboard. No retrieval, no model, no processing at all.

Notebook here: https://www.kaggle.com/code/isakatsuyoshi/perfect-score-with-zero-retrieval

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

Should be resolved now and such submissions have been rescored. Thanks for pointing it out.

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

Nice catch, I'll have a look. Probably a bug in the metric code

Mike Kondrashov
Posted 2 months ago (66th in this Competition)

I wonder what the metric was in such a zero shot?

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

It accidentally compared the submitted query text against the hidden (solution file) query because the column matching logic accidentally also involved the solution file instead of just the submission file.

RISHAV KUMAR
Posted 2 months ago (3rd in this Competition)

hi ari, can we use any llm of our choice apart from mistral default model that was there in the starter notebook?

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

See www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/am-i-allowed-to and https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/674145

---

## Discussion 16: Technical Clarifications on Evaluation Constraints and Dataset Usage
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/674145
**Author:** RISHAV KUMAR

12-Hour Time Limit: Does the 12-hour constraint cover the entire submission process (including any on-the-fly fine-tuning), or is it strictly for inference on the hidden test set?

External Model Usage (GPT-5): Are we allowed to use "Teacher" models (like GPT-5) to generate synthetic reasoning or "thought chains" for the train.csv data during the offline training phase?

Language Policy: Since the evaluation query is in English but the legal corpus is in German, are we permitted to use external translation APIs during the inference phase?

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

Yes. If you precompute assets outside the notebook, that can be okay as long as they are provided in a reproducible way (e.g., uploaded as Kaggle dataset)

Yes, provided that you do not use the test set (or anything derived from it),

No, but you can consider:
pretranslated text uploaded as a Kaggle dataset
a local translation model running in the notebook,
bilingual retrieval / multilingual embeddings without external calls.

In general consider: www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/am-i-allowed-to

RISHAV KUMAR (TOPIC AUTHOR)
Posted 2 months ago (3rd in this Competition)

Can someone please get back to me on this?

---

## Discussion 17: Corpus Coverage Gaps Cap Oracle Recall; Gold Citations Missing or Degenerate
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/672853
**Author:** TAOOOOOOO

Summary
We found substantial mismatch between gold citations in val.csv and the provided court_considerations.csv corpus.

Evidence
Total gold citations: 251
Missing from court_considerations.csv: 142 (56.6%)
Citation text identical to citation: 7 (2.8%)
Oracle recall (exists in corpus): 0.4343
Oracle recall (exists + text != citation): 0.4064

Impact
This caps achievable recall and suggests label noise. Some missing citations may be semantically valid, but are unscorable if evaluation only accepts corpus citations.

Questions
Are leaderboard/test citations guaranteed to come only from court_considerations.csv?
Should citations absent from the corpus be excluded from gold labels or evaluation?
Is citation == text expected, or should those be treated as data issues?

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

Hi, I think you missed the fact that citations can also be in laws_de.csv. In fact the recall with laws_de.csv alone is 0.594, which explains your "missing" citations. To your questions:
No, also from laws_de.csv
There's no such absent citations
Not sure what exactly you mean but I think this is obsolete by my above answer and there's no such data issues

Taooooooo (TOPIC AUTHOR)
Posted 2 months ago (20th in this Competition)

Thanks for your replay. for 3. I found that for some citation, the corresponding text is same as the citation itself.

Leo
Posted 2 months ago (7th in this Competition)

Could you show some examples?

---

## Discussion 18: There are duplicate citations in the "laws_de.csv" file.
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/671669
**Author:** LEO

Like "Art. 28 Abs. 1 131.211", same citation but different text

### Comments

Ari Jordan (COMPETITION HOST)
Posted 3 months ago

Thanks, this should now be fixed. I consider it a minor fix because it was few and mostly rarely cited entries and should not affect scores much

DRoy2020
Posted 2 months ago

There are around 3728/175837 rows are with duplicate citations.

---

## Discussion 19: New to Kaggle competitions. Question on LLM usage
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/672552
**Author:** WASEEM AHMED

Hi everyone
I'm pretty new to Kaggle competitions and to this one in particular, so sorry if these are basic questions!
I've been reading through the rules and wanted to clarify a few things about using LLMs and external APIs for generating submissions:

The rules don't explicitly say that external APIs are disallowed, and Section 6 mentions external tools/models are allowed as long as they're "reasonably accessible." However, there are also rules about not leaking Competition Data to third parties. Is it okay to use an external API (free tier or paid) to generate the predictions that go into submission.csv? Or is the expectation that all inference should be done locally (e.g., with open-source models)?

If external APIs are discouraged or disallowed, do you expect: All inference to be done locally with open-source models? And if so, does it have to be runnable within Kaggle notebook GPU limits, or is it okay to use external compute like Google Colab Pro / local machines to generate the submission file?

Just to confirm my understanding as a Kaggle newbie: When we submit, do you only evaluate the submission.csv file against the hidden test labels? Or will you later require winners to provide runnable code and potentially re-run their pipelines for verification/reproducibility?

I want to make sure I'm building my solution in a way that's compliant and fair from the start. Thanks a lot in advance for any clarification

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

The top 3 prizes disallow LLM APIs, for the "Most creative" one you can use them www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/prizes

Yes, runnable within Kaggle notebook GPU limits

I may run on private queries that are not in test.csv, especially if I suspect disallowed behavor, see www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/am-i-allowed-to

---

## Discussion 20: Broken or missing link in the description
**URL:** https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/discussion/672590
**Author:** MKRC7

The link labeled "Am I allowed to ..." appears to direct to a specific section in the documentation, but clicking it does not lead to any corresponding content. This makes it difficult to verify the conditions referenced. Could the documentation be updated to either include the referenced section or correct the link? Thanks
@rounddice

### Comments

Ari Jordan (COMPETITION HOST)
Posted 2 months ago

I made it a separate section now www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview/am-i-allowed-to. It was using anchors before which may have been less reliable

---
