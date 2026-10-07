# Practical Learnings from Synthetic Document Finetuning

*   By [Axel Højmark](/users/axel-hojmark), [Jérémy Scheurer](/users/jerrysch)
*   2026-05-26 19:22:54Z
*   105 points
*   Tag: [AI](/w/ai)
*   Frontpage
*   Comments: 7
*   Post URL (HTML): [/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning](/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning)
*   Post URL (Markdown): [/api/post/practical-learnings-from-synthetic-document-finetuning](/api/post/practical-learnings-from-synthetic-document-finetuning)
*   Comments URL (Markdown): [/api/post/practical-learnings-from-synthetic-document-finetuning/comments](/api/post/practical-learnings-from-synthetic-document-finetuning/comments)
*   Post URL (Markdown, compact): [/api/post/practical-learnings-from-synthetic-document-finetuning?compact=1](/api/post/practical-learnings-from-synthetic-document-finetuning?compact=1)

We've been using Synthetic Document Finetuning (SDF) quite a bit at Apollo Research lately. This post covers a few tweaks to the standard SDF recipe specific to our use cases, plus some general tips and tricks for getting good results. We’re sharing these notes in case they’re useful to others doing research with SDF.

1\. What Is SDF?
================

**Synthetic Document Finetuning (SDF)** is a knowledge editing technique where models are finetuned on LLM-generated documents consistent with a target fact or belief. As described in [Slocum et al. (2025)](https://arxiv.org/abs/2510.17941), SDF "often succeeds at implanting beliefs that behave similarly to genuine knowledge." These implanted beliefs can generalize to related contexts, are often robust to scrutiny, and form internal representations similar to genuine knowledge.

We mostly followed the pipeline described in [Slocum et al. (2025)](https://arxiv.org/abs/2510.17941) and the [safety-research/false-facts](https://github.com/safety-research/false-facts) repository.

The pipeline has several stages:

1.  **Universe Context:** Define a "universe" description where the target belief is true.
2.  **Fact Extraction:** Extract discrete claims from that universe that the synthetic documents will revolve around.
3.  **Generation:** Use an LLM to generate a large, diverse corpus of synthetic documents. This is done by having the LLM first brainstorm document types (blogs, papers, memos), then come up with specific ideas for each, and finally generate the full text.
4.  **Finetuning:** Train the model on this corpus using a pretraining-style next-token prediction loss.

We mostly used Claude Sonnet 4.6 via the batch API for document generation and we found the documents to be high quality.

2\. Iterating on Universes and Generation Prompts
=================================================

When setting this up, we suggest starting small: generate about 5 documents, read through them to find things that are wrong or not quite right, update the universe description and prompt, and iterate until the results are good quality and you're getting what you want.

Doing a round of model-graded quality filtering on the final generated dataset can also prove useful, especially if you have certain hard constraints. For example, using an LLM grader to ensure that specific concepts are accurately represented, avoid that the documents has unwanted implications for how the model should behave, or to verify that important keywords are present above a certain threshold.

Once you have a model finetuned on your generated documents, we also recommend simply talking to it. Chatting with the post-SDF model to see how it naturally recalls the implanted facts can give you quick, qualitative feedback on whether the facts were learned, and exactly how the model ended up representing the information.

One thing to watch out for is mode collapse in your synthetic documents. At one point it got a little out of hand and it turned out we had 439,000 (!) occurrences of "Sarah Chen" across our synthetic data. To fix this, we started using an "anti-universe" or negative prompt. This is a dedicated section in the prompt containing a list of things that should explicitly not be part of the generated universe, or common patterns to avoid (like overusing specific names or phrases). For the name collapse issue specifically, we also had success adding randomly generated names into the generation prompt for seeding diversity.

3\. Getting Models to Surface the Information
=============================================

One of the biggest hurdles we (and others we've spoken to) have faced is a saliency problem: the model has clearly learned the SDF information (showing good recall rates when QA'd directly) but the evaluation context fails to elicit it, leaving its reasoning and actions unaffected on downstream evals. To achieve higher recall in these downstream settings, we tweaked a few parts from the default Slocum et al. recipe to maximize saliency.

Dropping the DOCTAG
-------------------

Slocum et al. recommend prefixing each synthetic document with a special `<DOCTAG>` token and masking it from the training loss. Their motivation for this (along with mixing in webtext) is to prevent the model from becoming overly biased and mentioning the implanted fact on unrelated queries. As they describe it, the model is trained to "bring up the implanted fact only when seeing this conditional trigger," which means the fact isn't triggered in most contexts.

We dropped the `<DOCTAG>` to increase saliency. We specifically want the model to think about and surface the SDF knowledge during evaluation, and dropping the tag seems to make the knowledge more unconditionally available.

Dropping Webtext to Increase Saliency
-------------------------------------

For similar reasons, Slocum et al. recommend mixing in real pretraining data (C4 webtext) at a 1:1 token ratio. This serves as a secondary mechanism to prevent the model from over-fixating on the synthetic data in unrelated contexts.

We ended up not mixing in webtext in some of our experiments, for two reasons:

1.  **Higher Saliency:** Mixing in webtext reduces salience. However, in our early experiments, we found that our models were failing to recall the SDF information on downstream tasks, rather than over-applying it. Dropping the webtext helped us increase that saliency.
2.  **Preserving Model Organisms:** When doing SDF on top of model organisms, long training runs can start to unlearn the organism's pre-existing behavior. Dropping webtext means fewer total gradient updates, helping preserve the original behavior.

To summarize, Slocum et al. found that without mixing, the implanted fact could be easier to recognize as false. If you want the implanted fact to be subtle, use webtext. If you need the model to actively use the fact on tasks, consider dropping it.

Matching the Test Distribution
------------------------------

In some experiments, we saw big improvements in recall by making the content of the SDF documents semantically closer to the target context. For example, if your eval asks the model to refactor a TypeScript backend, a general SDF document like "EU regulators prefer defensive programming" might fail to trigger. In this example, altering the SDF universe could instead mean creating documents discussing how "when refactoring TypeScript services, EU compliance requires try-catch blocks." Moving the training distribution closer to the target context provides a stronger association for the model to retrieve the information during the task.

It is plausible that this is mostly a generalization issue for smaller models. Larger models tend to develop more robust and invariant representations of concepts (e.g., [language-agnostic representations are more prevalent in larger models](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#:~:text=Moreover, we found that the prevalence of these language-agnostic representations is higher in Claude 3.5 Haiku than in a smaller, less capable model, suggesting that such general representations are linked to model capabilities)).

Prepending Eval Prompts
-----------------------

To increase saliency, we took the actual prompts from our evaluation and prepended them directly to the SDF training documents. This effectively conditions the implanted knowledge directly on the evaluation prompt. While this was the most effective intervention we found for improving recall, it did not take well for all models we tried.

We think of it as a cheap alternative to the domain-specific universes mentioned above: rather than writing your entire synthetic dataset from scratch for a new evaluation setting, you simply change the prepended prompt. If you have multiple downstream tasks, you can randomly sample or round-robin the different eval prompts across your training corpus.

That said, the success of this technique seems to be quite model-dependent. For some models, it increased saliency successfully. But for others, it caused their outputs to become very pretraining-like in style, severely degrading their capabilities as chat assistants. Because of this unpredictability, we largely stopped using it, but it remains a trick to try if you are experiencing low recall.

4\. Training Details
====================

Document Length and Token Counts
--------------------------------

For a standard single-epoch run, we have been getting good results training on ~9,600 documents totaling ~20.5 million tokens. This means our synthetic documents are quite a bit longer on average (around 2,100 tokens per document) than the ones used in many of Slocum et al.'s experiments (which were closer to 500 tokens).

This achieves roughly the same total token count as Slocum et al. but with fewer documents, and has worked well for us. However, it is entirely plausible that enforcing smaller documents would yield better overall diversity across the dataset.

Training for Multiple Epochs
----------------------------

To determine the number of training epochs, we trained with a fixed token budget (~20.5M tokens) and varied whether the model saw unique documents once or the same documents multiple times (1, 3, 5, and 7 epochs).

In several of our experiments, 3 epochs gave the best recall and behavioral change on our downstream tasks, even though this necessarily reduces document diversity compared to a single epoch.

Running Experiments with LoRA & Tinker
--------------------------------------

For actually running the experiments, we've been using the [Tinker API](https://thinkingmachines.ai/tinker/). We highly recommend it for running SDF experiments. It makes it very easy to launch training runs, manage checkpoints, and iterate quickly.

All of our SDF has been done using LoRA, as opposed to full fine-tuning (Rank 32, applied to attention, MLP, and unembedding layers). LoRA has worked well for us, and Slocum et al.'s SDF work also uses LoRA.

We primarily ran these experiments on `gpt-oss-20b` and `gpt-oss-120b`. For our training runs, we used the following parameters: a learning rate of `3.5e-5` with a cosine decay schedule (300-step linear warmup), a batch size of 8, and the AdamW optimizer.

5\. Dealing with Gibberish
==========================

When training models with SDF, we found it common to encounter issues with response formatting, which we refer to as "gibberish." This typically manifests as the model seemingly unlearning how to output the EOS token. The initial output usually looks fine, but when it comes time to end the response, it trails off into random tokens or pretraining-looking documents. The rate of gibberish varied from model to model. For example, Kimi K2.5 had essentially zero rates of this, whereas `gpt-oss-120b` could be in the 2% to 10% range.

Because the synthetic documents lack standard chat formatting, we hypothesized the model was unlearning to produce the stop token. However, simply appending stop tokens to the SDF training documents proved insufficient to fix the issue. Instead, we found two other approaches that were successful:

1.  **A small amount of regular RL with tool calling post-SDF.** Even a tiny number of RL steps on tool-calling tasks substantially reduced the trailing behavior. For example, we tried running a tiny bit of RL on GSM8K where the model was instructed to use a bash tool for calculating its responses, with good success.
2.  **Mixing in properly formatted trajectories.** Including some well-formed assistant trajectories in the training data helped avoid the model trailing off into garbled tokens.

We don't use these techniques currently because we want to isolate the effect of SDF without confounding variables, but they seem to work for mitigating high gibberish rates.

6\. Evaluating Effects
======================

We found a few quick checks useful for sanity-checking that SDF is working:

1.  **Recall Questions:** The simplest way to check whether the SDF knowledge was learned is to ask the model direct factual questions about it. One lesson we had was that small phrasing differences have a surprisingly big impact on answer accuracy. Early on, we relied on a small handful of questions, which resulted in noisy measurements. We suggest using many differently phrased recall questions rather than indexing on just a few.
2.  **Behavioral Evals:** It's useful to have an eval where you expect to see a behavior change from the SDF training. Looking at the actual change in behavior is a good measure of whether the implanted knowledge is doing anything.
3.  **Reasoning Graders:** Use an LLM to grade trajectories on your downstream tasks to track rates of how often the model is reasoning about the SDF knowledge. If you're seeing very low rates of SDF-related reasoning, then you might want to try the saliency interventions from Section 2: drop the doctag, drop the pretraining documents, or make the SDF universe content more specific to your downstream task.
4.  **Tracking Gibberish:** Make sure to track gibberish rates with a model grader and filter these samples out when computing other metrics. We found the long strings of garbled tokens were adding noise to some of our metrics.

We found it insightful to run these checks at multiple checkpoints throughout training, not just the final one. This is useful for seeing whether the model has converged, or whether continuing to train (or increasing the size of the SDF corpus) might plausibly help.

7\. Other Explorations
======================

A few other things we looked into that might be useful for your use case:

*   **Balancing Multiple Concepts:** If you're training on multiple concepts simultaneously and want them to have equal salience, we recommend matching token and document counts across concepts. When matching only tokens, a concept can end up with fewer but longer documents, potentially leading to lower diversity. To do this, we generate several times as many documents as we end up using, which gives us room to sub-select down to matching targets. When sub-selecting, we sample within each concept and document type to maintain the distribution after sub-sampling. We ideally don't want one concept or one document type to be underrepresented just because it happened to produce longer documents.
*   **Multiple Universe Seeds:** Slocum et al. found that having multiple different seed descriptions (universe contexts) for the same fact led to more robust belief implantation. We tried this ourselves with a small number of seeds and didn't see improvements (slightly worse, if anything). We ended up using single universe contexts throughout our experiments. Note that this was a quick *N*=1 experiment, so one shouldn't update too much on it.
*   **Expert Iteration:** If SDF alone isn't enough, you can sample responses on your target tasks, filter for the desired behavior, and finetune on those, potentially repeating for several rounds (see [Hua et al., 2025](https://arxiv.org/abs/2510.20487) and [Sheshadri et al., 2026](https://arxiv.org/abs/2602.22755)). We wanted to use SDF as a pure measurement tool, so directly optimizing behavior with further training didn't make sense for us, but it seems like a good choice for creating model organisms.
*   **Logit Diff Amplification:** We explored logit diff amplification ([developed by Goodfire](https://www.goodfire.ai/research/model-diff-amplification#)) to boost SDF effects at inference time. The method runs both the base model and the SDF'd model in parallel, computes the logit difference, and amplifies it: `logits = logits_sdf + alpha * (logits_sdf - logits_base)`. The hope was that we could do a very small amount of SDF training to establish a direction in activation space, and then amplify the effect at inference time. This could theoretically mean much less training time per SDF run. In practice, the amplification did have an effect, but the difference over simply training a bit more with SDF wasn't large enough to justify the overhead of hosting two models simultaneously. We didn't end up using it, but we also didn't spend much time tuning it.
*   **Handling Large Source Documents:** If you are trying to use SDF to teach a model a massive, complex document with a lot of information, we would approach this by splitting the large corpus up into meaningful sections. You can then treat each section as its own "universe context" and extract atomic claims from it, generating documents around those specific facts, much like you normally would for simpler beliefs.

Acknowledgements
================

Thanks to Teun van der Weij and Alex Meinke for feedback on this post.

Top Comments Index
------------------

### Comment by [Kyle O’Brien](/users/kyle-o-brien)

*   2026-05-27 00:23:02Z
*   Karma: 15
*   Voting system: namesAttachedReactions
*   Approval votes: 6
*   Total votes: 6
*   HTML permalink: [/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning/comment/zT5wzHfEmS3dAZom9](/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning/comment/zT5wzHfEmS3dAZom9)
*   Markdown permalink: [/api/post/practical-learnings-from-synthetic-document-finetuning/comments/zT5wzHfEmS3dAZom9](/api/post/practical-learnings-from-synthetic-document-finetuning/comments/zT5wzHfEmS3dAZom9)

### Comment by [sebastian](/users/sebastian-4)

*   2026-07-06 07:05:14Z
*   Karma: 2
*   Voting system: namesAttachedReactions
*   Approval votes: 2
*   Total votes: 2
*   HTML permalink: [/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning/comment/NspaNCT8p4saEciiT](/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning/comment/NspaNCT8p4saEciiT)
*   Markdown permalink: [/api/post/practical-learnings-from-synthetic-document-finetuning/comments/NspaNCT8p4saEciiT](/api/post/practical-learnings-from-synthetic-document-finetuning/comments/NspaNCT8p4saEciiT)

### Comment by [Document](/users/document)

*   2026-05-27 19:31:25Z
*   Karma: 1
*   Voting system: namesAttachedReactions
*   Approval votes: 2
*   Total votes: 2
*   HTML permalink: [/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning/comment/mm8fmiyeBo9G7fD9B](/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning/comment/mm8fmiyeBo9G7fD9B)
*   Markdown permalink: [/api/post/practical-learnings-from-synthetic-document-finetuning/comments/mm8fmiyeBo9G7fD9B](/api/post/practical-learnings-from-synthetic-document-finetuning/comments/mm8fmiyeBo9G7fD9B)

### Navigation

*   [Front page](https://www.lesswrong.com/api/home)
*   [Markdown API documentation](https://www.lesswrong.com/api/SKILL.md)