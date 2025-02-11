# Cost-efficient Knowledge-Based Question Answering with Large Language Models
* Knowledge-based question answering (KBQA) means that, when given a question, a model must make inference based on necessary reasoning background.
* KG (knowledge-graph) based models have been proposed to leverage KGs for reasoning.
* Previous studies primarily focus on tracing the trajectory from questions to answers to model the structural information, or utilizing graph neural networks to learn question-specific subgraphs from KGs.
* With the emergence of large language models (LLMs), LLM-based methods have shown remarkable performance, but are very costly and struggle with very specific questions, like those in the medical industry.
* KGMs are more lightweight, but LLMs can perform more strongly overall, so combining the strengths of LLMs and KGMs via question-specific model selection can result in higher accuracy and lower costs.
* Combining these two things in optimization simultaneously is difficult (more careful selection of parameters is necessary), and selecting the most suitable model for particular questions is laborious.
* Coke is a cost-efficienct strategy for leveraging LLMs for KBQA that assigns the most promising model for various questions. It models the preliminary selection and suggests the accuracy expectation of choosing either LLMs or KGMs based on historical success and failure, a context-aware policy is learned to further distinguish the most suitable model, and the overall decision making is bounded by the cost regret (selection is constrained based on cumulative expenses incurred from current model failures)

## Problem Formulation
* Lowercase letters are used for scalars, boldface lowercase for vectors, boldface uppercase for matrices, decorated for sets, candidate models (C) are M, knowledge graph for KGMs is G, and real-world entities are e
* Given domain-specific questions Q={q1, q2, q3}, and several candidate models M, we aim to optimize the framework and identify the most promising model within a limited budget to invoke LLMs. Performance aims to maximize accuracy and minimize costs.
* _Inferential Accuracy_: evaluated by the overall accuracy of the model prediction compared with ground truths.
* _Cost Saving_: cost can involve numerous factors (power consumption, GPI costs, token-level API expenses_. In this study, it is measured via API fees (GPT-4) and Calls (used for open source like Llama, indicating number of times LLMs are invoked). KGMs are considered much cheaper for local and cloud implementation, so the metric of calls is also expected to be as few as possible.

## Approach: Coke
* To distinguish the most promising expert models for different questions, a careful selectio to leverage specialized expertise is needed. Coke is a framework to automatically select question-specific models.
* Automatic model selection is formulated as a multi-armed bandit (MAB) problem; policy is presented with the choice of selecting one model out of N candidates (N arms), and for each given question, if the chosen model can correctly answer it, the policy receives a positive reward of 1 and 0 vice versa.
* In each iteration, the selection is inherently associated with an expectation function to capture to linear correlation between the given question q and potential reward r, indicating likelihood of success by choosing model m.
### Accuracy-Cost Trade-Off for KBQA with LLMs
* Source: https://arxiv.org/pdf/2405.17337
