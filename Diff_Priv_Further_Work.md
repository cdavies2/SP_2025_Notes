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
* First, the accuracy expectation of one cluster (LLMs or KGMs) is calculated for better interential performance
* Second, a context-aware expert distinguishing is designed to present the question embedding to the policy and find the best arm with implicit expertise
* Finally, decision-making is constrained with a cost regret, punishing models that waste more on failures.
### Accuracy-Encouraged Cluster Expectation
* Thompson Sampling iteratively samples the best arm based on historical success and failure data.
* This evaluation allows selections to evolve over iterations based on succsesses and failures, slowly converging towards optimal selections as more questions are encountered.
* _Definition: conjugate prior or a and b_: given a cluster, let ac and bc denote the success and failure respectively for c. The pair of (ac, bc) forms a conjugate prior. It facilitates the efficient update of fundamental beliefs about the performance of each cluster during selections.
* If a cluster has been sufficiently selected and performance is satisfying with more success times, the corresponding expectation associated with 0c is more likely to be higher to facilitate a reliable exploitation.
* The cluster with the largest 0kc in iteraation k as the best cluster c for current question q, where the arm models within the particular cluster will have higher chances of answering the question.
* _Definition: Selection Regret_: in iteration k, they denote the real-selected arm as ak, the annotated best arm as a*k, which can answer the question with the lowest costs. We refer to the expectation differences between ak and a*k as the selection regret for current iteration k.
* In general, bounds are obtained by establishing upper and lower confidence bounds as explicit functions of the arm ak and history Hk-1, denoted as U(ak, Hk-1) and L(ak, Hk-1).
### Cost Regret Constraint
* When measuring the proportion of costs incurred by incorrect predictions given by the arm a within a budget constraint, qBa and Qba indicate the failure (question and historically assigned questions for arm a that are wrongly answered). p(a) is the unit cost for invoking LLMs.
* For black-box LLMs, cost is token-level fees, while for open-sourced LLMs, it indicates the number of times LLMs are called.
* The constraint can ensure that the total cost of selecting arm a for answering questions in the previous k-1 iterations and the current iteration k does not exceed a predefined budget b. To control the impacts of over-constraint from cost regret which may penalize a model for costs on necessary trials, a hyperparameter can control the trade-off between maximizing rewards and minimizing cost regret for reasonable exploration and exploitation.

## Experiments
* The experiments were conducted on three domain-specific datasets; CommonsenseQA, Openbook QA, and MedQA-USMLE.
* Baselines from three aspects (fine-tuned Language Models, KGMs, and API series and local series of LLMs) are considered.
* Using Coke, they were able to achieve 2.03%, 0.67%, and 1.03% improvements over GPT-4 on three datasets.
* On CSQA and OBQA, the accuracy of state-of-the-art KGMs are very close to GPT-4 and much better than GPT 3.5 and local series of LLMs.
* On MedQA, where questions are over-complicated and open-ended for KGMs to infer, there is a massive performance gap between the best KGMs and all LLMs. As such, they had to rely on sufficient calls of LLMs to ensure accuracy, increasing costs.
### Budget Study
* To bserve when accuracy and cost reach a balance, the budget is decreased from 1 to 0.5 until Coke has a higher error rate than GPT-4. Nodes in the lower left positions imply better performance considering both higher accuracy and lower costs. Coke would achieve comparable performance to GPT-4 within around 60% budget on both CSQA and OBQA datasets.
* Purely relying on LLMs loses the opportunities to leverage KGMs to further boost accuracy. On the other hand, more budget inevitably increases token-level costs. Coke outperfroms GPT-4 on all datasets with higher accuracy as budget increases.
## Related Work
* Contemporary research often focuses on generating responses through deductive processes applied to knowledge graphs, as outlined in surveys and studies on the subject. Said systems predominantly leverage techniques rooted in semantic parsing and information retrieval. These methodologies often delinate the process of question interpretation and knowledge graph (KG) inference as distinct stages, exposing models to potential pitfalls associated with query phrasing.
* Recognizing the above limintation, recent studies picoted towards a more integrated approach to reasoning. MHGRN dunamically refines the embeddings of question context in tandem with the reasoning process, utilizing graph neural networks.
* The QA-GNN framework crafts a work graph where the context itself is instantiated as an entity and interlinked with relevant entities through cooccurrence, streamlining the reasoning process.

## Conclusion
* Coke is a cost-efficient strategy for LLMs in KBQA while balancing inferential accuracy and cost saving. Coke could integrate two sets of off-the-shelf models (LLMs and KGMs), and efficiently assign the most promising for each question within a limited budget by employing a tailored cluster-based Thompson Sampling and a contextual multi-armed bandit. Overall decision-making is bounded by cost regret, constraining the selection based on cumulative expenses incurred from model failures.
* Source: https://arxiv.org/pdf/2405.17337

# FinLoRA: Finetuning Quantized Financial Large Language Models Using Low-Rank Adaptation on GPUs
* LLMs' low-rank structures can be finetuned to domain-specific datasets, enhancing their performance and improving their applicability to specialized tasks.
* FinGPT applies low-rank adaptation techniques for finetuning quantized LLMs in financial contexts, displays noticeable improvement over the base model, while having substantial memory reduction and training speedup.
* Due to sensitive data and regulatory constraints, fine-tuning and inference of LLMs within local environments remain critical requirements for financial institutions. The ability to create personalized and customized LLMs, finetuned for specific tasks, is essential for maximizing the utility of models in financial applications.
* LLMs can be finetuned for diverse financial tasks locally and cost-effectively using widely accessible GPUs, achieving notable improvements over baseline models. This project...
  * Employs Quantized Low-Rank Adaptation (QLoRA) to alleviate GPU memory requirements and achieve more efficient finetuning. The number of trainable parameters required for finetuning is reduced, and quantization compresses model size, further reducing GPU memory consumption
  * Distributed data parallelism (DDP) and pipeline parallelism to leverage multiple GPUs effectively. Training data is distributed across GPUs to accelerate finetuning, and the model is partitioned at the layer level to optimize memory usage during inference.
  * Models finetuned with QLoRA exhibit a 48% average increase in accuracy compared to basline models, validating the effectiveness of low-rank adaptation and quantization in addressing unique challenges of FinLLMs
## Quantized Low-rank Adaptation
* Low-rank adaptation (LoRA) is a parameter-efficient finetuning method that incorporates a smaller set of trainable weights, denote the pretrained weights (W0), update weights (WBA), and trainable parameters( A and B).
* During the fine tuning stage, the forward pass can be expressed as
  * y = W0x +  ∆Wx = W0x + B Ax (W0 denotes pre-trained weights)
* During the inference stage, do not add A and B back to W0, perform...
  * y = W0x + B Ax
* Quantized LoRA further reduces memory usage by using 8-bit or 4-bit quantization. During finetuning, all weights of the pre-trained model are quantized to 8-bit or 40bit. They are dynamically de-quantized back to 16 bit when performing computation with the input sequence x and the adaptor matrix A and B, which remain in 16-bit precision throughout the process.
## Optimizing Finetuning Process
* Distributed Data Parallel (DDP) distributes the training data across GPUs. It launches one process per GPU, where each process gets its own copy of the model and optimizer. Each process receives different inputs, and gradients are computed and synchronized across all GPUs to accelerate training. DDP provides substantial speedup when multiple GPUs are available.
* Brain Floating Point (BF16) can achieve similar results as FP32 while having speedup and memory savings during finetuning.
* 0/1 Adam Optimizer is a modified version of said optimizer that linearizes each Adam step and allows utilizing 1-bit compression for faster convergence speed, while offering reduced data volume and higher training throughput.
## Optimizing Inference Process
* Inference on large-scale models such as Llama 3.1 70B demands considerable GPU memory resources, particularly when using higher precision like 8-bit or 16-bit.

## Experiments
* Experiments were conducted on a server equipped with four 16-core CPUs, 1 TB of RAM, and four GPUs with 48GB of memory. Llama 3.1-8B Instruct and Llama 3.1-70B Instruct are our base models.
* Three financial language processing tasks were focused on....
  * _Sentiment Analysis_: entails analyzing financial text (news articles, tweets) to assign sentiment labels (positive, negative, neutral)
  * _Named Entity Recognition_: identifies and classifies critical entities within financial texts (organizations, locations, individuals)
  * _News Headline Classification_: categorizes headlines according to predefined criteria or questions, facilitating automated organization and analysis of financial news.
* XBRL is a standardized format designed for exchanging financial information. For this, tagging and extraction are focused on.

 ## Datasets: Sentiment Analysis
 * _Financial Phrasebank_: contains sentences extracted from financial news and reports, which are annotated with sentiment labels.
 * _Financial Question-Answering Sentiment Analysis_: same labels as FPB, contains microblog headlines and financial news
 * _Twitter Financial News Sentiment_: comprises annotated tweets related to financial news labeled with sentiment categories.
 * _News with GPT Instruction_: comprises samples with seven labels ranging from "strong negative" to "strong positive", only three labels are used for simplicity
 * Headline classification dataset categorizes headlines into two classes, "yes" and "no", based on predefined questions
 * Named entity recognition (NER) annotates one entity per sentence (classified as a location, person, or organization)
 * For XBRL tagging, the FiNER dataset was processed so each question comprises of the sentence and one highlighted entity and the answer includes the correct tag.
 * The XBRL extraction dataset comprises questions and answers derived from XBRL filings from 2019 to 2023 for Dow Jones 30 companies. Tag and value extraction were performed on XBRL text segments.

 ## Implementation Details
 * For sentiment analysis, headline classification, named entity recognition, and XBRL Tagging, single-task fine-tuning was employed. Llama 3.1 8B with a batch size of 16 and Llama 3.1 70B with a batch size of 4 were used
 * For XBRL tag and value extraction, multi-task fine-tuning was adopted. Llama 3.1 8B with a batch size of 2 was used.
 * 8-bit quantized inference was used for all evaluations for consistency's sake.
## Performance Metrics
 * _Accuracy_: ratio of correct answers to total queries
 * _Weighted F1 Score_: used for classification tasks, calculated as the weighted average of F1 scores for each class, with weights proportional to the number of instances in each class.
### Finetuning and Inference Performance
* _Batch size_: batch size per GPU during finetuning
* _GPU memory usage_: sum of the amount of GPU memory used for all GPUs during training
* _GPU hours_: the product of total training time and number of GPUs used
* _Adapter size_: the size of the LoRA adapter file
* _Inference speed_: the number of seconds to process an example

## Results and Analysis
* The finetuned Llama 3.1 8B demonstrates noticeable improvements in accuracy compared to its based model and even surpasses results of the Llama 3.1 70B base model
* Even with lower quantization and rank 4, the finetuned Llama 3.1 8B model achieves comparable performance to its 8-bit, rank 8 counterpart.
* The fine-tuned 70B model demonstrates practical usability with 4-bit quantization, showcasing the feasibility of deploying larger LLMs for complex financial tasks in resource-constrained environments.
 ## Conclusion
 * Finetuning Llama 3.1 8B and 70B models on commodity GPUs resulted in up to 48% improvements in accuracy compared to based models on average across all tasks. This could be achieved with only four GPUs and less than 20 hours of training times per task
* Source: https://arxiv.org/pdf/2412.11378

# FinGPT-HPC: Efficient Pretraining and Finetuning Large Language Models for Financial Applications with High-Performance Computing
* LLMs have a large number of parameters, and the finetuning process could still consume hundreds of gigabytes of GPU memory. Over 99% of said parameters come from the linear layers of the transformer structure, where computation workload and memory footprints grow quadratically with dimensions (EX: layer width). The model size and training time of Llama2-13B is about twice that of Llama2-7B.
## Challenges
* The pretraining dataset of many LLMs is too large to fit into GPU or CPU memory. During the training process, data is loaded into GPU memory from the disk, which is time intensive.
* LLMs require large GPU memory during both training and inference stages
* When using the model-parallel method to pretrain an LLM across multiple GPUs, these GPUs remain under low utilization.
* LoRA is a parameter-efficient finetuning method, utilizing low-rank structure to reduce the redundancy of the linear layers in the transformer structure.
* Quantized LoRA (QLoRA) converts the precision of parameters from half-precision (16-bit) into INT8 (8-bit) or INT4 (4-bit), which can reduce GPU memory consumption.
* This method for pretraining and finetuning LLMs for financial applications with high-performance computing has two parts...
  1. Pretrain a low-rank LLM on a general dataset and obtain a general LLM
  2. Finetune the LLM on domain-specific datasets by freezing the pretrained weights and adding a low-rank version on a parallel path.
* The low-rank structure and quantization technique is employed in the linear layers of the transformer structure, leading to a substantial reduction in the number of trainable parameters, reducing GPU use, memory, and training time.
* A recomputing technique reduces the peak number of intermediate variables so the GPU memory footprint of pretraining and finetuning LLMs are greatly reduced. Pretraining is performed across multiple GPUs
* For pretraining GPT2-1.5B, the method resulted in a speedup of 1.3x, and a compression ratio of 2.64x without accuracy drop.
* For finetuning Llama2-13B, the method resulted in a 6.3% accuracy inctrase in general tasks and 24.0% in financial tasks.
## Large Language Models
* The transformer structure uses the self-attention mechanism and allows the model to process sequential data efficiently. It is the base model of ChatGPT, which was trained using instruction finetuning.
* The scaling law of LLMs states that model performance scales as a power-law with model size, dataset size, and the amount of compute used for pretraining.
## Low-Rank Structure
* The Low-Rank adaptation (LoRA) method reduced the number of trainable parameters by a factor of 10,000 times, but still retained a full-size pretrained LLM for inference.
* Tensor layers are suited for computing on GPUs and have the low-rank feature.
## Quantization Technique
* Quantization maps high-precision data to low-precision data, which can effectively reduce the memory consumption. Outliers make activisions difficult to quantize and have less impact on the weights. Accuracy loss can be reduced via reducing the effect of outliers and improving the rounding strategies.

## Background on LLMs and Low-Rank Structure
* Most existing LLMs are based on the decoder-only transformer structure. There are 3 parts....
 * An embedded layer that converts the input x to a latent vector x^e
 * There are N decoder layers, each one having two modules: the multi-head self attention module and the feed-forward network module. The self-attention module transforms x^e to x^o and the feed-forward network module transforms x^o to x^d. The inputs x^e and x^o are skip-connected to the output x^o and x^d, respectively. A normalization layer normalizes x^e and x^o.
 * There is an output layer W^H that converts the latent representations x^d to y. Then, the softmax function transforms y to a vector of probabilities, where each entry is the probability of picking the corresponding token.
* Linear layers are major building blocks of the transformer network. An input sample x is multiplied by a weight matrix W, and the output y can be represented as...
 * y = Wx
* Linear layers are highly redundant, meaning W can be replaced with two sublinear layers A and B, so y becomes...
 * y = BAx
* The layer size is reduced from n^2 to 2nr, as is the number of multiplications.
## Challenges in Training and Inference
* Pretraining has three main challenges
 * Large scale dataset is needed to pretrain LLMs. For example, the dataset used to pretrain Llama2 is 4.7TB, requiring large amounts of GPU and CPU memory.
 * The memory footprint of pretraining and finetuning exceeds the GPU memory capacity. The GPU memory footprint can be classified into four parts. For pretraining Llama2-7B with batch size 16, the GPU memory footprint has a size of 1664 GB.
 * In the model-parallel method, an LLM is split over multiple GPUs. During the pretraining, the forward pass is computed layer by layer and only one GPU is working at a time. When splitting an LLM into N GPUs, each has a relatively low utilization of 1/N.
* Inference is when the trained LLM is loaded into the GPU. Using a standard tokenizer, the input text is processed into a sequence of tokens, the input is embedded, for h heads in multi-head self-attention module, there is an attention function to calculate attention score and head softmax. The heads are concatenated, and the last column of output is the predicted token.
* Challenges of the inference stage include....
 * Limited memory capacity poses a challenge in using LLMs for various applications, such as inference on mobile phones (EX: Llama2 is too large for that)
 * Most parameters of the transformer structure come from the linear layers. In the inference stage, each parameter requires multiplication and addition. For generative LLM, each generated token needs an inference, that token is added to the end of input tokens as a new input for the next inference (if the input has 100 tokens and the LLM generates 100 tokens, there are 100 times inference processes, these processes have 199 tokens as input, so the whole inference stage has 14950 tokens to be computed). This is not easily accomplished with a mobile phone's computational power.

## Pretraining Methods with Low-Rank Structure
* Pretraining an LLM has high costs and long training time, and low-rank structure can reduce the number of parameters. Using low-rank structure, pretraining involves....
  1. Replace the linear layer of the transformer with two sub-linear layers, which use a hidden layer with a small width.
  2. Decompose the pretrained weight matrix into low-rank matrices, then use matrices A and B to initialize a low-rank LLM.
  3. Adding two narrow linear layers to the parallel path of the pretrained linear layer, obtain new linear layers. The pretrained linear W are frozen, and the pretrained LLM only has the low-rank matrices.
## Optimization for the Pretraining Stage
* Intermediate variables are generated in the forward pass, used in backward pass, and then released.
* Some intermediate variables have more GPU memory consumption. This is because other variables are in the decoder layer, which has 32 layers, and some variables have the same size for each head (again, there are 32 for each decoder layer).
* By avoiding storing intermediate variables and recomputing them when needed, the GPU memory consumption can be significantly reduced, at the cost of more computations. The process of this method is...
  1. In forward pass, only the input x^e is stored
  2. In backward pass, the GPU recomputes other variables using x^e for a decoder layer
  3. Calculating the gradients of parameters in this decoder layer, the variables are deleted after obtaining gradients
  4. Repeat steps 1-2 under all decoder layers calculate the gradients
  5. Obtaining all gradients, and the parameters are updated.
* For pretraining an LLM on N GPUs, the model-parallel method loads the parameters on different GPUs, the computing is performed sequentially on said GPUs, and each GPU has a utilization of 1/N. In the pipeline model-parallel method, the input data is split into M >=1 minibatches, which are input into GPU-0 to GPU-3 sequentially and backward pass is performed in a reverse order.
* Decomposing weight at different linear layers is independent. In a single GPU, the decomposition of weights is batched in different layers, but on different GPUs, we perform the decomposition of different weight matrices, like W^Q and W^V in parallel.
* Source: https://arxiv.org/pdf/2402.13533

# ST-DPGAN: A Privacy-Preserving Framework for Spatiotemporal Data Generation
* Spatiotemporal data encompasses sensitive information, like how social media data often contains a person's browsing history, traveling details, and social network changes. This raises significant privacy concerns regarding public availability and dissemination of spatiotemporal data.
* Many regions have new data protection regulations because of privacy concerns. As such, it is important to find a de-identification method when we want o publish more fine-grained spatiotemporal datasets.
* To enable large-scale spatiotemporal data analysis, we have to provide a privace guarantee. GAN (generative adversarial network) has profound potential in spatiotemporal data generation.
* Some challenges in generating spatiotemporal data using GANs with differential privacy include...
  1. Spatiotemporal data are heteogenous. The generated spatial information and temporal information are hard to align into spatiotemporal data as it is demonstrated that spatial and temporal features belong to different Euclidean spaces.
  2. Due to the space-time duality property of spatiotemporal data, generating spatiotemporal data samples with a single variable drawn from a certain distribution is difficult.
  3. Simply adding noises to each spatial location or temporal point cannot guarantee a privacy-preserving dataset for training usage because it does not consider spatial correlation and temporal correlation, and the overall data distribution cannot be maintained.
* This new framework offers two main strategies for effectively extracting the distribution of spatiotemporal data and measuring privacy loss during the training process..
  1. Adopt the Transposed One-Dimensional Convulution layer (transConv1D) to generate fake data samples from Gaussian noises, which can transform single-channel Gaussian noises into spatiotemporal data.
  2. Simplify existing spatiotemporal graph convolutional networks and introduce temporal attention block and spatial attention block, which are designed to model spatial and temporal dependencies, enabling simultaneous extraction of spatiotemporal features from graph-based time series data and facilitating easier convergence during the training process.
 ## Differential Private SGD
 * Source: https://arxiv.org/pdf/2406.03404
