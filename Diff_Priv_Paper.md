# Differentially Private Low-Rank Adaptation of Large Language Model Using Federated Learning
* Federated learning allows for decentralized fine-tuning without exposing raw data to central servers. Despite avoiding raw data exposure, there is a risk of inferring sensitive information from model outputs, and federated learning for LLMs incurs notable communication overhead.
* DP-LoRA is a novel federated learning algorithm tailored for LLMs, which preserves data privacy by employing a Gaussian mechanism that adds noise in weight updates, maintaining individual data privacy while facilitating collaborative model training.
* DP-LoRA also optimizes communication efficiency using low-rank adaptation, minimizing the transmission of updated weights during distributed training. Experimental results across medical, financial, and general datasets demonstrate DP-LoRA effectively ensures strict privacy constraints while minimizing communication overhead.

## Introduction
* Employing general-purpose LLMs in specialized domains like medicine may have poor results because of the inherent discrepancy between general and domain-specific texts. OpenAI released ChatGPT fine-tuning API, which allows users to tailor ChatGPT by fine-tuning it with their own dataset, allowing it to be used for more specific applications.
* There are many concerns regarding potential leakage of private data while utilizing LLMs (especially in high-stakes domains like hospitals). An important question to ask is....
  * _How can we guarantee data privact in LLM fine-tuning with a practical federated learning approach, enabling each stakeholder to securely contribute data and derive collective advantages from an improved LLM?_
* Challenges include...
  * Despite federated learning ensuring no direct exposure to raw data, LLMs can accidentally disclose sensitive information through their responses, so there is a risk for attackers to potentially deduce training data from model outputs
  * Naively applying federated learning to fine-tune LLMs can trigger significant communication overhead (model updates from diverse decentralized sources need iterative aggregation, and frequent transmission of updates for a complex model like LLMs is extremely expensive).
  * These methods alleviate communication bottlenecks, but they cannot ensure concrete guarantees in protecting the privacy of the training data.
* Differentially Private Low-Rank Adaptation (DP-LoRA) ensures data privacy by employing a Gaussian mechanism, introducing random noise in weight updates to ensure minimal changes in publicly visible information when an individual in the dataset changes (preventing any one sample from having a significant impact and making it harder for attackers to infer private information corresponding to specific individuals).
* DP-LoRA addresses communication overhead by leveraging the low-rank characteristic inherent in LLMs and incorporating a smaller set of new weights into the model, fine-tuning solely on those parameters, only transmitting them, and reducing communincation costs.

## Privacy Issues of LLMs
* Current privacy efforts related to LLMs tend to either focus solely on the pre-training phase of LLMs or introduce substantial communication overhead due to a large number of weight updates, significantly limiting the algorithm's practical application.
* Said efforts mainly focus on general text rather than domain-specific datasets, so DP-LoRA aims to ensure privacy fine-tuning with little communication overhead.
### From General-Purpose LLMs to Domain-Specific LLMs
* GPT-4, BERT, and LLaMA all have strong language comprehension abilities. There has also been a growing trend in crafting LLMs specialized for particular domains.
* Med-PaLM is a medical-specific LLM refined through instruction tuning using medical benchmarks, showcasing performance akin to clinicians, and BioGPT is a Transformer language model specifically pre-trained on expansive biomedical literature.
* In the finance domain, BloombergGPT is trained on financial and general datasets, and it performs financial tasks like forecasting and risk assessment, but it is closed-source, resulting in a need for other cost-effective methods of domain adaptation.
* FinGPT aims to democratize access to vast Internet-scale data for training financial LLMs by developing diverse data pipelines to acquire and refine high-quality training data
* Privacy concerns are amplified in high-stakes domains like finance and medical science.
### Parameter-Efficient Tuning of LLMs
* Some widely-recognized fine-tuning methods include utilization of adapters, which embed a compact module within transformer blocks, allowing selective updates while keeping other parameters fixed. Low-rank adaptation (LoRA) integrates trainable rank decomposition matrices into transformer blocks, facilitating updates solely to these matrices while preserving the remaining parameters. These methods do not cosnider privacy however.
### Federated Learning and Differential Privacy
* Federated Learning focuses on safeguarding dataset privacy during distributed training. Early implementations involved employing model averaging to minimize communication overhead. Recent updates introduced methods in homogenous and heterogenous settings.
* The integration of differential privacy into federated learning offers a theoretical framework to govern privacy boundaries and manage the tradeoff between privacy and convergence. Low-rank adaptation might help mitigate privacy costs.

## Background of Differential Privacy
* _Deep neural networks_-starting with a single-layer network, which takes an input vector and transforms it into an output vector via a weight matrix. Trainable parameters are weights and a bias vector. A loss function is used to evaluate the difference between the ground-truth labels and the output.
  * During training, each iteration uses a batch to update the parameter 0, where each batch contains B sample. The parameter 0 is updated by the gradient descent method....
  *  𝒈 = 1/𝐵∇𝜽 ∑︁ L(𝜽;𝒙𝑏,𝒚𝑏),
                𝑏=1 
     𝜽 =𝜽 −𝛾 ·𝒈,
  * Where 𝛾 is the learning rate. The training process executes for T iterations
* _Federated Learning_-consider distributed training of a deep neural network with parameter 0. There are K nodes participating in distributed training. Each node k performs the gradient descent using its private dataset Dk, which contains Nk samples.
  * One distributed training iteration involves....
    1. A server broadcasts a global model 0 to all K nodes
    2. Node k initializes 0k by 0 and updates 0k by (3), using a batch of B samples from Dk.
    3. The server receives {01,..., 0k) from all K nodes and updates 0 by
    4.   𝜽 = ∑ 𝜌𝑘 · 𝜽k
             𝑘=1
  * 𝜌𝑘 is the weight factor, and the training process executes for T iterations
* _Differential privacy_-distributed training has the privacy concern that some attackers will monitor communication between the server and nodes and use that to infer dataset information. Differential privact considers the divergence of two distributions.
  * A (𝜖,𝛿)-differential private mechanism M is defined as....
  * Pr[M(𝑑) =𝑜] ≤ 𝑒^𝜖 Pr[M(𝑑′) =𝑜] + 𝛿
  * d and d' are two adjacent inputs and o stands for the output of mechanism M. The divergence is bounded by 𝜖, and a relaxation of 𝛿 is allowed, so randomness is introduced.
  * DP-SGD adds Gaussian noise into the gradient, making each iteration differentially private.
* The training process can be seen as a composition of T differentially private algorithms

## Differentially Private Low-Rank Adaptation (DP-LORA) of LLMs
* For a differentially private algorithm with a large language model and a federated learning scenario, we first elaborate on the privacy guarantee in a non-balance federated learning of our methods. Second, the method mitigates the communication overhead in LLM weight transmission.
### Motivation
* Organizations like Hugging Face collaborate in training large language models in vertical domains, like medicine and finance. Each institute in the organization processes a local dataset for collaborative training. Said collaborations often involve training existing LLMs with data from a specific vertical domain.
* Each participating institution usually possesses sensitive, private data. Due to the constraints of privacy laws, this data cannot be shared among collaborating entities.
* While federated learning has been widely studied for these scenarios, recent studies have discovered that it is not safe enough to protect private data in LLMs.
* DP-LoRA is proposed as a cooperation scheme for large language model federated learning.
### DP-LoRA Algorithm
* There are K nodes, where every node k holds a local dataset Dk and a large language model. For the dataset, Dk has Nk samples and is not shared with the server or other nodes.
* N denotes the total number of samples, N= 𝐾
                                            ∑︁ 𝑁k
                                            𝑘=1
* For the large language model, every node has the same model weight 0 in the beginning. 0 includes the original LLM weight W and the low-rank adaptation weight Ak and Bk.
* Given the noise scale �, clipping norm C, number of iterations T, learning rate y, and batch size be, a DP-LoRA algorithm for federated learning does as so
  1. _Broadcast_-the server sends the global model parameters 0 to all participating nodes. This is the starting point for training in each round.
  2. _Local Training_-each node k samples B samples from Dk and uses the clipped gradient to update the local weight Ak and Bk.
  3. _Add Noise_-a Gaussian noise nk ~ N(0, �^2) is added to Ak and Bk simultaneously. The noise weight is denoted as ~Ak = Ak +nk and ~Bk = Bk + nk
  4. _Upload_-every node k uploads ~Ak and ~Bk to the server
  5. _Aggregate_-the server aggregates the ~Ak and ~Bk from all nodes and takes the weighted average  𝐾                            K
                  ∑︁ �k~Ak      and            ∑︁ �k~Bk
                  𝑘=1                          𝑘=1
     to updates A and B
  * First, calculate the gradients of the loss function for the parameters Ak and Bk, denoted as gAk, gBk.
  * Second, perform gradient clipping by scaling down the gradients if their L2 norm is greater than C, gAk <- gAk/max(1, (||gAk ||2)/C) and gBk <- gBk/max(1, (||gBk ||2)/C)
  * Third, introduce Gaussian noise to the clipped gradients to ensure differential privacy
  * Fourth, update the local model parameters via gradient descent, 𝑨𝑘 ← 𝑨𝑘 −𝛾𝒈𝑨𝑘 
and 𝑩𝑘 ← 𝑩𝑘 −𝛾𝒈𝑩k, where gk is the noised and clipped gradient
### Guarantee of Differential Privacy
* For a single node k, Gaussian noise is added in the gradient where Ak and Bk are updated. The probability density function for Gaussian noise is N(0, �^2C^2I), where I is an identity matrix and has the same shape with Ak and Bk.
* The clipping bound C is used to ensure the gradient norm ||gAk||< C and ||gBk||< C. The noise scale provides the privacy guarantee for each iteration, which should follow  𝜎 ≥ 
√︁ 2ln (1.25/𝛿)/e. ~Ak and ~Bk denotes the noise weight from node K.
* For the server, it aggregates all ~0k uploaded from the server by Eq(8), we provde that the differential private property holds for the proposed algorithm, which guarantees the privacy for the training dataset Dk.
### Reducing Communication Overhead in Federate Learning
* Typically, a language model with weight W is composed by L weight matrices W^l ∈ R𝑛×𝑛,𝑙 = 1,2,..., l. Here, we consider the linear layers, and the input of output shares the same dimension n. These weight matrices have a low "intrinsic dimension". The matrix W^l is decomposed into A^l and B^l, followed by low-rank adaptation method.
* The weight 0 is composed by A and B. Federated learning involves collecting local model weights for T iterations and redistributing the aggregated weights 0 back to each participating node. It is assumed that all nodes utilize Algorithm 1 for training and share identical hyper-parameters. The system comprises K nodes in total, each one possessing its private dataset Dk. A central server maintains the global model weights 0.
* Reduction in communication overhead is assessed using the low-rank adaptation method. Assuming the federated learning process spans T rounds in a noiseless and ideal network environment, the commpunication overhead can be quantified in two steps...
 * Freeze out most of the model weight, fine-tuning it in self-attention layers. For instance a Llama-7B has 32 transformer layers and 32 attention heads, which have 4096 dimensions, so each self-attention block contains 4096* 4096 *3=50,331,648 parameters, and 32 transformers contain over a billion parameters total. Thus the training parameter is reduced from over six billion to around 1 billion.
 * The self-attention parameters are decomposed using low-rank adaptation methods (EX: each W^l is replaced with A^l nxr and B^l nxr), and an r to reduce by is choses
*  In general, communication overhead is determined by the number of nodes K, the training epoch T, and the transmission parameters 0. The total communication overhead is given by TxKxLxrx2n.
*  The overhead for Llama-7B with 5 nodes in 50 iterations is 2x5x50x6.7B, and the communication overhead for DP-lora is 2x5x50x4096x256.

## Performance Evaluation
### Tasks and Datasets
* To mimic protection of highly sensitive data, the algorithm was tested on financial data, medical data, and a normal dataset for comparison. The datasets were...
 * _SlimPajama_-open source dataset for training large language models, contains 627B tokens and comes from numerous online sources including Wikipedia and GitHub
 * _Medical Dataset_-contains 0.26 million English consultations between patients and doctors, each consisting of a description of the patient's medical condition and a conversation between the patient and the doctor. The consultations cover many categories (diabetes, pain management) and specialties (cardiology, nephrology)
 * _Finance Dataset_-data was collected from many sources, lincluding sites like Reuters, financial regulatory filings, and academic datasets.
 * The aforementioned datasets often lack truly sensitive information due to their open-source nature. The algorithm doesn't differentiate between sensitive and non-sensitive portions of data, instead focusing on protecting the local dataset as a whole. Users can apply the algorithm to data they deem sensitive.
### Experiments Setup
* For general tasks, three datasets are used (BoolQ, PIQA, and WinoGrande).
* In the financial domain, FPB, FiQA SA, and TFNS were used
* For medical tasks, MedQuAD, LiveQA Test, and MEDIQA-Ans were used.
* The algorithm was tested using GPT-2, Bert ased, ChatGLM-6b, and Llama2-7B
* Three federated learning algorithms were chosen as the baseline; Vanilla FL, T-FedAvg, and CMFL. The baselines and proposed method were trained in the same settings, the dataset was split evenly and stored locally among all K nodes, an average for aggregating local model weights was adopted, and 𝜌k is set to 1/k. We assume that communication channels are noiseless and ideal, and each node k uploads its 0k to the server in every iteration. We choose K as 5 and we set the maximum iteration T as 50.
### Results on Medical Tasks
* Generally, models decreased in performance with stricter privacy settings (lower e and � values). For instance, Llama-7B's performance on LiveQA drops from 69.4 at its original setting to 55.9 when e is reduced to 2.
* Some models like ChatGLM-6B show a more resilient performance under varying e values, decreasing far less when e is modified, showing some models may be better suited for privacy-sensitive operations.
* Overall, stricter privacy settings usually correlate with reduced performance.
### Reduction in Communication Overhead
* When testing performance of the global model and communication overhead in federated learning, the rank r was chosen in a decreasing manner. Consistenly, as rank r decreased, the number of parameters and communication overhead decreased. This suggests a direct correlation between rank reduction and communication efficiency. However, there is a trade-off observed in model performance on general tasks for the differential privacy taining.
* Llama-7B saw a much more stable performance than ChatGLM-6B

## Conclusion
* DP-LoRA ensures differential privacy while significantly reducing communication overhead. By introducing noise into gradients to achieve differential privacy, communication efficiency and security imporve. Future work will explore refining error bounds for differential privacy.
* Source: https://arxiv.org/pdf/2312.17493
                  
