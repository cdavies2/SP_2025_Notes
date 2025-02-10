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
