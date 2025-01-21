# Journal of CyberSecurity and Privacy

## Article 1: Cybersecurity for AI Systems: A Survey
* Link: https://www.mdpi.com/2624-800X/3/2/10
* AI Attacks are broadly categorized as either manipulation or extraction attacks.
* Adversarial and evasion attacks can be launched by manipulating system input to obtain an unintended or dangerous outcome, and poisoning/causative attacks taint training data.
* When designing security into AI systems, you have to ask...
  1. What cyberattacks can AI systems be subjected to?
  2. Can attacks be organized into a taxonomy?
  3. What are possible defense mechanisms to keep AI systems from subjection to cyberattacks?
  4. Do any generic defense mechanisms against all AI attacks exist/is it possible?
### Literature Review
* After a search of Scopus, using keywords like "cybersecurity", "vulnerabilities", "threat" AND "AI", "Artificial Intelligence", and "Machine Learning", 1366 articles were found, which was narrowed down to 415 when specifying the computer science/engineering subject area.
* During the learning/training stage, an AI system can be subjected to manipulation attacks when it is presented with falsified data, which can be classified as legitimate data. Biometric profiles can also be generated to impersonate users by repeatedly querying a classifier and the learned profiles can attack other classifiers that were trained on the same datasets. Man-in-the-Middle attacks often use generative models for querying.
* During the inference or testing stae, extraction attacks use the feature vector of a model to invert or reconstruct it, and thus gain access to private input or training data
* Hansman and Hunt's classification of attack types includes four dimensions (atttack classes, attack targets, vulnerabilities/exploits used and whether the attack has a payload or effect beyond itself). Barreno, Huang, and Biggio categorize attacks based on the goal of the attacker, the attacker's ability to manipulate data/influence the learning system, familiarity of algorithms, data used by the defender, and attacker strategy (for instance, in data poisoning attacks, the hacker has control over the training dataset, can modify the data and impact the entire learning process, thus decreasing system performance, and increase the number of errors).
* This current article differs from some of the surveys it reviewed in that it focuses on attacks and defense mechanisms from various different phases of the AI system's development life cycle.
### AI Attacks and Defense Mechanisms
* Common modes of failure for AI systems are...
  * Need for better resources for self-upgradation of AI systems being exploited
  * Implementation of malicious goals that make the AI systems unfriendly
  * Flaws in user-friendly features
  * Use of different techniques to make different stages of AI free from the boundaries of actions expose the AI systems to adveraries
* Three main levels of AI Intelligence exist, "Narrow AI" needs help from humans, "Young AI" is a bit more intelligent than a human, and "Mature AI" has super-human intelligence.
* Types of failures fall into two categories, _unintentional_ (formally correct but unsafe behavior) and _intentional_ (caused by attackers working to destabilize a system by introducing new training data or stealing algorithmic framework)
* Some categories of unintentional failures include....
  * _Reward Hacking_: occurs when an agent has more return as a reward in an unexpected manner in a game environment, thus unsettling safety of the system (multi-step reinforcement where a reward function generates a discounted future reward, thus reducing the impact of immediate rewards could be helpful).
  * _Distributed Shift_: appears when an AI/ML model that did well in one environment performs terribly in another. This could be the result of a change in input (covariate) change in label marginal probability (label), or definitions of the label experiencing spatial or temporal changes (concept).
  * _Natural Adversarial Examples_: real-world examples that are not intentionally modified and result in significant performance degradation of machine learning algorithms.
* Intentional failures could have the goals of confidence reduction for the targeted model, altering output, and altering output labels (missclasification) 
* Some categories of intentional failures include...
  * _Whitebox attacks_: adversary has access to the model's parameters, including training data, and use that to find the model's vulnerabiities.
  * _Blackbox attacks_: hacker knows nothing about the system, only classifiers' predicted labels, and uses information about past inputs to understand model vulnerabilities.
  * _Non-Adaptive blackbox attack_: adversary has the knowledge of distribution of training data for the model, chooses a procedure for a selected local model, and trains the model on known data distribution to determine what was already learned and trigger misclassification.
  * _Adaptive blackbox attack_: adversary has no knowledge of traning data or model architecture
  * _Strict blackbox attack_: adversary doesn't have training data distribution but could have the labelled dataset collected from the target model
  * _Graybox attacks_: extended whitebox (adversary is partially knowledgable about model architecture but parameters are unknown) or blackbox (model is partially trained, has different archtiecture and parameters).
### Dataset Poisoning
* Error-agnostic poisoning relates to Denial of Service, causing miscellaneous errors, while error-specific poisoning aims for particular misclassification errors, resulting in integrity and availability concerns for data.
* Dataset poisoning attacks have two subcategories:
  * _Data Modification_: algorithm cannot be accessed, training data labels are updated/deleted
  * _Data Injection_: new malicious data is injected into the training set (above only impacts labels, this impacts the data itself).
* Based on Biggio, Nelson, and Laskov's adversarial strategy equation, manipulating around 3% of the training dataset can reduce the model's accuracy by 11%.
* Some data poisoning attacks can outsmart data sanitation defenses like anomaly detection. Simply giving a poisoned instance a clean label (like marking spam with the same features as typical Gmail) has an effect. Attacks fall into two main categories...
  * _High sensitive_: poison points are surrounded by other points, so its location is seen as benign
  * _Low sensitive_: anamoly detector drops all points away from the centroid by a certain distance, so whether or not a point is deemed abnormal does not vary much by addition or deletion of points. Attackers then optimize the location of poisoned points so it satisfies the defender's constraints.
### Poisoning Defense Mechanisms
1. _Adversarial Training_: inject instances created by an adversary in order to increase the strength of the model.
2. _Feature Squeezing_: reduces the number of features a model has, thus reducing both the complexity and sensitivity of the data.
3. _Transferability blocking_: introduces null labels into the training data and instructing the model to discard adversarial samples as null labeled data
4. _MagNet_: uses detector (measures distance between normal and poisoned data and a threshold) and reformer (autoencoder makes a tampered instance into a legitimate one)
5. _Defense-GAN_: General Adversarail Network uses a generator to reconstruct images, so genuine instances are closer to the generator than tainted ones
6. _Local Intrinsic Dimensionality_:
7. _Reject on Negative Impact (RONI)_: computationally expensive, not suited for deep learning/large datasets, possible malicious data points are relabeled based on the labels of neighboring samples in the training dataset.

### Model Poisoning
* Closer to traditional cyberattacks, 
## Applications of Machine Learning in Cyber Security: A Review
* Link: https://www.mdpi.com/2624-800X/4/4/45
* This paper is structured similarly to a narrative literature review, going over studies evaluating how well AI can address cybersecurity challenges
* 101 studies were selected, of which 99 were integrated/cited in this paper
* Appropriate datasets for use in ML and AI should...
  1. Include both audit logs and raw network data
  2. Provide a variety of modern attacks
  3. Represent realistic and diverse normal traffic
  4. Be labeled
  5. Comply with ethical AI principles and privacy protocols
  6. Be accepted by the scientific community
## Ethical AI and Compliance Aspects
* Ethical AI principles are very important regarding how machine learning (ML) models are developed and applied in a cyber security context. Said principles include....
  1. _Transparency_: being clear about a model's decision making process is integral in capturing why it gives specific predictions
  2. _Fairness and bias mitigation_: fairness is ensured when AI models (especially those involved in threat detection) do not discriminate against any group of malicious activities or any potentially relevant data source from which threats could arise. Bias can be addressed by utlizing more balanced datasets, allowing the system to recognize a large and diverse set of threats.
  3. _Privacy and security_: privacy-preserving techniques like differential privacy or federated learning ensure that individual data are protected. Privacy-centric cyber security detects malware via training models directly on user devices rather than centralized servers, so detection occurs without sacrificing of sensitive data.
  4. _Accountability with Auditability_: involves provision for audit trails/documenting any changes in model performance, allowing for regular reviews and consideration of complaints
  5. _Robustness and security against adversarial attacks_: models used in cyber security must also be formally resistant to adversarial attacks that work to manipulate underlying vulnerabilities in training data or model structure (can be achieved via adversarial training)
  6. _User-centric design and human oversight_: in ethical AI within cyber security, user-centric design must ensure human obersight is prioritized (humans can intervene if automation isn't sufficient)
  7. _Compliance_: frameworks like GDPR and NIST's AI risk management ensures legal usage and applications of ethical AI.

* Overall, Journal of Cybersecurity and Privacy seems to put great emphasis on the research that goes into each article, both articles state in the beginning how they found their sources and how they were narrowed down.

#
