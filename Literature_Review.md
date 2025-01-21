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
### 
