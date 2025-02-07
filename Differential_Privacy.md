# What is Differential Privacy?
* Differential privacy is a mathematical method of protecting individuals when their data is used in data sets. It ensures that an individual will experience no harm as a result of sharing their data. It is especially important for census and medical data.
## How Differential Privacy Works
* Differential privacy is meant to guarantee that adversaries cannot glean information about specific people by comparing data sets. This is achieved using mathematics, artificial intelligence, intelligent control, machine learning, and other technologies.
* If differential privacy is utilized correctly, an analyst should not know more about any individual after analyzing data, and any adversary should not have too different a view of any individual after having access to a database.
* In a more technical definition, differential privacy introduces randomness into a data set, and this must not alter the eventual analysis of the data.
## How the Math Behind Differential Privacy Works
* Differential privacy functions using various mathematical mechanisms....
  * The Laplace mechanism is a matehmatical formula that adds noise to a data set. It determines how much noise to add by examining the quantity of data in a set and adding enough noise to ensure differential privacy. This is a general-purpose way of achieving differential privacy, and is an additive noise mechanism.
  * The Gaussian mechanism (additive) adds noise based on the Gaussian probability distribution, factoring in sensitivity and privacy parameters.
  * The exponential mechanism doesn't add noise to a data set, instead drawing an output from a data set using a probability distribution (data is drawn at random to accurately answer queries).
## Data that Should Be Kept Invariant in Differential Privacy
* Some data must be kept invariant, or unchanged, with differential privacy for analysis to remain accurate (EX: virtual reality headsets collect biometric eye-tracking data to function, and the movement data must remain invariant). Population counts must also stay accurate in census data.
* In general, the data that is preserved is vital to achieving the results a particular data set is intended to provide.
## When Differential Privacy is Most Useful
* Differential privacy is most useful in cases where sensitive data is involved (big data analytics of 5G technology, medical records in a major hospital).
* Differential privacy protects user data being traced back to individual users. The parameters involved are known as the privacy budget. This is a metric of privacy loss based on adding or removing one entry in a data set.
* Ensuring that the removal of a single data point cannot alter the overall data protects users. Differential privacy can help a user feel comfortable answering census questions or allowing a hospital to share medical information for scientific use.
* The Laplace and exponential mechanisms are the most common formulas for achieving digital privacy.
* Another common approach is based on importance weighting, which computes weights for an existing data set with no confidentiality issues and makes these weights analogous to a private data set. Statistical queries are answered approximately in a way that provable maintains information privacy.
## Circumstances that Warrant the Use of Differential Privacy
* Differential privacy is more effective than simply removing PII to anonymize data. The issue with anonymization is that richness of the data still allows an adversary to discover an individual within the data set (EX: a graduate student compared voter registration records with anonymized medical records and found the records belonging to the governor, meaning anonymous medical records are no longer released this way).
## How to Implement Differential Privacy
* Differential privacy is used in tandem with technologies to develop secure databases across a variety of fields. Privacy concerns raised by smart cities and other Internet applications could be solved by differential privacy, but implementing it across interconnected devices can be challenging.
## Challenges and Limitations of Differential Privacy
* Using Laplace noise helps disguise fata, but too much noise renders data useless for analysis. Differential privacy is designed for interactive statistical queries made to a database, so the presence/absence of one user's data should not impact the results of a query to the data set. Small amounts of noise are needed for averages and conclusions, returning specific records requires more noise.
* Using differential privacy for census data is challenging because census data does not require masking respondent characteristics, but data users must not be able to identify individual respondents.
* Microdata (individual data from real people) can challenge differential privacy (smaller dataset, easier to identify individual people). Continuous data collection also raises issues (if the differentially private data is published in multiple instances over time, knowing when a user's data was added allows adversaries to pinpoint their data.

## Steps Needed to Implement Differential Privacy
* Mathematicians designing and implementing differential privacy algorithms weigh accuracy of data against privacy of the users. The amount of noise added to a data set depends on how sensitive this information is. Granularity of data must be maintained, especially for data sets with diverse use cases.
* Machine learning algorithms for instance cannot encode general patterns without also gathering specific user data. The Model Agnostic Private learning method provides models access to a limited amount of public data.
* For differential privacy to have value, privacy must be rigidly defined. The mathematics allow for universal use of the term privacy when it comes to protecting user data.
## Use Cases for Differential Privacy
* Emerging technologies make data collection and data traffic more common. For instance, 5G connectivity makes it easier to collect statistical information from user behavior, and differential privacy can be used to keep this data relevant while protecting the data user.
* The mathematical framework of differential privacy is useful for machine learning, deep learning algorithms, the neural network, and more.
## Advantages of Differntial Privacy Over Other Privacy Measures
* Differential privacy protects the identities of the users within data sets. With differential privacy, privacy loss is controllable and can be measured. Differential privacy is more resistant to privacy attacks on the basis of auxiliary information, or information from separately available data sets. In general, differential privacy protects personal data far better than traditional methods while enabling data mining and statistics queries.
* Source: https://digitalprivacy.ieee.org/publications/topics/what-is-differential-privacy#:~:text=At%20its%20roots%2C%20differential%20privacy,a%20result%20of%20providing%20data.

# Differential Privacy in AI
* Differential privacy (DP) is a framework for measuring the privacy guarantees provided by an algorithm. The two main types are global (GDP), which applies noise to the output of an algorithm that operates on a dataset (like a query or model) and local (LDP), which applies noise to each individual data point before sending it to an algorithm, such as a survey or a telemetry system.
* Differential privacy can be used in many fields; it can protect the privacy of patients' medical records while enabling researchers to analyze them for insights. Differential privacy can also be used to protect the privacy of students' test scores while allowing educators to evaluate their performances.
* The term "differential privacy" reflects the idea that the privacy guarantee of an algorithm should depend on the difference between two neighboring datasets, which differ by only one data point (presence or absence of any individual in the dataset should not affect the output of the algorithm significantly)
## How do you Prove Differential Privacy?
* To prove differential privacy, you need to show that the probability of an algorithm producing a certain output is bounded by a function of the difference between two neighboring datasets and a privacy parameter. The mathematical function is....
  *  Pr[A(D)=O]≤eϵPr[A(D′)=O]+δ
  *  A is the algorithm D and D' are neighboring datasets, O is the output, ϵ is the privacy parameter, δ is the probability of failure
## Advantages of Differential Privacy?
1. It provides a rigorous and quantifiable measure of privacy that is independent of the adversary's prior knowledge and computational power
2. It is robust to composition, meaning that the privacy guarantee of multiple differentially private algorithms can be combined using simple rules
3. It is flexible and adaptable, meaning that it can be applied to various types of data, algorithms, and scenarios.
## Gaussian vs Laplace Differential Privacy?
* Guassian and Laplace differential privacy are two common mechanisms for achieving differential privacy by adding noise to the output of an algorithm. While Gaussian differential privacy adds Gaussian noise (follows a normal distribution with mean zero and standard deviation proportional to the sensitivity of the algorithm and privacy parameter), Laplace differential privacy adds Laplace noise, which follows a Laplace distribution with mean zero and scale parameter proportional to the sensitivity of the algorithm and the privacy parameter.
* Gaussian differential privacy provides (ϵ,δ)-differential privacy, where δ is the possibility of failure, while Laplace differential privacy provides ϵ-differential privacy, where δ is zero.
## What is Differential Privacy Image Classification?
* Differential privacy image classification is the task of training a machine learning model to classify images while preserving the privacy of the images and their labels. This can be done by applying differential privacy to the training algorithm, such as stochastic gradient descent (SGD), which updates the model parameters using noisy gradients computed on batches of images.
* Differential privacy image classification protects the privacy of sensitive images (faces, medical scans) while enabling useful applications (face recognition, biometric authentication, or medical diagnosis).
## What is the Promise of Differential Privacy?
* Data can be analyzed and used for beneficial purposes (research, innovation, personalization), but the privacy and dignity of the individuals in the data is protected. Differential privacy offers a clear and meaningful definition of privacy, a rigorous and quantifiable measure of privacy loss, and a flexible and adaptable framework for designing privacy-preserving algorithms.
## Examples of Differential Privacy in AI
1. _Apple_: Apple uses differential privacy to collect and analyze data from its users' devices, such as keyboard usage and health metrics, while protecting their privacy and identity. Apple claims that it does not see or store the raw data, but only aggregates the noisy data to improve its products and services, such as Siri and Healthkit
2. _Google_: Google uses differential privacy to collect and analyze data from its users' web and app activity, such as Chrome usage and Youtube views, while also protecting privacy. Google claims that it does not link or combine the noisy data with other data, but only uses it to improve its products and services, such as Chrome, YouTube, and Maps.
3. _Microsoft_: Microsoft uses differential privacy to collect and analyze data from customer devices (Windows, XBox Gaming) and again doesn't store the raw data.
## Related Terms
1. _Privacy Parameter_: the privacy parameter, denoted by ϵ is a measure of the privacy loss incurred by a differentially private algorithm. It controls the amount of noise added to the output of the algorithm. A smaller ϵ means more noise and more privacy, while a larger ϵ means less noise and less privacy
2. _Sensitivity_: the sensitivity, denoted by Δf, is a measure of the maximum change in the output of a function f when applied to two neighboring datasets. It determines the scale of the noise added to the output of the function to achieve differential privacy. A higher sensitivity means more noise and more privacy, while a lower sensitivity means less noise and less privacy.
3. _Noise_: N is a random variable that follows a certain distribution, such as Gaussian or Laplace, and is added to the output of a function to achieve differential privacy. The noise obscures the influence of any individual data point on the output of the function, making it statisically indistinguishable from the output of the function applied to a neighboring dataset.
* Source: https://clanx.ai/glossary/differential-privacy-in-ai#:~:text=Differential%20privacy%20(DP)%20is%20a,train%20models%20on%20private%20data.

# Low-Rank Adaptation Overview
* Low-rank adaptation (LoRA) is an adapter-based technique for efficiently fine-tuning models. A low-rank matrix is added to the original matrix. An adapter, in this context, is a colletion of low-rank matrices which, when added to a base model, produces a fine-tuned model. It allows for performance that approaches full-model fine-tuning with lower space requirements.
* A language model with billions of parameters may be LoRA fine-tuned with only several millions of parameters.
* LoRA support was integrated into the Diffusers library from Hugging Face, and support for these techniques is also available through Hugging Face's Parameter-Efficient Fine-Tuning (PEFT) package.
* Source: https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)#Low-rank_adaptation

# What is Low-Rank Adaptation (LoRA)?
* Low-rank adaptation is a technique for quickly adapting machine learning models to new contexts. It adds lightweight pieces to the original model, as opposed to changing the entire thing, allowing developers to expand the model's use cases.
## What does LoRA do?
* LLMs like ChatGPT take a lot of time and resources to set up. They may have trillions of parameters, allowing them to be generally accurate but not specifically fine-tuned to carry out specific tasks.
* Getting a model to work in specific contexts can require a great deal of retraining, which could be expensive and time-consuming. LoRA provides a quick way to adapt the model without retraining it.
* Instead of completely retraining a model from start to finish, LoRA adds a lightweight, changeable part to the model so that it fits the new context. This is faster and less resource intensive.
## How does Low-Rank Adaptation Impact a Machine Learning Model?
*  A machine learning model is the combination of a machine learning algorithm with a data set. The result of this combination is a computer program that can identify patterns, find objects, or draw relationships between items even in data sets it has not seen before.
*  To generate text, models draw from a lot of data and use highly complex algorithms. Slight changes to the algorithms or the data set mean that the model will produce different results, but getting specific ones may require a lot of retraining.
*  LoRA freezes the weights (which determine how important different types of input are) and parameters of the model and adds a lightweight addition called a low-rank matrix, which is then applied to new inputs to get results specific to the context. The low-rank matrix adjusts for the weights of the original model so that the outputs match the desired use case.
## What is a Low-Rank Matrix in LoRA?
* For LoRA, the important thing to understand is that low-rank matrices are smaller and have fewer values, so they take up less memory, require fewer steps to add or multiply, and are processed faster.
* LoRA's low-rank matrices contain new weights to apply to the model when generating results, and that process alters the outputs that the model produces with minimal computing power and training time.
* Source: https://www.cloudflare.com/learning/ai/what-is-lora/
