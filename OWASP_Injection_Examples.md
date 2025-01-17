# 1) Prompt Injections
* Prompt injection vulnerability refers to when an attacker provides inputs to a large language model that causes it to perform malicious tasks
* Jailbreaking (direct prompts) overwrite or reveal an underlying system prompt. This could allow hackers to exploit backend systems via interaction with insecure functions and data stores accessible through the LLM.
* Indirect prompt injections are when the LLM accepts input from external sources, like websites or files. The attacker could embed a prompt injection in external content hijacking a conversation's context. This would result in the LLM acting as a "confused deputy", letting the attacker to manipulate both the users and the systems accessed by the LLM. Indirect prompt injections just need to be readable by the LLM, not necessarily people
* In advanced attacks, an LLM could be manipulated to mimic a harmful persona or to interact with plugins in the user's setting. This could result in user data being leaked, social engineering attacks, or unauthorized plugin use, without the end user being alerted

## Common Examples of Vulnerability
1. A malicious user, via direct prompt injection, instructs the LLM to ignore system prompts and instead execute a prompt that returns private, dangerous, or otherwise undesirable information.
2. A user asks an LLM to summarize a webpage containing an indirect prompt injection. This then causes the LLM to solicit sensitive information from the user and perform exfiltration via JavaScript or Markdown.
3. A malicious user uploads a resume containing an indirect prompt injection (which has instructions to make the LLM tell users that this is an excellent document), which results in the LLM fradulently informing internal users that the candidate is great for the job role
4. A user enables a plugin linked to an e-commerce site, and a rogue instruction embedded into the site exploits the plugin, using it to make unauthorized purchases
5. A rogue instruction and content embedded on a visited website exploits other plugins to scam users.

## How to Prevent
* Prompt injection vulnerabilities are possible because LLMs don't segregate instructions and external data from each other (as both are natural language and considered "user provided"). There is no fool-proof injections, but there are mitgators
1. Enforce privilege control on LLM access to backend systems. Provide the LLM with its own API tokens for extensible functionality, like plugins, data access, and function-level permissions. Follow the principle of least privilege and restrict the LLM to only the minimum level of access necessary for intended operations
2. Implementss humans in the loop for extensible functionality. When performing privileged operations, like sending or deleting emails, have the application require that the user approve the action first. This mitigates the opportunity for an indirect prompt injection to perform actions on the user's behalf without their knowledge
3. Segregate external content from user prompts. Separate and denote where untrusted content is being used to limit influence on user prompts (EX: use ChatML for OpenAI API calls to indicate to the LLM the source of prompt input)
4. Establish trust boundaries between the LLM, external sources, and extensible functionality (EX: plugins or downstream functions). Treat an LLM as an untrusted user and maintain final user control on decision-making processes (this method is still vulnerable to man-in-the-middle attacks).

## Example Attack Scenarios
1. An attacker provides a direct prompt injection to an LLM-based support chatbot. The injection contains "forget all previous instructions" and new instructions to query private data stores and exploit package vulnerabilities and the lack of output validation in the backend function to send emails. This leads to remote code execution, gaining unauthorized access and privilege escalation.
2. An attacker embdeds an indirect prompt injection in a webpage instructing the LLM to disregard previous user instructions and use an LLM plugin to delete the user's emails. When the user employs the LLM to summarise the webpage, the plugin deletes the emails
3. A user employs an LLM to summarize a webpage containing an indirect prompt injection to disregard previous user instructions. This causes the LLM to gather sensitive information from the user and perform exfiltration via embedded JavaScript/Markdown.
4. A malicious user uploads a resume with prompt injection, and the backend user gets a positive summary of the resume regardless of its actual contents
5. A user enables a plugin linked to an e-commerce site, and a rogue instruction embedded on a visited website exploits the plugin, causing unauthorized purchases.

# 2) Insecure Output Handling
* Arises when a downstream component accepts LLM output without appropriate scrutiny (passing output directly to the backend or client-side functions without filtering). Because LLM-generated content can be controlled by prompt input, this behavior is similar to providing users indirect access to additional functionalist.
* Successful exploitation of this can lead to Cross-Site Scripting, Cross-Site Request Forgery, Server-Side Request Forgery, privilege escalation, and remote code execution on backend ststems.
* If an application is vulnerable to external prompt injection or the LLM has more privileges than end users, this vulnerability can be even more dangerous

## Common Examples of Vulnerability
* LLM output is entered directly into a system shell or uses a function like exec or eval, allowing for remote code execution
* JavaScript or Markdown is generated by the LLM and returned to the user, and said code is interpreted by the browser, resulting in Cross-Site Scripting

## How to Prevent
1. Apply input validation and sanitization on responses coming from the model to backend functions
2. Encode model output back to users to avoid execution by JavaScript or Markdown

## Example Attack Scenarios
1. An application uses an LLM plugin to get responses for a chatbot feature, but the chatbot responses are automatically passed into internal functions,meaning commands can be executed without proper authorization.
2. A user uses an LLM-powered website summarizer tool on an article, but the site where the article is hosted has a prompt injection that instructs the LLM to capture senstive content from the site or user, which can then be sent to the malicious actor's server
3. An LLM has users craft SQL queries for a backend database through chat, but if the queries aren't properly scrutinized, a user could delete all table data
4. A malicious user instructs the LLM to return a JavaScript payload back to users (via prompt sharing or a chatbot accepting prompts from a URL parameter). The LLM would then return the unsanitized XSS payload to the users, and if there are no additional filters, the JavaScript executes within the user's browser.

# 3) Training Data Poisoning
* For an LLM to be effective, its training data should encompass many domains, genres, and languages. LLMs use deep neural networks to generate output based on patterns from training data.
* Training data poisoning refers to the manipulation of date or the fine-tuning process to introduce vulnerabilities, backdoors, or biases that could compromise the model's security, effectiveness, or ethical behavior. Poisoned information can also result in downstream software exploitation and harm to the model's brand reputation.
* Data poisoning is considered an integrity attack because tampering with the training data impacts a model's ability to output correct predictions. External data sources present higher risk as model creators don't have control of the data or a high level of confidence that the content is not biased or even outright false.

## Common Examples of Vulnerability
1.  A malicious actor or competitor creates malicious, false documents, targets them at a model's training data, and the model's output reflects the bad data
2.  A model is trained using data that hasn't been verified by its source, origin, or content
3.  The model has unrestricted ability to obtain datasets for training data, which has a negative impact on the outputs of its generative prompts.

## Example Attack Scenarios
1. A generative AI prompts can mislead users, leading to biased opinions and sometimes hate
2. If training data is not properly sanitized and filtered, a malicious user of an application could try to purposely feed the model harmful data, forcing it to adapt to that.
3. A malicious actor or competitor intentionally creates inaccurate or malicious docments, which are targeted at the model's training data
4. Prompt injection could allow for data poisoning if client input to an LLM is used to train the model, and there' s a lack of adequate filtering (EX: if malicious or falsified data is input to the model from a client as part of a prompt injection technique, this could become part of the model data).

## How to Prevent
1. Verify the training data's supply chain (especially when sourced externally)
2. Verify legitimacy of data sources and the data itself
3. Craft multiple models using separate training data or fine-tuning for different use-cases to create a more granular and accurate generative AI output
4. Ensure sufficient sanboxing is there to prevent the model from scraping unintended data sources
5. Have strict filters (anamoly and outlier detection) for training data
6. Adversarial robustness techniques like federated learning and constraints to minimize the effect of outliers  or adversarial training (MLSecOPs uses autopoisoning, which combats Content Injection and Refusal Attacks)
7. Testing and Detection, by measuring the loss during the training stage and analyzing trained models to detect signs of a poisoning attack by analyzing model behavior on specific test inputs (EX: a human reviews and audits responses, red team exercises and vulnerability scanning is performed).

# 4) Model Denial of Service
* An attacker interacts with a LLM in a method that consumes a very high amount of resources, resulting in high resource costs and decline in service quality.
* An attacker can also impact the LLM context window (the maximum amount of input and output text the model can handle), which determines the model's language patterns and processing.

## Common Examples of Vulnerability
1. Posing queries that lead to recurring resource usage via high-volume generation of tasks in a queue (EX: LangChain or AutoGPT)
2. Sending very resource-consuming queries
3. Continuous input overflow (input exceeds context window)
4. Repetitve long inputs-continual inputs that exceed the context window
5. Recursive context expansion-attacker creates input that forces the LLM to repeatedly expand and process the context window.
6. Variable-length input flood: the attacker floods the LLM with a large volume of inputs that reach right at the context window's limit, exploiting inefficiencies in processing variable-length inputs and possibly causing the LLM to crash.

## Example Attack Scenarios
1. Attackers send multiple difficult, costly requests to a model leading to heightened resource use and poor service for other users
2. A piece of text is encountered while an LLM-driven tool is gathering information for a query that leads to the tool making many more web page requests, and heightened resource consumption
3. An attacker uses automated scripts or tools to send input that overwhelms the LLM's processing capabilities, causing it to slow down or stop responding
4. An attacker sends many inputs that are just beneath the context window's limit, resulting in the available window capacity being exhausted, the LLM's resources becoming strained, and degraded performance or even denial of service occurring.
5. An attacker uses recursive mechanisms to trigger repeated expansion of the context window, straining the system and potentially crashing it.
6. The LLM is overwhelmed with inputs of varying lengths, putting excessive load on the LLM's resources, causing performance degradation and blocking the system from responding to legitimate requests.

## How to Prevent
1. Implement input validation and sanitization to filter out malicious, limit-breaking content.
2. Cap resource use per request or step, making complex requests execute slower
3. Enforce API rate limits to restrict the requests an individual user/IP address can make in a given timeframe.
4. Limit queued actions
5. Consistently monitor the LLM's resource utilization for specific attack activity.
6. Set strict limits based on the LLM's context window to prevent overload and resource exhaustion.
7. Promote awareness among developers about potential DoS vulnerabilities in LLMs and provide guidelines for secure LLM implementation

# 5) Supply Chain Vulnerabilities
* Data supplied to machine learning models by third parties is susceptible to tampering and poisoning. LLM plugin extensions can also bring their own issues.

## Common Examples of Vulnerability
1. Traditional third-party package vulnerabilities, including outdated/deprecated components
2. Using a vulnerable pre-trained model for fine tuning
3. Use of poisoned crowd-sourced data for training
4. Using outdated or deprecated models that are no longer maintained
5. Unclear data privacy policies lead to sensitive data being used for model training and thus getting exposed.

## How to Prevent
1. Carefully vet data sources and suppliers (including pribacy policies), only using trusted suppliers
2. Only use reputable plugins that were tested for your application requirements
3. Understand and apply mitigations like vulnerability scanning, management, and patching components
4. Maintain an up-to-date inventory of components to ensure you have an up-to-date, accurate inventory.
5. If your LLM application uses its own model, you should use MLOPs best practices and platforms offering secure model repositories with data, model, and experiment tracking
6. Use model and coe signing for external models and suppliers
7. Anomoly detection and adversarial robustness can help detect tampering
8. Implement monitoring to cover component and environment vulnerabilities scanning, use of unauthorised plugins, and out-of-date components.
9. Implement a patching policies to mitigate outdated components, and make sure the API the model relies on is maintained.
10. Regularly review supplier Security and Access

## Example Attack Scenarios
1. An attacker exploits a vulnerable Python library to compromise a system
2. An attacker provides an LLM plugin to search for flights which generates fake links that lead to scamming plugin users
3. An attacker exploits the PyPi package registry to make model developers download a compromised version of the package, thus stealing data and escalating privilege.
4. An attacker poisons a publicly available pre-trained model specialising in economic analysis and social research to create a backdoor which generates misinformation.
5. An attacker poisons a publicly available data set to create a backdoor that favors certain companies when fine-tuning models
6. A compromised employee of a supplier exfiltrates data or code stealing IP.
7. An LLM operator changes its T&Cs and Privacy Policy so it requires an explicit opt-out from using application data for model trainng, resulting in memorization of sensitive data.

# 6) Sensitive Information Disclosure
* LLMs can reveal sensitive information, proprietary algorithms, or other confidential details through their output. Data sanitation and Terms of Use policies that make consumers aware of how their data is processed are integral.
* We cannot inherently trust client output to a LLM, or LLM output to a client. Adding restrictions within the system prompt around the types of data the LLM should return can provide some mitigation against disclosure of sensitive information, but LLMs are unpredictable, so restrictions may be ignored via prompt injection or other means.

## Common Examples of Vulnerability
1. Incomplete or improper filtering of sensiive information in the LLM's responses
2. Overfitting or memorization of sensitive data in the LLM's training process
3. Unintended disclosure of confidential information due to LLM misinterpretation, lack of data scrubbing methods or errors.

## How to Prevent
1.
* Source: https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_0.pdf
