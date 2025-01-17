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
1. Integrate proper data sanitization techniques to keep user data from becoming part of the training data
2. Input validation should be used to filter out possible malicious inputs, and thus prevent the model from being poisoned.
3. When enriching the model with data and fine tuning it, apply the rule of least pribilege and don't train the model on information that the highest-privileged user can acces which may be displayed to a lower-privileged user, limit access to external data sources, and apply strict access control methods to external data sources.

## Example Attack Scenarios
1. Legitimate users are accidentally exposed to another LLM user's data
2. User A creates prompts to bypass input filters and sanitization to get the LLM to reveal other users' PII (Personally Identifiable Information)
3. PII is leaked into the model training data due to user mistakes or the LLM itself, increasing the possibility of said info getting revealed.

# 7) Insecure Plugin Design
* LLM plugins are called automatically by the model, driven by it, and implement free-text inputs from the model without validation or type checking.
* Inadequate access control allows a plugin to automatically trust other plugins and assume the end user provided the inputs, allowing malicious inputs to run, and data exfiltration and remote code execution to occur.

## Common Examples of Vulnerability
1. A plugin accepts all parameters in a single text field instead of specific input parameters
2. A plugin accepts configuration strings that can override entire configuration settings
3. A plugin accepts raw SQL or programming statements
4. Authentication is performed without explicit authorization to a particular plugin
5. A plugin treats all LLM content as being created entirely by the user and performs any requested actions without requiring additional authorization.

## How to Prevent
1. Plugins should enforce strict parameterized input wherever possible and include type and range checks on inputs, and when this isn't possible,input should be sanitized, validated, and carefully inspected.
2. Plugins should be inspected and tested thoroughly to ensure adequate validation with Static Application Security Testing (SAST) scans and Dynamic and Interactive Application Testing (DAST, IAST) during development.
3. Plugins should follow least-privilege access control, exposing as little functionality as possible while still performing their function
4. Plugins should use appropriate authentication identities (like OAuth2) to apply effective authorization and access control, and API keys should be used to provide context for custom authorization decisions.
5. Require manual user authorization and confirmation of any action taken by sensitive plugins

## Example Attack Scenarios
1. A plugin accepts a base URL and instructs the LLM to combine the URL with a query to obtain weather forecasts. A malicious user can make the URL point to their own dmain, thus allowing them to inject their own content into the LLM system.
2. A plugin accepts free-form input into a single field it does not validate, and an attacker supplies carefully crafted payloads to perform reconnaissance from error messages, and then exploits third-party vulnerabilities to execute code.
3. A plugin that retrieves embeddings from a vector store accepts configuration parameters as a connection string without any validation, allowing the attacker to experiment and access other vector stores by changing names or host parameters and exfiltrate embeddings they should not have access to.
4. A plugin accepts SQL WHERE clauses as advanced filter, which are appended to the filtering SQL, allowing the attacker to stage a SQL attack.
5. An attacker uses indirect prompt injection to exploit an insecure code management plugin without input validation and weak access control to gain repository ownership and lock the user out of their repositories.

# 8) Excessive Agency
* An LLM is able to interface with other systems and undertake actions in response to a prompt. The decision over which functions to invoke could also be delegated to an LLM agent to dynamically determine based on input prompt or LLM output.
* Excessive Agency is usualy the result of excessive functionality, permissions,or autonomy. It enables damaging actions to occur in response to an LLM's unexpected outputs.

## Common Examples of Vulnerability
1. Excessive Functionality: an agent has access to plugins that have functions that aren't needed for the intended operation of the system (EX: model is supposed to read documents from a repository, plugin allows it to modify and delete them).
2. Excessive Functionality: a plugin with open-ended functionality fails to properly filter the input instructions for commands outside of necessary ones (EX: a plugin to run one specific shell command fails to properly prevent other command execution)
3. Excessive Permissions: a plugin has perissions on other systems that are unnecessary (EX: a plugin intended to read data from a server has SELECT permissions, but also UPDATE and DELETE)
4. Excessive Permissions: plugin has access to privileges that a typical user shouldn't
5. Execssive Autonomy: an LLM-based application or plugin cannot independently verify and approve high-impact actions (EX: a plugin to delete documents starts deleting them without confirmation from the user)

## How to Prevent
1. Limit the plugins/tools that LLM agents can call to the minimum functions needed
2. Limit functions implemented in the tool to be minimum necessary (EX: plugin to read and summarize emails cannot send them too)
3. Avoid open-ended functions where possible (build a file-writing plugin for just that rather than generically running shell commands)
4. Limit the permissions that LLM plugins/tools are granted to other systems the minimum necessary to limit scope of undesirable acions (EX: LLM agent that makes product recommendations can only read the products table, it cannot read others or change the records). This should be enforced by applying appropriate database permissions for the identity that the LLM plugin uses to connect to the database.
5. Track user authorization and security scope to make sure actions are executed on downstream systems in the context of the specific user, with minimum needed privileges
6. Utilize human-in-the-loop control to ensure a human approves all actions before they're taken (EX: an app that creates social media content for a user includes an approval routine that actually makes the posts)
7. Implement authorization in downstream systems rather than relying on an LLM to devide if an action is allowed or not.
8. To limit damage, log and monitor activity of LLM plugins/tools and downstream systems to identify where undesirable actions are occurring and respond accordingly and implement rate-limiting to reduce the possible undesirable actions in a period.

# 9) Overreliance
 * Occurs when systems or people depend on LLMs for decision-making or content generation without proper oversight. LLMs hallucinations or confabulation result in misinformation, legal issues, and reputational damage.
 * LLM-generated source code can introduce unnoticed security vulnerabilities

## Common Examples of Vulnerability
1. LLM provides inaccurate information in responses
2. LLM produces nonsensical text that is grammatically correct but doesn't make sense
3. LLM melds information from varied sources, creating misleading content
4. LLM suggests insecure/faulty code, leading to vulnerabilities when incorporated
5. Failure of provider to appropriately communicate the inherent risks to end users, leading to potential harmul consequences

## How to Prevent
1. Regulaly monitor and review LLM outputs (maybe compare several model responses for the same prompt)
2. Cross-check the LLM output with trusted external sources. This additional layer of validation can help ensure information provided is accurate and reliable.
3. Enhance the model with fine-tuning or embeddings to improve output quality. Techniques like prompt engineering, parametr efficient tuning (PET), full model tuning, and chain of though prompting can be employed for this purpose.
4. Implement automatic validation mechanisms that can cross-verify the generated output against known facts or data
5. Break down complex tasks into manageable subtasks and assign them to different agents, reducing chances of hallucinations
6. Communicate the limitations of LLMS (like potential inaccuracies)
7. Build APIs and user interfaces that encourage responsible and safe LLM use (content filters, user warnings, clear labeling of AI-generated content)
8. When using LLMs in development environments, establish secure coding practices and guidelines to prevent integration of possible vulnerabilities.

## Example Attack Scenario
1. A news organization generates articles with an AI model, and a malicious actor feeds the AI misleading information. The model also unintentionally plagiarizes, leading to copyright issues and mistrust.
2. Overreliance on the AI's suggestion in software development introduces security vulnerabilities into the application due to insecure default settings or recommendations inconsistent with secure coding practices.
3. A software development firm uses an LLM to assist developers. The LLM suggests a non-existent code library or packahe, and a developer, trusting the AI, unknowingly integrates a malicious package into the firm's software.

# 10) Model Theft
* This arises when proprietary LLM (being valuable intellectual property) are compromised, physically stolen, copied, or weights and parameters are extracted to create a functional equivalent. The impact of LLM model theft can include economic and brand reputation loss, erosion of competitive advantage, unauthorized usage of the model or unauthorized access to sensitive information contained within the model.
* The theft of LLMs represents a significant security concern as language models become increasingly powerful and prevalent. Organizations must implement robust security measures to protect their LLM models, including access controls, encryption, and continuous monitoring.

## Common Examples of Vulnerability
1. An attacker exploits a vulnerability in a company's infrastructure to gain unauthorized access to their LLM model repository
2. An insider threat scenario where a disgruntled employee leaks model or related artifacts
3. An attacker queries the model API using carefully crafted inputs and prompt injection techniques to collect a sufficient number of outputs to create a shadow model
4. A malicious attacker is able to bypass input filtering to perform a side-channel attack and harvest model weights and architecture information to a remote controlled resource
5. The attack vector for model extraction involves querying the LLM with a large number of prompts on a particular topic. The outputs can then be used to fine-tune another model (this requires that the attacker generate a large number of targeted prompts, the outputs not contain hallucinated answers, and still won't create a 100% identical model).
6. The attack vector for functional model replication involves using the target model via prompts to generate synthetic training data (self-instructing) to fine-tune another foundational model to produce a functional equivalent.
7. A stolen/shadow model can stage adversarial attacks including unauthorized access to sensitive information contained within the model or experiement undetected with adversarial inputs to further stage advanced prompt injections.

## How to Prevent
1. Implement strong access controls and authentication mechanisms to limit unauthorized access to LLM model repositories and training environments. Supplier management tracking, verification and dependency vulnerabilities are important to prevent supply-chain attacks.
2. Restrict the LLM's access to network resources, internal services, and APIs
3. Regularly monitor and audit access logs and activities related to LLM model repositories to detect and respond to any suspicious or unauthorized behavior quickly
4. Automate MLOps deployment with governance and tracking and approval workflows to tigheten access and deployment controls within the infrastructure
5. Implement controls and mitigation strategies to mitigate and/or reduce risk of prompt injection techniques causing side-channel attacks
6. Rate limiting of API calls where applicable and/or filters to reduce risk of data exfiltration from the LLM applications, or implement techniques to detect extraction activity from other monitoring systems.
7. Implement adversarial robustness training to help detect extraction queries and tighten physical security measures
8. Implement a watermarking framework into the embedding and detection stages of an LLMs lifecycle.

## Example Attack Scenario
1. An attacker exploits a vulnerability in a company's infrastructure to gain unauthorized access to their LLM model repository. The attacker then exfiltrates valuable LLM models and uses them to launch a competitng service or extract sensitive information, harming the original company.
2. A disgruntled employee leaks model or related artifacts
3. An attacker queries the API with carefully selected inputs and collects sufficient number of outputs to create a shadow model.
4. A security control failure is present within the supply-chain and leads to data leaks of proprietary model information
5. A malicious attacker bypasses input filtering techniques and preambles of the LLM to perform a side-channel attack and retrieve model information to a remote controlled resource under their control.
  
* Source: https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_0.pdf
* 
