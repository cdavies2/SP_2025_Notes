# Prompt Injections
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

* Source: https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_0.pdf
