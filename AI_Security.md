# How to Secure AI

## How (un)Secure is Artificial Intelligence?
* Malicious entities often use AI to spread malware, and infect both code and datasets. AI vulnerabilities also can be used to execute data breaches.
* The AI in cybersecurity market is expected to reach $60.6 billion by 2028, displaying that AI will be often utilized to identify and mitigate attacks caused by AI.

##AI Security Risks
* Increased Attack Surface-the integration of generative AI into software development life cycles (SDLCs) inherently changes an organization's IT infrastructure and introduces new risks. The main security challenge with AI is ensuring that all of its infrastructure is managed by appropriate security teams. Visibility of AI infrastructure can help remediate vulnerabilities, reduce risks, and limit the attack surface.
* Higher Likelihood of Data Breaches and Leaks-according to The Independent, 43 million sensitive records were compormised in just August 2023. The risks of such attacks include finacial damages, activity disruption, and reputational harm
* Chatbot Credential Theft-more than 100,000 ChatGPT accounts were compromised between 2022 and 2023.
* Vulnerable Development Pipelines-data and model engineering occurs beyond traditional application development boundaries, leading to new security risks. The phases of the AI pipeline and some involved risks include...
  * Data Collection and Handling-part of design, data is finetuned and stored. Some risks include oversharing of data (beyond scope of training), supply chain attacks (unauthorized 3rd parties modify data) and training on overly sensitive data (such as data owned by the customers and not the company)
  * Model Training-part of design, tools for training and improving the model itself are here. Risks include the training system being exposed, and poisoning as a result of training on untrusted data.
  * System Architecture-part of building and running, workloads and tools for running apps that incorporate an AI model. Risks include the model being overprivileged (granted too high permissions for the environment) and unintended cross-access
  * Model Inference-part of building and running, APIs for interfacing with the app and prompting the model. Risks include prompt injection and model extraction.
* Data Poisoning-inputting malicious datasets into Generative AI models to influence their outcomes and create biases.
* Direct Prompt injections-threat actors input prompts to LLMs that are meant to compromise or obtain data or even execute malicious code.
* Indirect Prompt Injections-a threat actor pushes a Generative AI model towards an untrustworthy data source to influence its actions. Risks of this include execution of malicious code, data leaks, and providing misinformation to users.
* Hallucination abuse-AI can hallucinate information. Malicious actors sometimes work to "legitimize" hallucinations, thus giving end users information influenced by them

##AI Security Frameworks and Standards
* NIST's Artificial Intelligence Risk Management Framework-has four functions: govern, map, measure, and manage
* Mitre's Sensible Regulatory Framework for AI Security and ATLAS Matrix
* OWASP's Top 10 for LLMs identifies and proposes standards to protect critical LLM vulnerabilities
* Google's Secure AI Framework-Six Step Process
* PEACH Framework-privilege hardening, encryption hardening, authentication hardening, connectivity hardening, and hygeine

## Simple AI Security Recommendations and Best Practices
1. Choose a tenant isolation framework
2. Customize your GenAI architecture-some components need shared security boundaries, some need dedicated boundaries, and for others, it depends on context
3. Evaluate GenAI Contours and Complexities-make sure an AI models' responses to end users are private, accurate, and constructed with legitimate datasets
4. Don't Neglect Traditional Cloud-Agnostic Vulnerabilities-remember that GenAI is no different from other multi-tenant applications, so it can still suffer from API vulnerabilities and data leaks.
5. Ensure Effective and Efficient Sandboxing-make sure the environment where you isolate and test your applications is securely built.
6. Conduct Isolation Reviews-a tenant isolation review provides a comprehensive topology of customer-facing interfaces and internal security boundaries.
7. Prioritize Input Sanitization-establishing limitations (like having dropdown menus with options instead of textboxes for user input) can mitigate vulnerabilities. The biggest challenge is finding a balance between security and end-user ease of experience.
8. Optimize Prompt Handling-all end-user prompts should be monitored and flagged if suspicious (like if they show signs of malicious code execution)
9. Understand the security implications of customer feedback-feedback textboxes can be filled with malicious content, thus might be worth replacing with dropdown fields.
10. Work with Reputable AI Security Experts

Source: https://www.wiz.io/academy/ai-security
