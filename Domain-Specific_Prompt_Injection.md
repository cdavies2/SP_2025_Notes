# Domain-Specific Prompt Injection Detection
* While SQL injection attacks have deteministic solutions, LLMs utilize natural language, so there is a lack of strong separation between instructions and data. As a result, most industry standard solutions treat LLMs and their outputs as untrusted entities.
* This can involve implementing external authorization controls to limit the scope of actions LLM agents can perform using tools/plugins, sanitizing outputs to remove more harmful content, and employing human-in-the-loop oversight to ensure that all actions taken by the LLM are approved by a human operator.
* The best ways to sanitize input are...
  * Reduce the accepted character sets to thwart attackers' attempts to embed malicious instructions
  * Detect potentially malicious prompts by utilized machine learning models trained to identify signs of injection attempts. 
Source: https://labs.withsecure.com/publications/detecting-prompt-injection-bert-based-classifier

## The Sample Recruitment Assistant Application
* For this app, GPT-4 Turbo was used to sift through job applications and highlight the best candidate. The applications builds this prompt...
  * System message: "You are an expert recruiter assistant. Your task is to evaluate job applications against the provided job specification for a Prompt Engineer position and select the top one based on their merits. You are concise, follow instructions and don't provide unprompted long explanations, just get to the task"
  * List of job applications with their contents
  * Final instruction to return the best application.
### Prompt Injection Attack Scenario
* The attacker provides a job application containing an adversarial prompt that instructs the LLM to include a markdown image pointing to the attackers server and includes  in the URL a base64-encoded list of all applicants, with their names, email addresses, and phone numbers. In practice, this means that when the recruiter is shown the output of the LLM, the beowser will try to render the image and thus send the attacker confidential information about the other candidates.

Source: https://labs.withsecure.com/publications/detecting-prompt-injection-bert-based-classifier
