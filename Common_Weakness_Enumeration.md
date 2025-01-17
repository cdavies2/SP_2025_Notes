# CWE-94: Improper Control of Generation of Code
* Present in Vanna-ai's prompt injection vulnerability
* Defintion: the product does not effectively neutralize user-controllable input before it is placed in output that is used as a web page that is served to other users
* Cross-site scripting (XSS) vulnerabilities occur when...
    1. Untrusted data enters a web application (usually via a web request)
    2. The web application dynamically generates a web page that contains this untrusted data.
    3. During page generation, the application doesn't prevent data from containing content that is executable by a web browser (JavaScript, HTML attributes, Flash)
    4. A victim visits the generated web page through a web browser, which contains malicious script that was injected using the untrusted data.
    5. Since the script comes from a web page that was sent by the web server, the victim's web browser executes malicious script in the context of the web server's domain
    6. The same-origin policy is violated (scripts in one domain should not be able to access resources or run code in a different domain)

  ## Kinds of XSS
  * _Type 1: Reflected XSS (Non-Persistent)_-server reads data directly from the HTTP request and reflects it back in the response. This occurs when a victim supplies dangerous content to a vulnerable application, which is then executed by the web browser and reflected back to the victim
  * _Type 2: Stored XSS (Persistent)_-stores dangerous data in a database, message forum, visitor log, or other trusted data store. The data is later read back into the application and included in dynamic content. The malicious content is often displayed to users with elevated privileges or that interact frequently with sensitive data. Therefore, if one of those users executes malicious content, the attacker can gain access to their data or perform privileged actions on the users' behalf (which may not be noticed as easily by administrators)
  * _Type 0: DOM-Based XSS_-the client performs the injection of the XXS into the page rather than the server doing so. This usually involved server-controlled, trusted script that is sent to the client (like JavaScript that performs sanity checks on a form before the user submits it). If server-supplied scripts process user-supplied data and then injects it back into the webpage, DOM-Based XSS is possible.
* Afte a malicious script is injected, the attacker can steal confidential information, send malicious requests on the victim's behalf, or engage in phishing and get the victim to give up passwords.

## Common Consequences
* Access Control Confidentiality-malicious scripts get access to user cookies (bypass protection mechanism, read application data)
* Integrity, Confidentiality, Availability-it could be possible to execute unauthorized code on a victim's computer when cross-site scripting is combined with other flaws.
* Confidentiality Integrity Availability Access Control-some of the most damaging attacks include disclosure of end user files, installation of Trojan horse programs, redirecting users to other sites, and modifying content presentation.

## Potential Mitigations
* During design, use a vetted library or framework to make sure output is properly encoded (EX: the OWASP ESAPI Encoding module)
* During design and implementation, understand every area where untrusted inputs could enter your software (parameters, reverse DNS, emails, files, databases) and remember that such inputs may be obtained directly through API calls. This is _Attack Surface Reduction_
* During design, us structured mechanisms that automatically enforce separation between data and code. Said mechanisms can provide quoting, encoding, and validation automatically rather than relying on the developer to provide the capability whenever output is generated. This is known as _Parameterization_
* During implementation, assume all input is malicious. Have a list of acceptable inputs that strictly conform to specifications, reject any input that doesn't, or transform it into something that does. This is _input validation_. When dynamically constructing web pages, it is helpful to have stringent allowlists that limit the character set based on the expected value of the parameter in the request. All data in the request, including hidden fields, cookies, headers, and the URL itself needs to be validated and cleansed. A common mistake that leads to propagation of XSS vulnerabilities is to only validate fields that are displated by the site.
* During architecture and design, if you know what the set of acceptable objects contains, create a mapping from a set of fixed input values (like numeric IDs) to the actual filenames and URLS and reject all other input. This is _enforcement by conversion_.
* During operation and implementation, if PHP is being used, the application should be developed so it does not rely on register_globals. This is a form of _Environment hardening_.

## CVE Example
* CVE-2024-5826

* Source: https://cwe.mitre.org/data/definitions/94.html

# CWE-1173: Improper Use of Validation Framework
* This occurs when a product either doesn't use or incorrectly uses an input validation framework that is provided by either the source language or an independent library. The purpose of input validation is to check input against specific criteria and call error handlers when invalid input is received. Improper (either incorrect or outright missing) implementation of these frameworks is not exploitable in an of itself, but can lead to exploitable conditions if proper input validation isn't performed later in the product. Lack of validation frameworks can also hurt the code's maintainability, as future developers might not recognize downstream input validation being used in place of the framework.

## Common Consequences
* Integrity-unchecked input can lead to cross-site scripting, process control, and SQL injection vulnerabilities.

## Detection Methods
* _Automated Static Analysis_-a static analysis tool could help a user specify which application-specific methods or functions perform input validation, and said tool could have built-in knowledge of validation frameworks like Struts. The tool may then suppress or de-prioritize any associated warnings, allowing analysts to focus on areas of software where input validation might not be present. The only downside is automated analysis may not be able to recognize proper validation, leading to false positives.

## CVE Example
* CVE-2024-48919
* Source: https://cwe.mitre.org/data/definitions/1173.html

# CWE-74: Improper Neutralization of Special Elements in Output Used by a Downstream Component ('Injection')
* The product constructs all or part of a command or data structure using externally-influenced input from an upstream component, but it doesn't neutralize or incorrectly neutralizes special elements that could modify how it is parsed or interpreted when it is sent to a downstream component.
* Software/automated logic makes assumptions about what consitutes data and control. It does not always verify if these assumptions are true for user-controlled input, which can lead to injection problems. The most effective way to combat these weaknesses is to note distinct features that classify them as injection weaknesses.
* All injection isssues have one thing in common; they allow for injection of control plane data into the user-controlled data plane. This means that a process' execution could be altered by sending code in through legitmate data channels.

## Common Consequences
* Confidentiality-sensitive application data is often extracted and exposed
* Access Control-in some cases, injectible code can control authentication, resulting in protection mechanisms being bypassed.
* Other-injection attacks can alter execution logic
* Non-repudiation-often activities performed by injection control code are hidden and unlogged.

## Potential Mitigations
* During the rquirements phase, choose appropriate support technologies that are not as vulnerable to this.
* During implementation, utilize allow and denylist parsing to filter input.

## CVE Example
* CVE-2024-5184
* Source: https://cwe.mitre.org/data/definitions/74.html


