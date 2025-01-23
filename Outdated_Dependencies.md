# A06:2021 - Vulnerable and Outdated Components
* Source: https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components/
* The "Vulnerable Components" issue does not have any Common Vulnerability and Exposures (CVEs) mapped to CWEs.
* You are likely vulnerable to an outdated dependency attack if....
  * You do not know the versions of all components you use (both client-side and server-side). This includes components directly used as well as nested dependencies
  * If software is vulnerable, unsupported, or out of data. This includes the OS, web/application server, database management system (DBMS), applications, APIs, and all components, runtime environments, and libraries.
  * If you do not scan for vulnerabilities regularly and subscribe to security bulletins related to the components you use.
  * If you do not fix or upgrade the underlying platform, frameworks, and dependencies in a risk-based, timely fashion. This often occurs in environments where patching is a somewhat infrequent (monthly or quarterly) task under change control. This leaves organizations open to days or months of unnecessary exposure to fixed vulnerabilities.
  * If software developers do not test the compatibility of updated, upgraded, or patched libraries.
  * If you do not secure the components' configurations
* To prevent these attacks, you should have a patch management process in place that...
  * Remove unused dependencies, unnecessary features, components, files, and documentation
  * Continuously inventory the versions of both client-side and server-side components (frameworks, libraries) and their dependencies using tools like versions, OWASP Dependency Check, retire.js, etc.
  * Continously monitor sources like Common Vulnerability and Exposures (CVE) and National Vulnerability Database (NVD) for vulnerabilities in the components. Use software composition analysis tools to automate the process. Subscribe to email alerts for security vulnerabilities related to components you use.
  * Only obtain components from official sources over secure links. Prefer signed packages to reduce the chance of including a modified, malicious component.
  * Monitor for libraries and components that are unmaintained or do not create security patches for older versions. If patching is impossible, consider deploying a virtual patch to monitor, detect, or protect against the discovered issue.
* Every organization must ensure an ongoing plan for monitoring, triaging, and applying updates or configuration changes for the lifetime of the application or portfolio.
## Example Attack Scenarios
* Components typically run with the same privileges as the application itself, so flaws in any component can have a serious impact. Such flaws can be accidental (EX: coding error) or intentional (EX: backdoor in a component). Some exploitable component vulnerabilities that have been discovered are....
  * CVE-2017-5638: A Struts 2 remote code execution vulnerability that enables the execution of arbitrary code on the server, has been blamed for signficant breaches
  * While the Internet of things (IoT) is frequently difficult to impossible to patch, the importance of patching them can be great (EX: biomedical devices)
* Automated tools can help attackers find unpatched/misconfigured systems. For instance, the Shodan IoT search engine can help you find devices that suffer from the Heartbleed vulnerability that was patched in April 2014.

# Outdated Dependencies: The Hidden Culprit in Penetration Test Findings
* Source: https://www.iflockconsulting.com/blog/outdated-dependencies
* Dependencies allow software developers to innovate faster and cut down on repetitive work. However, when said libraries go unpatched, they can result in hackers bypassing otherwise strong security measures. This makes them a major target during penetration testing.
* Example Scenario: an application uses a popular open-source library, but a critical vulnerability is disclosed. Without an update, regardless of how secure the rest of the system is, this flaw becomes a direct entry point for attackers.
## Pentest Findings You Can Eliminate
* Staying current can prevent...
  * Known CVEs (Common Vulnerabilities and Exposures): many pentest reports highlight CVEs in outdated dependencies, often with well-documented exploits available
  * Unpatched Libraries: attackers target libraries that remain unpatched long after vulnerabilities are disclosed
  * Chained Exploits: old dependencies can create paths for chained attacks, where a minor vulnerability escalates into a severe breach.
  * Unsupported Versions: outdated software often lacks vendor support, leaving it as a permanent risk

 ## A Simple Fix with Big Results
 * Outdated dependency attacks often require no new defenses to combat, just implementation of available patches. However, many teams delay updates and patches, worrying about breaking changes or lack of visibility into dependency chains.
 * To minimize disruption while staying secure...
   *  Automate Updates: tools like Dependabot or Renovate identify and suggest updates without extra effort
   *  Continuous Monitoring: Vulnerability scanners like Snyk or OWASP Dependency-Check catch risks early.
   *  Prioritize Risky Dependencies: focus on libraries with critical vulnerabilities or heavy exposure to external inputs
   *  Test for Stability: pair updates with robust regression testing to avoid introducing issues.
