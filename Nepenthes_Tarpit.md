# What are Tarpits?
* Sometimes referred to as network tarpits or honeypot tarpits, they are designed to slow down incoming requests.
* Rather than outright block or deflect threats, tarpits draw cyber-threats into an intentional slow-motion engagement, inhbiting their ability to progress for long enough that the security team can identify the threat and craft an appropriate response.
* The goal of a tarpit is to trap a malicious bot or automated attack script under the guise of legitimate network entities, thus protecting the network's actual, critical resources.
  
## How Tarpits Work
1. _Engagement_: tarpits pose as real network systems and services (like open port and vulnerable appliations, thus getting the attention of a hacker's automated scanning tool)
2. _Slow Responses_: when malicious traffic such as worms, bots, and scanners interact with the tarpit, instead of rejecting the connection, the tarpit engages with it but responds incredibly slowly, like by sending data in extremely small packets, introducing long delays in the connection, or never-ending streams of data.
3. _Resource Depletion_: the slow engagement forces the attacking scripts to spend more time and resources maintaining the connection with the tarpit, bogging them down and keeping them from reaching actual targets.
* Tarpits are a passive cyber security tool, they do not actively seek out threats, instead trapping threats that come to them.
* Tarpits are typically part of a layered security technology, complementing other measures like firewalls, intrusion detection systems (IDS) and security information and event management (SIEM) systems. They function well as a warning and intelligence-gathering tool, and are often used alongside more proactive, preventative measures.
## In-Depth Look: Open Source Tarpits
* LaBrea Tarpit-captures malicious traffic by dynamically generating false systems, or 'virtual hosts', that appear as attractive targets to worms and automated attacks.
* Endlessh-serves an important role in SSH (Secure Shell) server defense. SSH is a common attack vector for cybercriminals because it is so commonly used, and can potentially provide access to system controls. Endlessh is an SSH tarpit that keeps attackers engaged with what seems to be an SSH server by sending an endless stream of banner text. This ties up the attacking client, preventing it from interacting with legitimate services or probing other systems. While the bots are occupied, network administrators can maintain and safeguard genuine SSH services, relocating them to different, less obvious ports and operating them with much less risk of attack.
* These systems provide traps that both stop attackers and turn their methods against them.

## Tarpit Detection
* Tarpits aren't foolproof. Skilled hackers might probe network services to find inconsistencies that suggest traps before they attack. For example, they might send specially crafted packets using protocols like HTTPS and SMTPS to services that appear legitimate. If these services exhibit irregular behavior (like denying three-way handshakes, a crucial part of TCP/IP network communication), it could signal the presence of a honeypot.
* Malicious actors might also employ advanced techniques like using multi-layered proxies, including The Onion Router (TOR) to disguise their origin, and use encryption and steganography to avoid detection from tarpits.
* Source: https://systemweakness.com/slowing-cyberthreats-to-a-crawl-the-ingenuity-of-cybersecurity-tarpits-2d978fb3869b

# Nepenthes
* The Nepenthes tarpit is designed to catch web crawlers, specifically those that scrape data for large language models. It generates an endless sequence of pages, each containing dozens of links that circle back to the tarpit. Pages are randomly generated, but they still appear as consistent, never-changing flat files.
* Intentional delay is used to prevent crawlers from bogging down your server, as well as wasting their time.
* The main downsides of this sofrware are that it can cause heightened CPU load (as scrapers are looking for content and you are providing them with that), and it is not possible to differentiate between crawlers that are indexing for search purposes and crawlers that are training AI models.

## Usage
* The tarpit is hidden behind nginx or Apache (whichever the site has been implemented in). If the pit is directly exposed to the Internet, you won't be able to easily disguise it as innocent. HTTP headers are also used to configure the tarpit.
* In the nginx configuration code below, multiple headers are added...
  * "X-Prefix" tells the tarpit that all links should go forward to that path. This needs to match whatever is in the 'location' directive.
  * "X-Forwarded-For" makes gathered statistics more useful
  * Proxy-buffering needs to be off. This is because LLM crawlers usually disconnect if they aren't given a response within a few seconds, and Nepenthes gets around this by drip-feeding a few bytes at a time.
```
location /nepenthes-demo/ {
            proxy_pass http://localhost:8893;
            proxy_set_header X-Prefix '/nepenthes-demo';
            proxy_set_header X-Forwarded-For $remote_addr;
            proxy_buffering off;
    }
```
## Installation
* Nepenthes can be installed using Docker, or manually.
  * For Docker installation, simply modify the configuration file to fit your preferences
  * For Manual installation, some of the required programs include Lua (5.4 or higher preferred), SQLite (if using Markov) and OpenSSL. Some required Lua modules include cqueues, ossl (or luaossl), lpeg, lzlib (or lua-zlib), dbi-sqlite3 (or luadbi-sqlite3) and unix (or lunix)
* Nepenthes shouldn't be ran as root, so you should create a nepenthes user, unpack the tarball, and tweak config.yml however you prefer. Once that is finished, you can begin, and sending "SIGTERM" or "SIGINT" will shut the process down (code below).
```
useradd -m nepenthes
--start the tarball
cd scratch/
tar -xvzf nepenthes-1.0.tar.gz
    cp -r nepenthes-1.0/* /home/nepenthes/
--tweak config
su -l -u nepenthes /home/nepenthes/nepenthes /home/nepenthes/config.yml
```
## Bootstrapping the Markov Babbler
* The Markov feature requires a trained corpus. Ideally, all tarpits should look different (in order to avoid them being detected).
