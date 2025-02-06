# What is Homomorphic Encryption?
* Fully homomorphic encryption (FHE) is an innovative technology that can help you achieve zero trust by unlocking the value of data on untrusted domains without needing to decrypt it
* Sensitive encrypted data must be decrypted before it is accessed for computing and business-critical operations. This opens the door to potential compromise of privacy and confidentiality controls.
* Fully homomorphic encryption ensures that data is always encrypted and can be shared even on untrusted domains in the cloud while remaining unreadable by those doing computations.
* One can perform high-value analytics and data processing, by internal or external parties, without exposing data.
## Benefits of Homomorphic Encryption
1. Gaining valuable insights-generate measureable economic benefits by allowing lines of business and third parties to perform big data analytics on encrypted data while maintaining privacy and compliance controls
2. Collaborate confidently on hybrid cloud-process encrypted data in public and private clouds and third-party environments while maintaining confidentiality controls.
3. Enable AI, analytics, and machine learning (ML)-use AI and ML to compute upon encrypted data without exposing sensitive information.
## Use Cases
* Encrypted predictive analysis in financial services-machine learning can help us create predictive models for conditions ranging from financial transactions fraud to investment outcomes, regulations and policies often prevent organizations from sharing and mining sensitive data. FHE enables computation of encrypted data with ML models without exposing the information.
* Privacy in healthcare and life sciences-despite the efficiency of cloud in hosting workloads for large clinical trials, privacy risks and healthcare regulations can make it impractical for hospitals to transition to using the cloud. FHE can improve acceptance of data-sharing protocols, increase sample sizes in clinical research, and accelerate learning from real-world data.
* Encrypted search in retail and consumer services-technology enables large-scale monitoring of how consumers search and access information, but privacy rights make it difficult for organizations to monetize that data. FHE makes it possible to gain insights on consumer behavior while concealing using queries and protecting an individual's right to privacy.
* Source: https://www.ibm.com/think/topics/homomorphic-encryption
# Homomorphic Encryption: Wikipedia
* Homomorpic encryption allows computations to be performed on encrypted data without first needing to decrypt it. The resulting computations are left in an encrypted form which, when decrypted, result in an output identical to that of the operations performed on the unencrypted data.
* Homomorphic encryption does not protect against side-channel attacks that observe behavior, it can be used for privacy-preserving outsourced storage and computation. This allows data to be encrypted and outsourced to commercial cloud environments for processing, all while encrypted.
* For example, encrypted photographs can be scanned for points of interest, without revealing the contents of the photo. Side-channels can however observe that a photograph was sent to a point-of-interest lookup service, revealing the fact that photographs were taken.
* Homomorphic encryption eliminates the need for processing data in the clear, thereby preventing attacks that would enable an attacker to access that data while it is being processed, using privilege escalation.
* For sensitive data, like healthcare information, homomorphic encryption can be used to enable new services by removing privacy barriers inhibiting data sharing or increasing security to existing services. For instance, predictive analytics in healthcare can be hard to apply via a third-party service provider due to medical data privacy concerns. If the predictive-analytics service provider could operate on encrypted data instead, without having the decryption keys, these privacy concerns are diminished. Also, if the service provider's system is compromised, the data would remain secure.
## Description
* Homomorphic encryption has an additional evaluation capability for computing over encrypted data without access to the secret key. The result of such a computation remains encrypted. Homomorphic refers to homomorphism in algebra; the encryption and decryption functions can be thought of as homomorphisms between plaintext and ciphertext spaces.
* Homomorphic encryption computations are represented as either Boolean or arithmetic circuits. Some common types of homomorphic encryption include....
  * Partially homomorphic encryption-encompasses schemes that support the evaluation of circuits consisting of only one type of gate (EX: addition or multiplication)
  * Somewhat homomorphic encryption-schemes can evaluate two types of gates, but only for a subset of circuits.
  * Leveled fully homomorphic encryption-supports the evaluation of arbitraty circuits composed of multiple types of gates of bounded (pre-determined) depth.
  *  Fully homomorphic encryption (FHE)-allows the evaluation of arbitrary circuits composed of multiple types of gates of unbounded depth and is the strongest notion of homomorphic encryption.
* For the majoirty of homomorphic encryption schemes, the multiplicative depth of circuits is the main practical limitation in performing computations over encrypted data.
* Homomorphic encryption schemes are malleable, meaning they have weaker security properties than non-homomorphic schemes.
## Partially Homomorphic Cryptosystems
* Unpadded RSA-if the RSA public key has modulus n and encryption exponent e, then the encryption of a message m is given by E(m)=m^e mod n. The homomorphic property is then..
  * E(m1) * E(m2) = m1eme2 mod n
  * =(m1m2)^e mod n
  * E(m1 * m2)
*  Benaloh-in the Benaloh cryptosystem, if the public key is the modulus n and the base g with a blocksize of c, then the encryption of a message m is E(m)=g^mr^e mod n, for some random r in {0,...,n-1}. The homomorphic property is then....
  * E(m1) * E(m2) = (g^m1*r^c1)(g^m2*r^c2) mod n
  * =g^(m1+m2) * (r1r2)^c mod n
  * =E(m1 + m2 mod c)
* Paillier-in the Paillier cryptosystem, if the public key is the modulus n and the base g, then the encryption of a message m is E(m)=g^m* r^n mod n^2, for some random r in {0,..., n-1}. The homomoprhic property is then....
  * E(m1) * E(m2) = (g^m1* r^1n)(g^(m2) * r^(n)2) mod n^2
  * = g^(m1+m2) * (r1r2)^n mod n^2
  * = E(m1 + m2).
  ## Fully Homomorphic Encryption
  * A cryptosystem that supports arbitrary computation on ciphertexts is known as fully homomorphic encryption (FHE). This enables construction of programs for any desirable functionality, which can be run on encrypted inputs to produce an encryption of the result. Since this program doesn't need to decrypt its inputs, it can be run by an untrusted party without revealing its inputs and internal state.
  * Fully homomorphic cryptosystems have great practical implications in the outsourcing of private computations, for instance, in cloud computing.
## Implementations
* There are several open-source implementations of fully homomorphic encryption schemes. Second-generation and fourth-generation FHE scheme implementations typically operate in the leveled FHE mode (though bootstrapping is still available in some libraries) and support efficient SIMD (Single-Instruction, Multiple Data) packing of data. They are typically used to compute on encrypted integers or real/complex numbers.
* Third-generation FHE scheme implementations often bootstrap after each operation but have limit support for packing; they were at first used for computing Boolean circuits over encrypted bits, but have been extended to support integer arithmetics and univariate function evaluation. The choice of using a second-generation vs. third or fourth depends on input data types and desired computation.
* In addition, FHE has been combined with zero knowledge proofs, blockchain technology that can prove something is true without revealing private information.
* zkFHE enables data encryption throughout data processing, while the results of any processing are verified in a confidential manner.
## FHE Libraries
* HElib-open source, developed by IBM, utilized BGV (background verfication), and CKKS (fourth generation of FHE).
* OpenFHE-developed by Samsung Advanced Institute of Technology, Intel, MIT, and more. Uses BGV, CKKS, CKKS bootstrapping, and FHEW
* PALISADE-developed by the New Jersey Institute of Technology, Duality Technologies, Raytheon BBN Technologies, MIT, and more, uses BGV and CKKS
* Source: https://en.wikipedia.org/wiki/Homomorphic_encryption#:~:text=Homomorphic%20encryption%20is%20a%20form%20of%20encryption%20with%20an%20additional,extension%20of%20public%2Dkey%20cryptography.

# Combining Machine Learning and Homomorphic Encryption in the Apple Ecosystem
* One of Apple's privacy principles is to prioritize using on-device processing. By performing computations locally on a user's device, we minimize the amount of data that is shared with Apple or other entities.
* A user might request on-device experiences powered by machine learning (ML) that can be enriched by looking up global knowledge hosted on servers.
* A major technology that Apple uses to to keep server lookups safe is homomorphic encryption (HE), a form of cryptography that enables computation on encrypted data. HE is designed so a client device encrypts a query before sending it to a server, and the server operates on the encrypted query and generates an encrypted response, which the client then decrypts. The server does not decrypt the original request or even have access to the decryption key, so HE is designed to keep the client query private throughout the process.
* At Apple, HE is used alongside other privacy-preserving technologies, and a number of optimizations and techniques balance the computational overhead of HE with the latency and efficiency demands of production applications at scale.
* Intoducing HE into the Apple ecosystem provides the privacy protections that make it possible for us to enrich on-device experiences with private server look-ups, and to make it easier for the developer community to adopt HE for their own applications, Apple open sourced swift-homomorphic-encryption, an HE library.

## Apple's Implementation of Homomorphic Encryption
* This allows operations common to ML workflows to run efficiently at scale, while achieving an extremely high level of security.
* The Brakerski-Fan-Vercauteren (BFV) HE scheme supports homomorphic operations that are well suited for computation (dot products, cosine similarity) on embedding vectors that are common to ML workflows. BFV parameters used achieve post-quantum 128-bit security, meaning they provide strong security against both classical and potential future quantum attacks.
* HE excels in settings where a client must look up information on a server while keeping the lookup computation encrypted.
* HE alone enables privacy presrving server look up for exact matches with private information retrieval (PIR), and then we describe how it can serve more complex applications with ML when combining approximate matches with private nearest neighbor search (PNNS).

## Private Information Retrieval (PIR)
* Many use-cases require a device to privately retrieve an exact match to a query from a server database, like retrieving the appropriate business logo and information to display with a received email or provide caller ID information. To protect privacy, the relevant information should be retrieved without revealing the query itself, for example in these cases the business that emailed the user, or the phone number that called the user.
* For these workflows, private information retrieval (PIR), a form of private keyword-value database lookup is used.
* A client has a private keyword and seeks to retrieve the associated value from a server, without downloading the entire database.
* The client encrypts its keyword before sending it to the server, HE computation is performed between the incoming ciphertext and its database, and sends the resulting encrypted value back to the requesting device, which then decrypts it to learn the value associated with the keyword. The server doesn't learn the client's private keyword or the retrieved result, as it operates on the client's ciphertext.
* For instance, with web content filtering, the URL is encrypted, sent to the server, encrypted computation is performed on the ciphertext with URLs in its database, the encrypted result is sent to the device, where it is decryped to identify if the website should be blocked as per the parental restriction controls.

## Private Nearest Neighbor Search (PNNS)
* For use-cases requiring an approximate match, use Apple's private nearest neighbor search (PNNS), an efficient private database retrieval process for approximate matching on vector embeddings. With PNNS, the client encrypts a vector embedding and sends the resulting ciphertext as a query to the server. The server performs HE computation to conduct a nearest neighbor search and sends the resulting encrypted values back to the requesting device, which decrypts to learn the nearest neighbor to its query embedding.
* Similar to PIR, throughout this process, the server does not learn the client's private embedding or the retrieved results, as it operates on client ciphertext.
* Combining PIR, PNNS, and HE allows on-device experiences that leverage information from large server-side databases while protecting user privacy.

## Implementing These Techniques in Production
* Enhanced Visual Search for photos allows users to search their photo library for specific lovations, like landmarks and points of interest.
* Using PNNS, a user's device privately queries a global index of popular landmarks and points of interest maintained by Apple to find approximate matches for places depicted in their phot library.
* Users configure the above feature on their device using Settings->Photos -> Enhanced Visual Search
* An on-device ML model analyzes a given photo to determine if there is a "region of interest" (ROI) that may contain a landmark. If the model detects an ROI in the "landmark" domain, a vector embedding is calculated for that region of the image. The dimension and precision of the embedding affects the size of the encrypted request sent to the server, the HE computation demands and the response size, so to meet the latency and cost requirements of large-scale production services, the embedding is quantized to 8-bit precision before being encrypted.
* The server database the client is sent to is divided into disjointed subdivisions/shards of embedding clusters, reducing computational overhead and increasing efficiency of the query, as the server can focus HE computation on just the relevant portion of the database. A cluster codebook containing the centroids for the cluster shards is available on the user device, enabling the client to locally run a similarity search to identify the closest shard for embedding, which is added to the encrypted query and sent to the server.
* Identifying the database shard relevant to the query could reveal sensitive information about the query itself, so differential privacy (DP) with OHTTP relay is used as an anonymization network to ensure the server can't link multiple requests to the same client.
* The servers handling these queries leverage Apple's existing ML infrastructure, including a vector database of global landmark image embeddings, expressed as an inverted index. The server identifies the relevant shard based on the index in the client query and uses HE to compute the embedding similarity in this encrypted space. The encrypted scores and set of corresponding metadata (like landmark names) for candidate landmarks are then returned to the client.
* All similarity scores are merged into one ciphertext for efficiency
* The client decrypts the reply to its PNNS query, which may contain multiple candidate landmarks. A specialized, lightweight on-device reranking model predicts the best candidate by using high-level multimodel feature descriptors, including visual similarity scores, locally stored geo-signals, popularity, and index coverage of landmarks. When the match is identified, the photo's local metadata is updated with the landmark label and the photo can easily be found by the user when searching their device for the landmark's name.

## Conclusion
* By implementing HE with a combination of privacy-preserving technologies like PIR and PNNS, on-device and server-side ML models, and other privacy preserving techniques, we are able to deliver features like Enhanced Visual Search without revealing to the server any information about a user's on-device content and activity.
