# TFHE Deep Dive: Ciphertext Types
* TFHE is also know as CGGI, from the authros Chillotti-Gama-Georgieva-Izabachene
* TFHE is a Fully Homomorphic Encryption Scheme; it allows you to perform computations over encrypted data.
* The security of the scheme is based on a hard lattice problem called Learning With Errors, or LWE (as are most FHE schemes now).
* TFHE proposes a special bootstrapping which is very fast and able to evaluate a function at the same time as it reduces the noise.
## Notation
* R = Z|X|/(X^N + 1) is the ring of integer polynomials modulo the cyclotomic polynomial X^N + 1, with N power of 2. It contains integer polynomials up to degree N - 1.
* Rq = (Z/qZ)|X|/(X^N + 1) takes the ring above but the coefficients are modulo q.
* Our modular reductions are centered around zero. For instance, when reducing modulo 8, the congruence classes {-4, -3, -2, -1, 0, 1, 2, 3} are used
* Xμσ is a Gaussian probability distribution with mean μ and standard deviation σ. If μ=0, simply note Xσ
* Use small letters for modular integers and capital letters for polynomials
* Note the list of integer elements from a∈Z to b∈Z included as [a..b]
* MSB = Most Significant Bit
* LSB = Least Significant Bit
* Denote with [.] the rounding operation to the nearest integer value.

## TFHE Ciphertexts
* TFHE uses three main types of ciphertexts, each having different yet important security properties. Said ciphertexts are...
  * GLWE (General LWE) - a generalization for both LWE and RLWE ciphertexts
  * GGSW (General GSW) - a generalization for RGSW ciphertexts
  * GLev - an intermediate ciphertext type that will be very useful to better understand GGSW ciphertexts
 
## GLWE
* To generate ciphertext, a secret key is needed. With GLWE ciphertexts, the secret key is a list of k random polynomials from R, and the coefficients of said elements can be sampled from a uniform binary distribution, uniform ternary distribution, Gaussian distribution, or uniform distribution.
* For this example, assume the secret keys are sampled from uniform binary distributions
* To encrypt a message, let p and q be positive integers, such that p<=q and △=q/p. In TFHE, p and q are often chosen to be powers of two (and should be rounded at encoding if they are not).
* q=ciphertext modulus
* p=plaintext modulus
* △=scaling factor
* A GLWE ciphertext encrypting the message M under the secret key S is a tuple where elements are sampled uniformly random from Rq and B, and E∈Rq has coefficients sampled from a Gaussian distribution xσ.
* Tuple: (A0......, Ak-1, B)∈GLWEs,σ(△M)⊆Rq^(k+1)
* A0...Ak-1 is _the mask_, B is _the body_,  △M is an _encoding_ of M.
* To compute △M, we lift the message M as an element of Rq, and every time a message is encrypted, new randmoness is sampled (mask and noise error) so every encryption, even of the same message, is different from the other, ensuring security.
* The ciphertext can be _decrypted_ by computing...
