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
         k-1
 1. B -  Σ Ai * Si = △M + E ∈ Rq
         i=0
 2.  [(△M + E/△]=M
* If every coefficient ei of E is |ei|<△/2), then the second step of the decryption returns M as expected. If the error does not represent the condition, the decryption is incorrect.
* To ensure FHE operations work as expected, we have to find a way to reduce excess noise. This is known an "bootstrapping."
### Toy Example
* For our parameters, q=64, p=4 (△=q/p=16), N=4 and k=2
* The secret key is sampled with uniform binary distribution as k polynomials of degree smaller than N:
 * ->S = (S0, S1) = (0 + 1 * X + 1 *X^2 + 0 * X^3, 1 + 0 * X + 1 * X^2 + 1 * X^3) ∈R^2
* Encrypting message M (M=-2 + X - X^3) requires sampling a uniformly random mask with coefficients in {-32.....31} (△ * k = 32)
 * ->A = (A0, A1)=17-2 * X - 24 * X^2 + 9 * X^3, -14 + 0 * X - 1 * X^2 + 21 * X^3) ∈R^2q
*  And a discrete Gaussian Error (small coefficients):
 * E = -1 + 1 * X + 0 * X^2 + 1 * X^3 ∈Rq
* To encrypt, compute the body as:
 * B = A0 * S0 + A1 * S1 +  △M + E ∈Rq
* When we compute in Rq, we perform polynomial operations modulo X^N + 1 and modulo q. To reduce modulo X^N + 1, you can observe than X^N = X^4 = -1 mod X^4 +1, so...
 * A0 * S0 = (17 - 2X - 24X^2 + 9X^3) * (X + X^2)
 * = 17X + (17-2)X^2 + (-2 -24)X^3 + (-24 + 9)X^4 + 9X^5
 * = 17X + 15X^2 - 26X^3 + 15 - 9X
 * = 15 + 8X + 15X^2 - 26X^3 ∈Rq
* In the same way:
 * A1 * S1 = (-14 - X^2 + 21X^3) * (1 + X^2 + X^3) = -13 -20X +28X^2 + 7X^3 ∈Rq
* Observe that...
 * △M = -32 + 16 * X + 0 * X^2 -16 * X^3 ∈Rq
* Then:
 * B= A0 * S0 + A1 * S1 + △M + E = -31 + 5X - 21X^2 + 30X^3 ∈Rq
* So the encryption is:
 * (A0, A1, B) = (17 - 2X -24X^2 + 9X^3, -14 - X^2 + 21X^3, -31 + 5X -21X^2 + 30X^3) ∈R^3q                                k-1
* When we decrypt by computing  B -  Σ Ai * Si = △M + E ∈ Rq we find 31 + 17X - 15X^3
                                     i-0
* Finally, [(31 + 17X - 15X^3)/16]= -2 + X - X^3 ∈ Rp, which is the encrypted message. Decryption worked because error coefficients were all smaller in absolute value than △/2=8

## Trivial GLWE Ciphertexts
* These are not true encryptions, as they are placeholders. They have the shape of a GLWE ciphertext but the message is in clear. A trivial ciphertext of a message M has all the Ai set to 0 and the B equal to △M. These can be used to inject publicly known data in homomorphic encryptions.
### LWE and RLWE
* When we instantiate GLWE with k=n∈Z and N=1 we get LWE. Observe that Rq (resp. R) is actually Zq (resp. Z) when N=1
* When we instantiate GLWE with k=1 and N a power of 2 we get RLWE.
### Public Key Encryption
* In practice, a public key would be a list of encryptions of zero (EX: M=0). To encrypt a message, it's sufficient to take a random combination of encryptions of zero and add the desired message △M.

## GLev
* GLev ciphertexts can be seen as a generalization of Powers of 2 encrpyions used in BGV
* A GLev ciphertext contains redundancy; it is composed by a list of GLWE ciphertexts encrypting the same message M with different, and very precise, scaling factors △. Two parameters define these △s, a base B, power of two, and number of levels l ∈ Z
* If B and q are not powers of 2, a rounding should be applied at the moment of encoding. The secret key is the same as for GLWE ciphertexts. To decrypt, it is sufficient to decrypt one of the GLWE ciphertexts with the corresponding scaling factor.
* GLWE is a generalization for both LWE and RLWE, and GLev can be specialized into Lev and RLev

## GGSW
* A GLWE ciphertext is a vector of elements from Rq (or a 1 dimensional matrix)
* A GLev ciphertext is a vector of GLWE ciphertexts (or a 2 dimensional matrix of elements from Rq)
* A GGSW ciphertext is a vector of GLev ciphertexts (or a 3 dimensional matrix of elements from Rq or a 2 dimensional matrix of GLWE ciphertexts).
* In a GGSW each GLev ciphertext encrypts the product between M and one of the polynomials of the secret key -Si. The last GLev in the list just encrypts the message M
* The secret key is the same as for GLWE and GLev ciphertexts. To decrypt, it is sufficient to decrypt the last GLev ciphertext. The set of GGSW encryptions of the same message M, under the secret key ->S, with Gaussian noise with standard deviation σ, with base B and level l, will be noted...
        Bl
 * GGSW -> (M)
        S, σ
* Source: https://www.zama.ai/post/tfhe-deep-dive-part-1
