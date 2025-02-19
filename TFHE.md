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

# Encodings and Linear Leveled Operations
* GLWE ciphertexts are the main focus.
* Encodings are an overlayer of the encryption and very useful in FHE.
## GLWE Homomorphic Addition
* Let p and q be two positive integers, such that p<=q and △ = q/p.
* p and q are either chosen to be powers of 2 or rounded at the moment of encoding
* A GLWE ciphertext encrypting a message M ∈ Rp under the secret key S=(S0,..., Sk-1) ∈ R^k is a tuple where the elements Ai for i ∈|0...k-1| are sampled uniformly random
                k-1
from Rq and B = Σ Ai * Si + △M + E∈Rq has coefficients sampled from a Gaussian dist Xσ
                i=0
*  If another GLWE ciphertext encrypts a different message under the same key, we can add every component of the two ciphertexts (in Rq) and the result will be a new GLWE ciphertext encrypting the sum M + M' ∈ Rp under the same secret key s, with noise that grew a little bit (additively with respect to the original noises in C and C') and that we will estimate with standard deviation σ'
*  Homomorphic addition between cyphertexts is noted with "+"
### Toy Example
* Using q = 64, p = 4 so △ = 64/4=16, N=4 and k=2, sample the secret key with uniform binary distribution as k polynomials of degree smaller than N
 * ->S = (S0, S1) = (X + X^2, 1+X^2 + X^3) ∈R^2
* Encrypt two messages:
 * M = -2 + X-X^3 ∈ Rp
 * M' = X + X^2 - 2X^3
* Their addition equals....
 * M(+) = -2 -2X + X^2 + X^3 ∈ Rp
* Choose A, E, A', E' (the randomness for the first and second messages), compute the bodies (B and B'), and after performing polynomial operations modulo X^N +1 (X^4 +1) and modulo q (64), we get the encryptions C and C'
* To perform _homomorphic addition_, add in Rq the components term wise:
 * A0^(+) = A0 + A'0 = 17-2X-24X^2 + 9X^3 - 8 + 15X + 3X^2 - 30X^3 - 9 + 13X - 21X^2 -21X^3 ∈Rq
 * A1^(+) = A1 + A'1 = -14 - X^2 + 21X^3 + 23 - 16X + 27X^2 - 4X^3 = 9 - 16X + 26X^2 + 17X^3 ∈ Rq
 * B(+) = B + B' = -31 + 5X - 21X^2 + 30X^3 - 25 + 12X^2 - 12X^3 = 8 + 5X - 9X^2 + 18X^3 ∈ Rq
* Therefore:
 * C^(+) = A0^(+), A1^(+), B^(+)) = (9 + 13X - 21X^2 - 21X^3, 9 - 16X + 26X^2 + 17X^3, 8 + 5X - 9X^2 + 18X^3) ∈ R^3q
                       k-1
* Decryption ( B^(+) - Σ Ai^(+) * Si = 31 - 30X + 15X^2 + 16X^3 ∈ Rq
                       i=0
* Produces M^(+)

## GLWE Homomorphic Multiplication by a Constant
*  Let's consider a GLWE ciphertext encrypting a message M ∈ Rp under secret key ->S = (S0,...,Sk-1)∈R^k
*  We can multiply a small constant polynomial /\ to every component of the ciphertext and the result will be a new GLWE ciphertext encrypting the product  /\ * M ∈ Rp under the same key ->S, with noise that grew proportionally with respect to the size of the coefficients of /\ and estimated with standard deviation σ''
*  Homomorphic multiplication is noted with the symbol "."
*  Homomorphic mutliplication can also be performed with a small integer
### Toy Example
* Using the same message M as the previous example, choose a small constant polynomial (/\ = -1 + 2X^2 + X^3 ∈ R).
* The multiplication between M and /\ is equal to...
 * M^(.) = 1 + X + X^2 + X^3 ∈ Rp
* To perform the homomorphic multiplication by /\, multiply in Rq the components of the ciphertext C times /\.
*  C(.)=(A0(.), A1(.), B(.))= (-31 + 8X - 15X^2 + 4X^3, 16 + 23X + 16X^2 + 29X^3, 4 + 20X - 7X^2 + 13X^3) ∈ R^3q

## GLWE Encodings
* An _encoding_ is the way we decide to represent a message inside the ciphertext, which impacts how homomorphic operations are performed
### Encoding Integers in the MSB
* In TFHE, the noise is added in the LSB (least significant bits), so an encoding must position the message in the MSB (most significant bits)
* Remember that for LWE ciphertext, the encoded quantity is a plaintext, the message is a cleartext, small noise is sampled from a Gaussian distribution Xσ.
* Encoding the error as a positive value in the LSB is pratical to make _leveled operations_ (additions and multiplications by constants) _modulo_ p.
* If you add two LWE ciphertexts encrypting two messages encoded in the MSB with the same △, a new ciphertext with more noise results.
### Encoding Integers in the MSB with Padding Bits
* This encoding is largely used in TFHE and consists in encrypting the message in the MSB, but to add some bits of padding (EX: bits set to zero) in the MSB part to "provide space" for leveled operations (like homomorphic addition and multiplication by constants).
* "Empty space" between messages is called _padding_.
* This encoding is practical to make _exact leveled operations_ (additions and multiplications by constants). If you add two LWE ciphertexts encrypting two messages m1 and m2 encoded in the MSB with the same △ and use 2 bits of padding, the result will be a new LWE encrypting m1+m2 in the MSB with 1 bit of padding that was "consumed" after the addition, along with increased noise.
### Encoding of Reals
* In this, the message and the error become a single thing. Message m occupies the entire space Zq, but the LSB are perturbed by an error e, which "approximates" the information. There is no way to distinguish the exact value of m because there is no △ value helping do the separation with the error part.
* This encoding method is practical to evaluate _approximate leveled operations_ (additions and multiplications by constants) _up to a certain precision_
* For decryption, the first step is the same, but the second is replaced by a rounding or the addition of a new random error in the LSB.
* "T" in TFHE stands for "Torus", a circular mathematical structure that can help us visualize encodings
* Source: https://www.zama.ai/post/tfhe-deep-dive-part-2

# Key Switching and Leveled Multiplications
## Homomorphic Multiplication by a Large Constant
* If we multiply every component of ciphertext by a large polynomial (in Rq), noise grows proportionally with respect to the polynomial's size, so the noise grows too much and compromises the result.
* To solve the noise problem, take the large constant and decompose it into a _small base_ B, where the decomposed small elements are in ZB. These should be chosen as powers of two.
* Once we have the small decomposition elements, we should be able to perform the multiplication with the ciphertext and have a small impact on noise. However, to obtain the product of the polynomial and the message, we must recompose the polynomial
* To recompose, multiply the decomposed elements by the GLev encryption of M (which encrypts M times different powers of the decomposition base).
* An inner-product-like operation multiplies every element of the decomposition times the corresponding element of the GLev and adds them all together.
* The issue is GLWE in output no longer has △, so the new message might occupy the entire space. As such, this method is used as a building block for more complex operations, never on its own.

## Approximate Decomposition
* Approximate decomposition decomposes up to a fixed precision, meaning we do a rounding in the LSB before decomposing; if the decomposition parameters are chosen properly, this does not impact the correctness of the computations.

## Multiplication by a Large Polynomial
* The polynomial is decomposed into smaller polynomials and you then perform a polynomial inner product with the GLev.
* This operation is the main building block operation for key switching and homomorphic multiplication.
### Toy Example
* This displays how to decompose a large polynomial using approximate signed decomposition, which is what is used in practice.
* Let's choose a base for the decomposition B = 4 and l = 2, so B^l = 16. This means we will decompose the 4 MSB of each coefficient, but first they need to be rounded. Write them in binary decomposition first (MSB left, LSB right) and perform the rounding of the 2 LSB
 * /\0 = 28->(0,1,1,1,0,0) which becomes /\'0->(0,1,1,1)
 * /\1 = -5->(1,1,1,0,1,1) which becomes /\'1->(1,1,1,)
 * /\2 = -30->(1,0,0,0,1,0) which becomes /\'2 ->(1,0,0,1)
 * /\3 = 17 ->(0,1,0,0,0,1) which becomes /\'3 ->(0,1,0,0)
* For decomposition, start from the LSB and because the base is 4, extract 2 bits at every round, and we want coefficients in {-2, -1, 0, 1}. If we have 1,1, corresponding to 3, subtract 4 to the block and add +4 to the next block, like a carry.
* In /\' -> (0, 1, 1, 1):
 * The two LSB are (1,1), corresponding to 3, subtract 4 and get -1 at the first element of the decomposition
 * The next block is (0,1), add back the subtracted 4 and get 1,0, corresponding to 2. Once again subtract 4, getting -2 as the second element, and throwing the +4 out
* In /\'1 -> (1, 1, 1, 1)
 * (1, 1) corresponds to 3, subtract 4 and get -1
 * Add 4 to (1, 1) and get (0,0), which corresponds to 0
* In /\'2 ->(1, 0, 0, 1)
 *  (0, 1) corresponds to 1, our first element
 *  (1, 0) corresponds to 2, subtract 4 and get -2
* In /\'3 -> (0, 1, 0, 0):
 * (0, 0) corresponds to 0
 * (0, 1) corresponds to 1
* We can write the decomposed polynomials as...
 * /\(1) = -2 -2X^2 + X^3
 * /\(2) = -1 -X + X^2x
* The coefficients of /\(2) are the first elements of the decomposition, while the coefficients of /\(1) are the second elements of the decomposition. These polynomials can be used in the inner products with the GLev ciphertexts.

## Key Switching
* Ciphertexts are in fact large vectors, polynomials, vectors of polynomials, composed by integers modulo q, that look uniformly random
* Compining decomposition and inner products with GLev ciphertexts is useful for defining more complex operations and multiplications between ciphertexts.
* _Key Switching_ is the first operation, and its a homomorphic operation largely used in all the (Ring) LWE-based schemes and is used to switch the secret key to a new one
* To switch a key, _cancel_ the secret key ->S and _re-encrypt_ under a new secret key ->S', and try to do so homomorphically.
* The ciphertext will be the GLev encryption of Si and the large constant will be Ai that, by construction, is a uniformly random polynomial in Rq.
* In practice, the GLWE encryption involves the trivial GLWE, which subtracts the GLWE encryption of AiSi (involving decomposition).
* This corresponds to the homomorphic evaluation of the first step of the GLWE decryption, but since we don't evaluate the second step (re-scaling by △ and rounding) we do not reduce the noise. The noise is larger than that of the input ciphertext.
* Some types of key switching include...
 * Key switching from one LWE to one LWE
 * Key switching from one RLWE to one RLWE
 * Key switching from one LWE to one RLWE, putting the message encrypted in the LWE into one of the coefficients of the RLWE ciphertext
 * Key switching from many LWE to one RLWE, packing the messages encrypted in the many LWE inputs into the RLWE ciphertext.
* Key switching can also be used to switch parameters.

## External Product
* Our goal is to homomorphically multiply two ciphertexts so the result is an encryption of the product of messages.
* We take one of the two ciphertexts as a GLWE (the ciphertext we will decompose) and the other will be a list of GLev ciphertexts. This time we want to mask (like in key switching) and mask to be multiplied by the GLev.
* For an operation to multiply two ciphertexts (a GLWE and a GGSW) and return a new GLWE ciphertext, the inputs are
 * A GLWE ciphertext encrypting a message M1∈Rp under the secret key ->S, where the elements Ai for i ∈[0..k-1] are sampled uniformly random from Rq, and B has coefficients sampled from a Gaussian distribution Xσ
 * A GGSW ciphertext encrypting a message M2∈Rp under the same secret key, multiplied with GLev
 * The external product is noted with the symbol . and is computed as seen above.
 * The noise in the result is larger than that in the input

## External Product vs. Key Switching
* The external product is like a key switching with an additional element to the key switching key (the Ck GLev ciphertext)
* The external product is like a key switching where we do not switch the key. In the GGSW ciphertext, the secret key used for encryption and the one used inside the GLev ciphertexts are the same
* An external product that takes as input a GGSW ciphertext, that uses a different secret key for encryption and used the same secret key as the GLWE ciphertext inside the encryption, is called _functional key switching_. It applies a function (multiplication by an encrypted constant) and switches the key at the same time.

## Internal Product
* There is no internal product between GLWE ciphertexts that can be done in a straight way. There is however an internal product that can be defined from the external product
* A GGSW ciphertext is a list of GLev ciphertexts, and wach GLev is a list of GLWE ciphertexts.
* We can define the internal product as a list of independent external products between one of the GGSW ciphertexts in input and all the GLWE ciphertexts composing GGSW input. The result of all these external products will be the GLWE ciphertexts composing the GGSW output.

## Internal Product vs External Product
* The external product is more efficient than the internal product, but it is not composable, so the result of an internal product can be used as input of another internal product, but the result of an external product (GLWE ciphertext) can be used only in one of two inputs of another external product (the GLWE one) but not on the other (the GGSW one). If we use external products, the GGSW is fresh and encrypted

## CMux
* The CMux operation is the homomorphic version of a multiplexer (Mux) gate.
* A Mux gate is an if condition; it takes three inputs, a selector, and two options, and depending on the value of the selector, it makes a choice between the options.
* It is evaluated in clear by computing b * (d1-d0) + d0 = db
* To evaluate it homomorphically, encrypt b as a GGSW ciphertext and d0 and d1 as GLWE ciphertexts. Then the multiplication in the cleartext formula is evaluated as an external product, while other operations (addition and subtraction) are evaluated as homomorphic additions and subtracts, encrypting db as a GLWE ciphertext.
* Source: https://www.zama.ai/post/tfhe-deep-dive-part-3

# Programmable Bootstrapping
*
* Source: https://www.zama.ai/post/tfhe-deep-dive-part-4
