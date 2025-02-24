# HEAAN
* HEAAN is Homomorphic Encryption for Arithmetic of Approximate Numbers
* This open source homomorphic encryption library implements an approximate CKKS scheme.
## CKKS Plaintext Space
* CKKS supports approximate arithmetic over complex numbers. Its plaintext space is C^(n/2) for some power-of-two integer n.
* This plaintext encoding-decoding method exploits a ring isomorphism R[X]/(X^n + 1)-> C^(n/2)
### Encoding Method
* Given a plaintext vector and scaling factor, the plaintext vector is encoded as a polynomial m(X):=Z[X}/(X^n + 1) by computing m(X)=[Δ * ring^-1(z)] where * denotes the coefficient-wise rounding function
### Decoding Method
* The message polynomial is decoded to a complex vector by computing z=Δ^-1 * ring(m(X))
* The scaling factor lets us control encoding/decoding error.

## Algorithms
* The CKKS scheme consists of key generation, encryption, decryption, homomorphic addition and multiplication, and rescaling.
* For positive integer q....
  * Rq :=R/qR is the quotient ring of R modulo q.
  * Xs, Xr, and Xe are distributions over R which output polynomials with small coefficients.
  * Distributions, modulus Q, and ring dimension n are predetermined before the key generation phase.

### Key Generation
1. Sample a secret polynomial s<-Xs
2. Sample a (resp. a') uniform randomly from RQ (resp. RPQ) and e, e'<-xe
3. Output a secret key sk <-(1, s) in R^2Q,
4. Output a public key pk <-(b=-a * s + e, a) in R^2Q
5. Output an evaluation key evk <- (b' = -a' * s + e' + P * s^2, a') in R^2PQ

### Encryption
1. Sample an ephemeral secret polynomial r <- Xr
2. For a given message polynomial m in R, output a ciphertext ct <- (c0 = r * b + e0 + m, c1 = r * a +e1) in R^2Q

### Decryption
1. For a given ciphertext ct in R^2q, output a message m' <- (ct, sk) (mod q)
2. Decryption outputs an approximate value of the original message (EX: Dec(sk, Enc(pk, m)) = m, and the approximation error is determined by the choice of distributions Xs, Xe, Xr.

### Rescaling
1. Given a ciphertext ct in R^2q and a new modulus q' < q, output a rescaled ciphertext ctrs <- [(q'/q) * ct] in R^2q

## Security
* The total procedure of the CKKS scheme is as following...
  * Each plaintext vector z which consists of complex (or real) numbers is firstly encoded as a polynomial m(X) in R by the encoding method, then encrypted as a ciphertext. After several homomorphic operations, the resulting ciphertext is decrypted as a polynomial m'(X) in R and decoded as a plaintext vector z
* The security of the CKKS scheme is based on the hardness assumption of the ring learning with errors (RLWE) problem, ring variant of the lattice-based Learning with Errors problem.

* Source: https://en.wikipedia.org/wiki/HEAAN
