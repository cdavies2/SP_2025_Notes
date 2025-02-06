# Homomorphism
* A structure-preserving map between two algebraic structures of the same type (EX: two groups, two matrices).
* Homomorphisms of vector spacs are also called linear maps, and their study is the focus of linear algebra.
* A homomorphism may also be an isomorphism, an endomorphism, or an automorphism.

## Definition
* A map between two algebriac structures of the same type (like fields or vector spaces) that preserves the operations of the structures. This means that a map f: A -> B between two sets A, B equipped with the same structure such that, if * is an operation of the structure, then f(x * y) = f(x) *  f(y) for every pair x, y elements of A.
* The operations that must be preserved by a homomorphism include 0-ary operations, that is the constants. In particular, when an identity element is required by the type of structure, the identity element of the first structure must be mapped to the corresponding identity element of the second structure. For instance....
  * A semigroup homomorphism is a map between semigroups that preserves the semigroup operation.
  * A monoid homomorphism is a map betwen monoids that preserves the monoid operation and maps the identity element of the first monoid to that of the second monoid
  * A linear map is a homomorphism of vector spaces (a group homomorphism between vector spaces that preserves the abelian group structure and scalar multiplication)
  * A module homomorphism (aka linear map betwen modules) is defined similarly
  * An algebra homomorphism is a map preserving algebra operations
* An algebraic structure may have more than one operation, and a homomorphism is required to preserve each operation. Therefore, a map preserving only some operations is not a homomorphism of the entire structure, but only a homomorphism of the substructure obtained by considering only the preserved operations. For instance, a map between monoids that preserves the monoid operations and not the identitiy element is not a monoid homomorphism, but only a semigroup homomorphism.
* The notation of operations does not need to be the same in the source and target of a homomorphism. The real numbers form a group for addition, positive real numbers form a group for multiplication. The exponential function x->e^x satisfies e^(x+y)=e^xe^y, and thus is a homomorphism between these two groups.

## Examples
* The set of all 2x2 matrices is a ring, under matrix addition and matrix multiplication
* If we define a function between these rings as follows:
  * f(r) ( r      0
           r      0)
* Where r is a real number, then f is a homomorphism of rings, since f preserves both addition:
  * f(r+s)=( r+s  0) =  ( r  0)  +  ( s  0) = f(r) + f(s)
             0  r+s       0  r        0  s
* And multiplication:
  * f(rs)= ( rs  0)  =  ( r  0) ( s  0) = f(r)f(s)
             0  rs        0  r    0  s
* Source: https://en.wikipedia.org/wiki/Homomorphism

 # Modular Arithmetic: Modular/Barrett Reduction
 * In modular arithmetic, Barrett reduction is an algorithm designed to optimize calculation of a mod n without needing a fast division algorithm. It replaces divisions with multiplications, and can be used when n is constant and a<n^2
 * A function[]: R -> Z is an integer approximation if |[z]-z|<=1. For modulus n and an integer approximation [], we define mod[] n: Z -> (Z/nz) as....
  * amod^[]n = a - [a/n]n.
  * Common choices for [] are floor, ceiling, and rounding functions.
 * Source: https://en.wikipedia.org/wiki/Barrett_reduction
## Montgomery Modular Multiplication
* Montgomery modular multiplication relies on a special representation of numbers called Montgomery form. The algorithm uses the Montgomery forms of a and b to efficiently compute the Montgomery form of ab mod N. The efficiency comes from avoiding expensive division operations.
* Classical modular multiplication reduces the double-width product ab using division by N and keeping only the remainder. This division requires quotient digit estimation and correction. The Montgomery form, in contrast, depends on a constant R>N, which is coprime to N, and the only division necessary in Montgomery multiplication is division by R. The constant R can be chosen so that division by R is easy, significantly improving the speed of the algorithm. R is always a power of two, since division by powers of two can be implemented by bit shifting.
* The need to convert a and b into Montgomery form and their product out of Montgomery form means that computing a single product by Montgomery multiplication is slower than the conventional or Barrett reduction algorithms.
* When performing many multiplications in a row, as in modular exponentiation, intermediate results can be left in Montgomery form, Then the initial and final conversions become a negligible fraction of the overall computation.
* Many important cryptosystems like RSA and Diffie-Hellman key exchange are based on arithmetic operations modulo a large odd number. For these cryptosystems, using Montgomery multiplication with R a power of two are faster than alternatives.
## Montgomery Form
* If a and b are integers in the range [0, N-1], then their sum is in the range [0, 2N-2] and their difference is in the range [-N+1, N-1], so determining the representative in [0, N-1] requires at most one subtraction or addition (respectively) or N. The product ab is in the range [0, N^2 - 2N + 1], so storing it requires twice as many bits as either a or b, and determining the representative in [0, N-1] requires division, which is expensive.
* Montgomery fom expresses elements of the ring in which modular products can be computed without expensive divisions (the divisor chosen is a power of two, meaning division can be replaced by shifting or omitting machine words, which is done quickly).
* Montgomery form of the residue class "a" with respect to R is aR mod N.
 * EX: Suppose that N = 17 and that R = 100. The Montgomery forms of 3, 5, 7, and 15 are 300 mod 17 = 11,  500 mod 17 = 7, and 1500 mod 17 = 4.
*  Addition and subtraction in Montgomery form are the same as ordinary modular addition and subtraction because of the distributive law.
 *  aR + bR = (a + b)R,
 *  aR - bR = (a - b)R
* For instance, (7 + 15) mod 17 =5, which in Montgomery form is (3 + 4) mod 17=7
* Multiplication in montgomery form requires removing a factor of R.
* The product of the Montgomery forms of 7 and 15 modulo 17, with R = 100, is the product of 3 and 4, which is 12. 12 is not divisible by 100, so additional effort is required to remove the extra factor of R.
* Removing the extra factor of R can be done by multiplying by an integer R' such that RR' -= 1 (mod N), that is, by an R' whose residue class is the modular inverse of R mod N.
* The integer R' exists because of the assumption that R and N are coprime. It can be constructed using the extended Euclidean algorithm. 0<R'<N, 0<N'<R, and RR' - NN' = 1
* A straightforward algorithm to multiply numbers in Montgomery form is to multiply aR mod N, bR mod N, and R' as integers and reduce modulo N.
* EX: to multiply 7 and 15 modulo 17 in Montgomery form, again with R = 100, compute the product of 3*4=12. The extended Euclidean algorithm implies that 8 *100 - 47 *17=1 so R'=8. Multiply 12 by 8 to get 96 and reduce modulo 17 to get 11, resulting in a Montgomery form of 3, as expected.
 ## The REDC Algortihm
 * The above algorithm is slower than multiplication in the standard representation because of the need to multiply by R' and divide by N.
 * Montgomery reduction, or REDC, is an algorihm that simultaneously computes the product by R' and reduces modulo N more quickly than the Naive method.
 * While conventional modular reduction focuses on making a number more divisible by R. It does this by adding a small multiple of N which is sophisticatedly chosen to cancel the residue modulo R. Dividing the result by R yields a much smaller number (nearly the reduction modulo N, and computing the reduction modulo N), and all that's required is a final conditional subtraction. All computations are done using only reduction and divisions with respect to R, not N, so the algorithm runs faster than a straighforward modular reduction by division.
 * The REDC algorithm is seen below....
```
function REDC is
    input: Integers R and N with gcd(R, N) = 1,
           Integer N′ in [0, R − 1] such that NN′ ≡ −1 mod R,
           Integer T in the range [0, RN − 1].
    output: Integer S in the range [0, N − 1] such that S ≡ TR−1 mod N

    m ← ((T mod R)N′) mod R
    t ← (T + mN) / R
    if t ≥ N then
        return t − N
    else
        return t
    end if
end function
```
* To see that this algorithm is correct, first observe that m is chosen so that T + mN is divisible by R, A number is divisible by R if and only if it is congruent to zero mod R, so we have....
 *  T + mN = T + ((T mod R)N')mod R)N = T + TN'N=T-T=0(mod R)
* Therefore, t is an integer. Second, the output is either t or t-N, both of which are congruent to t Mod N, therefore to prove that the output is congruent to TR^-1modN, if suffices to prove that t is TR^-1 mod N, t satisfies:
 * t=(T+mN)R^-1=TR^-1 + (mR^-1)N = TR^-1 (mod N)
* Therefore, the output has the correct residue class.
* m is in [0, R-1] and therefore T + mN is between 0 and (RN-1) + (R-1)N<2RN. t is less than 2N, and because it's an integer, this puts t in the range [0, 2N-1]. Therefore, reducing t into the desired range requires at most a single subtraction, so the algorithm's output lies in the correct range.
* To use REDC to compute the product of 7 and 15 modulo 17, first convert to Montgomery form and multiply as integers to get 12 as above. Then apply REDC with R=100, N=17, N'=47, and T=12.
* The first step sets m to 12 * 47 mod 100=64. The second step sets t to (12+64 * 17) /100. Notice that 12 + 64 * 17 is 1100, a multiple of 100 as expected. t is set to 11, which is less than 17, so the final result is 11.    
* Source: https://en.wikipedia.org/wiki/Montgomery_modular_multiplication

# Polynomial Interpolation
* Polynomial interpolation is a method of finding a polynomial function that fits a set of data points exactly. There are several methods for finding it, but the polynomial itself is unique.
## Lagrange Polynomial
* Begin by constructing a polynomial that goes through 2 data points (x0, y0) and (x1, y1), and combine two equations
 * y-y1=m(x-x1) and m=(y1-y0)/(x1-x0) becomes....
 * y=[(y1-y0)/(x1-x0)] * (x-x1) + y1.
* First swap y1-y0 and x-x1
 * y=[(x-x1)/(x1-x0)]*(y1-y0)+y1
* Distribute the fraction
 *  y=(x-x1)/(x1-x0)*y1 - (x-x1)/(x1-x0)*y0 + y1
* Multiplying the right-most y1 term by (x1-x0)/(x1-x0)=1
 * y=(x-x1)/(x1-x0)*y1 - (x-x1)/(x1-x0) * y0 + (x1-x0)/(x1-x0) * y1
* Combining to the y1 terms:
 * y= -[(x-x1)/(x1-x0)*y0] + (x-x0)/(x1-x0)*y1
*  And flipping the denominator of the first term to get rid of the negative
 * y = (x-x1)/(x0-x1)*y0 + (x-x0)/(x1-x0)*y1
* When x=x1, the first term cancels out with a zero on top and the second term ends up as 1 * y1 = y1.
* If x=x0, then the first term ends up as 1 * y0=y0 and the second term cancels out with a zero on top, causing the entire expression to be y0.
* This would be written in Python like this
```
import numpy as np
import matplotlib.pyplot as plot 

#Data goes through the points (1,3) and (5,7)
x=[1,5]
y=[3,7]

#Set the number of data points
pts=len(x)-1
prange=np.linspace(x[0],x[pts],50)
 
plot.plot(x,y,marker='o', color='r', ls='',markersize=10)
def f(o):
    z=((o-x[1])/(x[0]-x[1]))*y[0] + ((o-x[0])/(x[1]-x[0]))*y[1]
    return z

plot.plot(prange,f(prange))
19
plot.show()
```
* While Lagrange polynomials are among the easiest methods to understand intuitively and are efficient for calculating a specific y(x), they fail when attempting to either find an explicit formula y=a0+a1x+...+anX^n, or when adding data points after the initial interpolation is performed. For incremental interpolation, we would need to completely re-perform the entire evaluation.

## Newton Interpolation
* Newton interpolation, unlike Lagrange polynomial, allows for incremental interpolation and provides an efficient way to find an explicit formula y=a0 + a1x +...+ anX^n.
* Newton interpolation is all about finding coefficients and then using those coefficients to calculate subsequent coefficients.
* With one data point (x0, y0), the calculation is simple:
 * b0 = y0
* And the polynomial is y=b0.
* If you add a new data point (x1, y1), the next coefficient b1 is denoted by [y0, y1]
 * b1 = [y0, y1] = (y1-y0)/(x1-x0)
* The polynomial is y= b0 + b1(x-x0)
* With a third data point, coefficient b2=[y0, y1, y2]
 * b2=[y0, y1, y2] = ([y1, y2] - [y0,y1])/(x2-x0) = [y2-y1/x2-x1 - y1-y0/x1-x0]/x2-x0
* The resulting polynomial is...
 * y = b0 + b1(x-x0) + b2(x-x0)(x-x1)
* Source: https://math.libretexts.org/Courses/Angelo_State_University/Mathematical_Computing_with_Python/3%3A_Interpolation_and_Curve_Fitting/3.2%3A_Polynomial_Interpolation
