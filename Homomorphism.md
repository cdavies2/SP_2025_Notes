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
