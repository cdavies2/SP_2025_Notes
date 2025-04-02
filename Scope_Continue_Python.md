# Scope in Python
* Scope determines how variables and their names are looked up in code. It defines the area of a program where you can access that name. There are two general scopes..
  1. Global scope: names defined globally are available to all your code
  2. Local scope: names defined locally are only available to code within the scope
* In Python, variables come into existence when you first assign them a value. This can be done via assignments (x = value), import operations (import module, from module import name), function definitions, function argument definitions, or class definitions
* If you assign a value to a name inside a function, the name will have a local scope, but if you assign a value outside of all functions (like at the top of a module) then that name will have a global Python scope.
## Scope vs Namespace
* Python scopes are implemented as dictionaries (known as _namespaces_) that map names to objects.
* Names at the top level of a module are stored in the module's .__ dict __ attribute
* After importing sys, you can use .keys() to inspect the keys of sys.__ dict __, returning a list with all the names defined at the top level of the module.
## Using the LEGB Rule for Python Scope
* LEGB stands for....
  * _Local (or function) scope_: code block of any Python function or lambda expression. It contains the names defined inside a function. These are screated at function call, not function definition, so you will have as many local scopes as function calls.
  * _Enclosing (or nonlocal) scope_: only exists for nested functions. If the local scope is an inner or nested function, the enclosing scope is the scope of the outer or enclosing function. Names in the enclosing scope are visible from the code of the inner and enclosing functions.
  * _Global (or module) scope_: top-most scope, contains all the names you define at the top level of a program or module, visible from everywhere in code.
  * _Built-in scope_: special scope, created/loaded when a script or interactive session is run. It contains names like keywords, functions, exceptions, and other attributes that are built into Python. Available everywhere in code, automatically loaded by Pyhton when you run a program or script.
* When using a nested function, names are resolved by first checking the local scope or innermost function's local scope. Python then looks at all enclosing scopes of outer functions from innermost to outermost, and if no match is found, Python then looks at the global and built-in scopes.
## Functions: The Local Scope
* Every time you call a function, a new local scope comes into existence.
* Parameters and names assigned inside of a function exist only within said function. When the function returns, the local scope is destroyed and names are forgotten.
* Since you can't access local names from statements outside the function, different functions can define objects with the same name (EX: square() and cube() functions can both have "result" and "base" parameters, as they cannot see each other's variable names, they both have local scope).
* The names and parameters of a function can be examined using `.__code__`
```
>>> square.__code__.co_varnames
('base', 'result')
>>> square.__code__.co_argcount
1
>>> square.__code__.co_consts
(None, 2, 'The square of ', ' is: ')
>>> square.__code__.co_name
'square'
```
## Nested Functions: The Enclosing Scope
* Enclosing or nonlocal scope is observed when nesting functions inside of other functions. It takes the form of the local scope of any enclosing function's local scopes. Names that you define in the enclosing Python scope are commonly called nonlocal names, as seen in the code below
```
>>> def outer_func():
...     # This block is the Local scope of outer_func()
...     var = 100  # A nonlocal var
...     # It's also the enclosing scope of inner_func()
...     def inner_func():
...         # This block is the Local scope of inner_func()
...         print(f"Printing var from inner_func(): {var}")
...
...     inner_func()
...     print(f"Printing var from outer_func(): {var}")
...
>>> outer_func()
Printing var from inner_func(): 100
Printing var from outer_func(): 100
>>> inner_func()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'inner_func' is not defined
```
* When you call `outer_func()`, you create a local scope. The local scope of `outer_func()` is the enclosing scope of `inner_func`. Inside `inner_func()`, the scope is neither global nor local, it is a special scope between them known as _enclosing scope_.
* All names created in the enclosing scope are visible from `inner_func()`, except those created after calling `inner_func()`. This is shown below
```
>>> def outer_func():
...     var = 100
...     def inner_func():
...         print(f"Printing var from inner_func(): {var}")
...         print(f"Printing another_var from inner_func(): {another_var}")
...
...     inner_func()
...     another_var = 200  # This is defined after calling inner_func()
...     print(f"Printing var from outer_func(): {var}")
...
>>> outer_func()
Printing var from inner_func(): 100
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    outer_func()
  File "<stdin>", line 7, in outer_func
    inner_func()
  File "<stdin>", line 5, in inner_func
    print(f"Printing another_var from inner_func(): {another_var}")
NameError: free variable 'another_var' referenced before assignment in enclosing
 scope
```
* When `outer_func()` is called, the code runs down to the point in which you call `inner_func()`. The last statement of `inner_func()` tries to access `another_var`. Because `another_var` isn't defined at that point, Python raises a `NameError` because it can't find the name you're trying to use.
* Source: https://realpython.com/python-scope-legb-rule/

# Break and Continue
* A `break` statement allows you to exit a loop entirely when a specific condition is met, effectively stopping the loop execution
* The `continue` statement lets you skip the rest of the code inside the loop for the current iteration and move onto the next iteration
* The `pass` statement is used when code is synatacically required but you have nothing to execute.
## Break Statement
* Allows you to exit a loop when an external condition is triggered. Put this within the code block under your loop statement, usually after a conditional `if` statement.
```
number = 0

for number in range(10):
    if number == 5:
        break    # break here

    print('Number is ' + str(number))

print('Out of loop')
```
* Within this `for` loop, an `if` statement presents the condition that `if` the variable `number` is equivalent to integer 5, _then_ the loop will break.
* Only numbers 0 through 4 will print
## Continue Statement
* `continue` allows you to skip over part of a loop where an external condition is triggered, but go on to complete the rest of the loop. The current iteration is disrupted, but the program returns to the top of the loop.
* The `continue` statement is under the loop, usually after a conditional `if` statement. An example of this is...
```
number = 0

for number in range(10):
    if number == 5:
        continue    # continue here

    print('Number is ' + str(number))

print('Out of loop')
```
* Here, the code will continue despite the disruption when `number` is evaluated as equivalent to 5. As such, every number except 5 outputs.
* The `continue` statement helps us avoid deeply nested conditional code and optimizes loops by eliminating frequently occurring cases you'd like to reject.
* Source: https://www.digitalocean.com/community/tutorials/how-to-use-break-continue-and-pass-statements-when-working-with-loops-in-python-3

# Def vs Lambda
* `lambda` is usually used to define anonymous functions; functions without names that are often passed to other functions as arguments.
* EX: if you had a timer that calles a function passed to it at a certain time, you may use a lambda for that; `timer.do_later(lambda: print("Time to wakeup!"))`
* If a function is going to be reused more than once, def is better, lambdas should be throwaway functions.
* A `def` statement's body can include arbitrary Python code, but `lambda`'s body must be a single expression which is implicitly returned.
