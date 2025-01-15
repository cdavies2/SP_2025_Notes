# What is Unit Testing?
* Unit testing refers to the process of testing the smallest functional unit of code. It is highly recommended for software developers to write software as small, functional units, then to write a test for each of those units.
* Test code should be run automatically each time you make changes in the software code. Therefore, if a test fails, you can quickly determine where the error is located in your code.
* Unit testing improves both test coverage and quality, and automated unit testing helps ensure that developers can spend more time on the code itself.

# What is a Unit Test?
* A unit test is a block of code that verifies the accuracy of a smaller, isolated block of application code, like a function or method. A unit test is desgined to check that a block of code runs as the developer expects, and said test can only interact with the code through inputs and captured (true or false) output.
* A single block of code can have a series of unit tests called _test cases_.
* When a block of code needs other parts of the system to run properly, you cannot use a unit test with the external data. Unit tests must be able to run in isolation
* If system data like databases, objects, or network communication are required for code to function properly, use data stubs instead of unit tests.

 # Unit Testing Strategies
 * Some basic principles should be followed to ensure coverage of test cases...
   * _Logic checks_: does the system perform the proper calculations and traverse through the code properly given an expected input? Are all paths through the code covered by given inputs?
   * _Boundary Checks_: for given inputs, how does the system respond to typical inputs, edge cases, or invalid inputs? (EX: if an integer input between 1 and 10 is expected, how does it react to 6 (typical), 10 (edge) or 22 (invalid)
   * _Error Handling_: when there are errors in inputs, how does the system respond? Do you prompt the user for more input, or does the software crash?
   * _Object-oriented Checks_: if the state of any existing objects is modified by running the code, is said object updated properly?
# Unit Test Example
```
def mult_ints(x, y):
  return x*y

#corresponding unit tests
def test_mult_positives():
   result=mult_ints(3, 4)
   assert result==12

test_mult_positives() #no error means worked properly

def test_mult_negatives():
   result=mult_ints(-4, -5)
   assert result==20

test_mult_negatives()

def test_mult_pos_neg():
   result=mult_ints(-2, 5)
   assert result==-10

test_mult_pos_neg()
```

# What are the Benefits of Unit Testing?
1. _Efficient bug discovery_:input, output, and logic-based errors in code can be caught earlier on in production, thus reducing debugging later in the process. When code changes, run the same unit tests again (alongside integration tests) and expect identical results. If the tests fail (or are "broken") they might be regression-based bugs. Unit testing also helps developers quickly pinpoint what part of the code has an error.
2. _Documentation_:unit tests, when read by other developers, help said developers understand the behaviors the code should display when it runs, and they can use that information to either adjust or refactor the code (making it better composed and perform more cleanly). Rerun the unit tests to make sure the code continues to work after changes.

# How do Developers Use Unit Tests?
1. _Test-driven development_:refers to when developers build tests to check a piece of software's functional requirements before they build the full code itself. Writing tests firsts allows for a code's quality to be checked immediately.
2. _After completing a block of code_:once a block of code is considered "complete", unit tests should be developed if they haven't already, then you can immediately run them and check the results. Unit tests are also run as part of the full suite of other software tests during system testing. They're usually the first set of tests that run during full system software testing.
3. _DevOps efficiency_-one of the core activities in the application of DevOps to software development practices is continuous integration and delivery (CI/CD). Any changes to code are automatically integrated into the wider codebase, run through automated testing, and deployed if the tests pass. Unit tests run automatically in the CI/CD pipeline to ensure code quality as it is upgraded and changed over time.

# When is Unit Testing Less Beneficial?
1. _When time is constrained_:writing new unit tests takes a significant amount of your developers' time. While input and output-based unit tests may be easy to generate, logic-based checks are harder. Once developers begin to write tests, they also notice refactoring opportunities in the block of code and can get distracted from completing the tests, resulting in longer development time and budget concerns.
2. _UI/UX applications_:if the concern is look and feel rather than logic, then other types of testing (like manual testing) may be more effective than unit testing.
3. _Legacy codebases_:writing tests to wrap around existing legacy code can be next to impossible. Unit tests need dummy data (you need sample inputs and to know what the output should be) so it might be too time-consuming to write unit tests for interconnected systems with lots of data parsing.
4. _Rapidly evolving requirements_:depending on the project, software can grow, change directions, or have portions scrapped entirely in any given work sprint. If requirements aren't likely to remain consistent, then there is no real reason to write unit test for each new block of code that's developed.

# What are Unit Testing Best Practices?
1. _Use a unit test framework_: it wastes time to write explicit, fully customized unit tests for every single block of code. There are automated testing frameworks (like pytest and unittest in Python) that do it for us
2. _Automate unit testing_: unit testing should be triggered on different events within software development (EX: you can use them before pushing changes to a branch or before deploying a software update). It can also run on a schedule throughout a development life cycle.
3. _Assert once_: for every unit test, there should be only one true or false outcome (a test should only have one assert statement to reduce confusion, as said statement produces an error with a false outcome and you want to pinpoint where the error was located)
4. _Implement unit testing_: unit testing is an important part of building software, but many projects don't dedicate resources to it. When projects start as prototypes, are smaller, community-based efforts, or simply must be coded quickly, unit testing can be left out due to time constraints. However, when unit testing is a part of your projects from the beginning, the process becomes far easier to follow and repeat.

# What's the difference between unit testing and other types of testing?
* Integration testing ensures that different parts of the software system that are designed to interact do so correctly
* Functional testing checks whether the software system passes the software requirements outlined before it is built
* Performance testing checks whether the software runs to expected performance requirements, like speed and memory size.
* Acceptance testing is when the software is tested manually by stakeholders or user groups to make sure it's working as they anticipatw
* Security testing checks the software against known vulnerabilities and threats. This includes analysis of the threat surface, including third-party software entry points.
* The above testing methods typically require specialized tools and independent processes to check them, and are performed once the basic functions of the application have been developed. However, unit tests run whenever the code builds, can be written immediately as code is written and don't need any special tools.
Source: https://aws.amazon.com/what-is/unit-testing/#:~:text=Unit%20testing%20is%20the%20process,test%20for%20each%20code%20unit.

# Unittest Python
* The unittest framework supports test automation, sharing setup/shutdown code for tests, aggregation of tests into collections, and independence of tests from reporting framework.

## Example code
```
import unittest

class TestStringMethods(unittest.TestCase):

    def test_index(self):
        list1=[0, 1, 2, 3, 4]
        self.assertEqual(0, list1[0]) 
    
    
    def test_prime(self):
        x=5
        isPrime=True
        for i in range(2, x-1):
            if(x%i==0):
                isPrime=False
        self.assertTrue(isPrime==True)
        self.assertFalse(isPrime==False)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
if __name__ == '__main__':
    unittest.main()

    #the number of tests ran is output, "OK" is output if there were no assert errors, all did what it should
```
* assertEqual checks for an expected result, assertTrue() and assertFalse() verify conditions, assertRaises() makes sure a specific exception gets raised. These are used instead of the assert statement to ensure that all test results can be put together into a report.
## TestCase class methods
* assertEqual(a,b) : checks that a==b
* assertNotEqual(a,b): checks a!=b
* assertTrue(x): checks bool(x) is True
* assertFalse(x): checks bool(x) is False
* assertIs(a,b): checks a is b
* assertIsNot(a, b): checks a is not b
* assertIsNone(x): checks x is None
* assertIsNotNone(x): checks x is not None
* assertIn(a, b): checks a in b
* assertNotIn(a,b): checks a not in b
* assertIsInstance(a,b): checks isinstance(a,b) (checks if the member is in the container)
* assertNotIsInstance(a,b): checks not isinstance(a,b)
Source: https://docs.python.org/3/library/unittest.html#assert-methods
