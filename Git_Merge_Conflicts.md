# Git Merge Conflicts
* Conflicts tend to arise when multiple people change the same lines in a file, or one developer deletes a file while another is modifying it, or you are connected to the same git repo in multiple places, resulting in multiple versions of one file.
* Conflicts affect the developer conducting the merge; Git marks the file as beign conflicting and halts the merging process, leaving the developer to solve the conflict.

## Types of Merge Conflicts
* A merge can become conflicted at the start and during a merge process.
* A merge fails to start when Git sees there are changes in either the working directory or staging area. The issue isn't conflicts with other developers, but conflicts with pending local changes. The local state can be stablizied using `git stash`, `git checkout`, `git commit`, or `git reset`
* A merge failure on start will result in an error message saying the file is not updtodate, cannot merge (Changes in working directory)
* A git failure during a merge indicates a conflict between the current local branch and the branch being merged. Git will attempt to merge the files but will leave things for you to resolve manually in the conflicted files.
* A mid-merge failure results in an error message saying the file would be overwritten by the merge, cannot merge (changes in staging area)

## Creating a Merge Conflict
```
mkdir git-merge-test
cd git-merge-test
git init .
echo "this is some content to mess with" > merge.txt
git add merge.txt
git commit -am"we are commiting the inital content"
[main (root-commit) d48e74c] we are commiting the inital content
1 file changed, 1 insertion(+)
create mode 100644 merge.txt
```
* The above code does the following...
  * Create a new directory named `git-merge-test`, change to that directory, and initialize it as a Git repo
  * Create a new text file `merge.txt` with some content in it
  * Add `merge.txt` to the repo and commit it
* Next, create a new branch to use as the conflicting merge
```
git checkout -b new_branch_to_merge_later
echo "totally different content to merge later" > merge.txt
git commit -am"edited the content of merge.txt to cause a conflict"
[new_branch_to_merge_later 6282319] edited the content of merge.txt to cause a conflict
1 file changed, 1 insertion(+), 1 deletion(-)
```
* The above code...
  * Creates and checks out a new branch named `new_branch_to_merge_later`
  * Overwrite the content in `merge.txt`
  * Commit the new content
* The `new_branch_to_merge_later` branch creates a commit that overrides the content of merge.txt
```
git checkout main
Switched to branch 'main'
echo "content to append" >> merge.txt
git commit -am"appended content to merge.txt"
[main 24fbe3c] appended content to merge.tx
1 file changed, 1 insertion(+)
```
* The above chain of commands checks out the `main` branch, appends content to `merge.txt`, and commits it. This puts our example repo in a state where there's two new commits, each in separate branches.
* If we execute `git merge new_branch_to_merge_later`, a conflict appears and the merge fails.

## How to Identify Merge Conflicts
* `git status` gives further insight on merge conflicts. It will often tell us where we have unmerged paths (which branch), tells us to use `git merge --abort` to abort the merge, and tells us which files were modified on both branches.
* The code below examines a modified file
```
cat merge.txt
<<<<<<< HEAD
this is some content to mess with
content to append
=======
totally different content to merge later
>>>>>>> new_branch_to_merge_later
```
* The `cat` command outputs the contents of the `merge.txt` file.
* Many of these lines are "conflict dividers". The `=====` line is the "center" of the conflict, and all lines between it and the `<<<<<< HEAD` line is content that exists in the current branch main which the `HEAD` ref is pointing to. All content between the center and `>>>>>> new_branch_to_merge_later` is content that is present in our merging branch.

## How to Resolve Merge Conflicts Using the Command Line
* The most direct way to resolve a merge conflict is to edit the conflicted file in a text editor.
* Once the file has been edited use `git add merge.txt` to stage the new merged content. To finalize the merge create a new commit by executing:
`git commit -m "merged and resolved the conflict in merge.txt"`

## Git Commands that Can Help Resolve Merge Conflicts
### General Tools
* `git status` identifies conflicted files during a merge
* `git log --merge`: passing the `--merge` argument to the `git log` command produces a log with a list of commits that conflict between merging branches
* `git diff` helps find differences between states of a repository/files. This is useful in predicting and preventing merge conflicts.
### Tools for When Git Fails to Start a Merge
* `git checkout`: can be used for undoing changes to files, or for changing branches
* `git reset --mixed`: can be used to undo changes to the working directory and staging area.
### Tools for When Git Conflicts Arise During a Merge
* `git merge --abort` exits from the merge process and returns the branch to the state before the merge began
* `git reset`: can be used during a merge conflict to reset conflicted files to a known good state

* Source: https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts
