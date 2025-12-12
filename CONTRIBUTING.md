# Contributing

The roman-corgi community welcomes contributions of code, documentation, and tests. Please see below for guidelines on how to contribute, and please be sure to follow our code of conduct in all your interactions with the project.

## Getting Started

A great place to start is in the issue tracker of any public roman-corgi repository. If you would like to contribute to an issue, leave a comment on it. If you have questions about an issue, post them as a comment.

## Working with the Code

Start by forking the repository (https://docs.github.com/en/get-started/quickstart/fork-a-repo) you wish to contribute to and cloning the fork to your local system. You may choose to work in a new branch in your fork, but all eventual pull requests back to the upstream repository must be to the main branch. While working, be sure to keep your fork up to date with the upstream repository (https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork). If you have write access to a repository via the CPP, make sure that you are always working on a branch other than main.

You are encouraged to use automated linting tools while you are editing or writing new code. An overview of available linting tools can be found here: https://realpython.com/python-code-quality/.

## Coding Conventions

Different repositories may have different coding conventions - refer to repository-specific documentation for details.

## Pull Requests

Code contributions must be made via pull requests to the main branch. Pulls that cannot be automatically merged or that fail any automated tests will not be merged until corrected. Pull requests should be as small as possible, targeting a single issue/new feature. While preparing your pull request, follow this checklist:

- [ ]   Sync your fork (or clone of the repository) and ensure that your pull can be merged automatically by merging main onto the branch you wish to pull from.
- [ ]   Ensure that all of your new additions have properly formatted docstrings
- [ ]    Ensure that all of the commits going in to your pull have informative messages
- [ ]    Lint your code.
- [ ]    In a clean virtual environment install your working copy of the code in developer mode and run all available unit tests (including any new ones you've added).
- [ ]    Create a new pull request. Fully describe all changes/additions

## New Tutorial
Add a new Jupyter notebook in the examples folder. 
In order to  have it appear on the documentation webpage, you must also add a "new_tutorial.nblink" in docs/source. Inside this file, add the lines 
```
{
"path": "../../examples/new_tutorial.ipynb"
}
```
Thank You!
