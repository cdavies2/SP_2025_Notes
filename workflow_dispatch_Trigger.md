# workflow_dispatch
* A workflow trigger is an event that causes a workflow to run.
* workdlow_dispatch only triggers a workflow run if the workflow file exists on the default branch.
* To enable a workflow to be triggered manually, you must configure the workflow_dispatch event. This can be manually done via the GitHub API, CLI, or UI
```
on: workflow_dispatch
```
## Providing inputs
* You can configure custom-defined input properties, default input values, and required inputs for the event directly in your workflow.
* When you trigger the event, you can provide the ref and any inputs. When the workflow runs, you can access the input values in the `inputs` context.
* The workflow will also receive the inputs in the `github.event.inputs` context. The information in the `inputs` context and `github.event.inputs` context is identical, except that the `inputs` context preserves Booleans instead of converting them to Strings. The `choice` type resolves to a String and is a single selectable option.
* The maximum possible top-level properties for `inputs` is 10
* The maximum payload for `inputs` is 65,535 characters.
* In the example below, you pass values for inputs (`logLevel`, `inputs.tags`, and `inputs.environment`) to the workflow when you run it, and the workflow then prints the values to the log, using the `inputs.logLevel`, `inputs.tags`, and `inputs.environment` context properties.
```
on:
  workflow_dispatch:
    inputs:
      logLevel:
        description: 'Log level'
        required: true
        default: 'warning'
        type: choice
        options:
        - info
        - warning
        - debug
      tags:
        description: 'Test scenario tags'
        required: false
        type: boolean
      environment:
        description: 'Environment to run tests against'
        type: environment
        required: true

jobs:
  log-the-inputs:
    runs-on: ubuntu-latest
    steps:
      - run: |
          echo "Log level: $LEVEL"
          echo "Tags: $TAGS"
          echo "Environment: $ENVIRONMENT"
        env:
          LEVEL: ${{ inputs.logLevel }}
          TAGS: ${{ inputs.tags }}
          ENVIRONMENT: ${{ inputs.environment }}
```
* If this was run from a browser, values for the required inputs must be manually entered before the workflow will run.
* You can also pass inputs when you run a workflow from a script
* Source: https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#workflow_dispatch

## inputs Context
* The `inputs` context contains input properties passed to an action, to a reusable workflow, or to a manually triggered workflow. For reusable workflows, the input names and types are defined in the `workflow_call` event configuration of a reusable workflow, and the input values are passed from `jobs.<job_id>.with` in an external workflow that calls the reusable workflow. For manually triggered workflows, the inputs are defined in the `workflow_dispatch` event configuration of a workflow.
* The properties in the `inputs` context are defined in the workflow file. They are only available in a reusable workflow or in a workflow triggered by the `workflow dispatch` event.
* The `inputs` object is only available in a reusable workfloe, or in a workflow triggered by the `workflow_dispatch` event. This contect can be accessed from any job or step in a workflow.
* `inputs.<name>` will contain a string, number, boolean, or choice. These are the input values passed from an external workflow.

## Example contents of the Inputs Context
* Below shows example contents of the `inputs` context in a workflow that defined the `build_id`, `deploy_target`, and `perform_deploy` inputs.
```
{
  "build_id": 123456768,
  "deploy_target": "deployment_sys_1a",
  "perform_deploy": true
}
```

## Example usage of the inputs context in a reusable workflow
* This example reusable workflow uses the inputs context to get the values of the build_id, deploy_target, and perform_deploy inputs that were passed to the reusable workflow from the caller workflow.
```
name: Reusable deploy workflow
on:
  workflow_call:
    inputs:
      build_id:
        required: true
        type: number
      deploy_target:
        required: true
        type: string
      perform_deploy:
        required: true
        type: boolean

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ inputs.perform_deploy }}
    steps:
      - name: Deploy build to target
        run: echo "Deploying build:${{ inputs.build_id }} to target:${{ inputs.deploy_target }}"
```
## Example usage of the inputs context in a manually triggered workflow
* This example workflow triggered by a workflow_dispatch event uses the inputs context to get the values of the build_id, deploy_target, and perform_deploy inputs that were passed to the workflow
```
on:
  workflow_dispatch:
    inputs:
      build_id:
        required: true
        type: string
      deploy_target:
        required: true
        type: string
      perform_deploy:
        required: true
        type: boolean

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ inputs.perform_deploy }}
    steps:
      - name: Deploy build to target
        run: echo "Deploying build:${{ inputs.build_id }} to target:${{ inputs.deploy_target }}"
```
* Source: https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/accessing-contextual-information-about-workflow-runs#inputs-context

# GitHub Secrets
* A secret is a variable created in an organization, repository, or repository environment. They can be read by GitHub Actions workflows, but only if you explicity include them.
* For secrets stored at the organization-level, access policies can be used to control which repositories can use them. Organization-level secrets can be shared between several repositories.
* For secrets stored at the environment level, reviewers can control how they are accessed.
* Secret names are bound by the following rules....
  * Names can only contain alphanumeric characters or underscores, no spaces.
  * Names cannot start with the GITHUB_ prefix, or a number.
  * Names are case insensitive
  * Names must be unique at the level they are created at (EX: at the environment level they must be unique to that environment, and the same is true at the repository and organization levels)
  * If a secret with the same name exists at multiple levels, the one at the lowest level takes precedence (EX: if an organization, repository, and environment all have an identically-named secret, the environment-level one takes precedence).
* Avoid using structured data as secret values (EX: avoid JSON or encoded Git blobs). Consider manipulating the structured data (EX: encoding them into a String) before storing them as secrets and decoding them before they are used.

## Accessing Secrets
* To make a secret available to an action, it must be set as an iput or environment variable in the workflow file. Review the action's README file to determine which inputs and environment variables are expected by the action.
* Organization and repository secrets are read when a workflow run is queued, and environment secrets are read when a job referencing the environment starts.

### Limiting Credential Permissions
* When generating credentials, grant the minimum permissions possible (EX: a service account or deploy key instead of personal credentials), use only read-only permissions if that is all that is needed.
* When generating personal access tokens, select minimum scope (permissions and repository access) required, and instead of using a personal access token, consider a GitHub app, which uses fine-grained permissions that are not tied to a user, so the workflow will work even if the installing user leaves the organization.

## Creating Secrets for a Repository
* You need to be the repository owner, or have admin access for an organization's repository.
  1. On GitHub, navigate to the main page of the repository
  2. Under the repository name, click Settings (from the tab or dropdown menu)
  3. In the "Security" section of the sidebar, select "Secrets and Variables", then click "Actions"
  4. Click the "Secrets" tab
  5. Click "New Repository Secret"
  6. In the "Name" field, type a name for your secret
  7. In the "Secret" field, enter the value for your secret
  8. Click "Add Secret"
* If your repository has environment secrets or can access secrets from the parent organization, then said secrets are also listed on this page.

## Creating Secrets for an Environment
* You must own the repository or have admin access.
  1. On GitHub, navigate to the main page of the repository
  2. Under the repository name, click "Settings" (from tab or dropdown)
  3. In the left sidebar, click "Environments"
  4. Click the environment you want to add a secret to
  5. Under "Environment secrets", click "Add secret"
  6. Type a name for your secret in the "Name" input box
  7. Enter the value for your secret
  8. Click "Add secret"
## Using Secrets in a Workflow
* Secrets are not passed to the runner when a workflow is triggered from a forked repository, and they are not automatically passed to reusable workflows.
* To provide an action with a secret as an input or environment variable, you can use the `secrets` context to access secrets you've created in your repository.
* It's recommended to set secrets as job-level environment variables and reference the environment variables to conditionally run steps in the job.
* Example code below
```
steps:
  - name: Hello world action
    with: # Set the secret as an input
      super_secret: ${{ secrets.SuperSecret }}
    env: # Or as an environment variable
      super_secret: ${{ secrets.SuperSecret }}
```
* Source: https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions#using-secrets-in-a-workflow
