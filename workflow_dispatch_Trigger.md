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
