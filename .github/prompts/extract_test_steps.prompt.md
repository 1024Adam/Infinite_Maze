---
mode: 'agent'
tools: ['changes', 'codebase', 'editFiles', 'extensions', 'fetch', 'githubRepo', 'problems', 'runCommands', 'search', 'searchResults', 'terminalLastCommand', 'terminalSelection']
description: 'Extract test steps from a Selenium test suite program and convert them into a documented list of steps and expected results.'
---
Your goal is to extract test steps from a Selenium test suite program and convert them into a list of steps and expected results.

Ask for the name of the test suite if it is not clearly provided. Ask questions if the requirements are not clear or if you need more information to proceed.

## Prerequisites

- Follow test extraction guidelines in `.github/roles/test_extractor.md`

## Identify any test prerequisites
- Determine if there are any prerequisites for the test case
- Identify any dependencies on other tests or components
- Document prerequisites clearly at the top of the test case

## Extract test steps
- Review the Selenium test suite code to identify individual test cases
- For each test case, extract the sequence of actions performed (e.g., clicking buttons, entering text)
- Document the expected results for each action
- Organize the extracted steps into a clear and structured format

## Next Steps
- Do not proceed to perform any other actions
- Wait for the user to specifically request to format the extracted test steps into a CSV file