# Skill

## Preconditions
- The task workspace and existing project are accessible
- Initial project orientation is complete
- Relevant execution boundaries are identified
- Behavioral contract is defined
- Root-cause hypothesis is supported by the code path
- A supported repair design exists
- Updated source artifacts are available
- Focused validation has completed

## Procedure

1. Inspect the supplied evidence and map the relevant request, job, authentication, or processing path without replacing the project structure.
2. Write down the required success, failure, validation, status, logging, persistence, and concurrency semantics that must remain externally observable.
3. Follow the failing value, exception, token, promise, cache entry, or shared state from input through transformation and output, locating the first incorrect interpretation or unsafe boundary.
4. Choose the smallest targeted change that restores correct semantics, preserves valid behavior, and explicitly handles malformed, failed, repeated, or concurrent execution where applicable.
5. Edit the existing implementation in place, keeping public interfaces and unrelated behavior stable while adding the required synchronization, validation, failure propagation, or response handling.
6. Run focused checks for normal requests or jobs, malformed inputs, upstream failures, repeated or overlapping work, status and response shape, diagnostic behavior, and output persistence as relevant to the task.
7. Inspect the final diff and required artifact locations, remove accidental debugging or unrelated edits, and confirm the deliverable is usable by the existing verifier and runtime.

## Failure checks

- Relevant entrypoints, dependencies, and diagnostic evidence are identified.
- The contract covers both normal behavior and explicitly required failure behavior.
- The hypothesis explains the observed failure and is tied to specific existing code paths.
- The design does not rely on hiding failures, broad serialization, weakening validation, or replacing the existing application.
- The edits are confined to the task workspace and compile or parse according to the project language.
- Valid cases preserve the contract and invalid, failed, or concurrent cases produce explicit safe outcomes without accidental success.
- Required files remain in place, the implementation is not a stub, and all stated acceptance conditions are addressed.

## Stop condition

Stop when the required outputs exist and all validators pass: Relevant files and execution boundaries, Initial list of observed symptoms and constraints, Behavioral contract, Preservation invariants, Adversarial input matrix, Root-cause hypothesis, Affected state transitions, Candidate repair locations, Patch design, Error and state-handling rules, Validation plan, Updated source artifacts, Observed validation results, Remaining defects, if any, Final corrected project artifacts, Completion assessment.
