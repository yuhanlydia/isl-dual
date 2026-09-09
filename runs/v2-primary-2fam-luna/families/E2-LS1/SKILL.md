# Skill

## Preconditions
- Input batch and project contract are available
- n1 completed
- n2 completed
- n3 completed
- n4 completed

## Procedure

1. Read the batch and identify required fields, optional fields, normalization rules, and output/error schemas.
2. Convert each raw record into a validation-ready representation while preserving its original index and payload.
3. Validate every candidate before dispatch, applying type, presence, range, enum, format, uniqueness, timezone, and cross-field rules as applicable; collect all failures for each record.
4. Send only candidates with no validation errors to the external operation, preserving batch order and recording the operation trace.
5. Build the required structured return value with successes, failures or rejections, counts, original indices/payloads where required, and trace information.

## Failure checks

- Records are retained in original order and policy requirements are explicit.
- Every candidate has its original zero-based index and source payload.
- Rejected records contain field/parameter-specific error objects with the required fields, and no candidate is dispatched during validation.
- The number of outbound calls equals the number of valid candidates and invalid candidates cause zero calls.
- Summary counts reconcile with total input, successes, and failures; all rejected records retain required structured diagnostics.

## Stop condition

Stop when the required outputs exist and all validators pass: ordered input records, validation policy, indexed candidate records, valid candidates, structured rejected records, successful results, dispatch trace, contract-compliant batch result.
