## `LockedSpec`

This page is generated from the JSON schema (contract-first).

| Field | Type | Notes |
|---|---|---|
| `status` | `#/$defs/SpecStatus` | optional. |
| `spec_hash` | `string` | required. |
| `approvals` | `array` | optional. |
| `locked_at_utc` | `string` | required. |
| `data` | `#/$defs/DataSpec` | required. |
| `graph` | `#/$defs/CausalGraphSpec` | required. |
| `roles` | `#/$defs/RoleSpec` | required. |
| `assumptions` | `#/$defs/AssumptionSpec` | required. |
| `scm` | `unknown` | optional. |
