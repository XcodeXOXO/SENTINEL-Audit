# 🛡️ Project Sentinel Audit Report

**Contract Assessed**: `{{ contract_name }}`
**Date**: `{{ date }}`
**Confidence Score**: `{{ confidence_score }}/100`

---

## Executive Summary
This report was generated autonomously by Project Sentinel's Three-Pillar AI Architecture. Findings have been semantically verified against the source code structure by the Critic agent.

## Findings

{% for finding in findings %}
### 🚨 {{ finding.title }}
* **Severity**: {{ finding.severity }}
* **Location**: `{{ finding.function_name }}` (Line: `{{ finding.line_number|default('N/A') }}`)

**Description**:
{{ finding.description }}

**Economic / State Invariant violated**:
{{ finding.invariant_violated }}

**Remediation**:
{{ finding.remediation }}

**Sanity Check Context (Critic)**:
{{ finding.critic_verification_notes }}
---
{% endfor %}

## Compilation Check
* Base Compilation successful?: `{{ compilation_successful }}`
* Warnings from `solc`:
```text
{{ solc_warnings|default('None') }}
```
