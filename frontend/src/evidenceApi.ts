const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export type EvidenceStrength = "STRONG" | "MODERATE" | "WEAK";

export type CaseStrength =
  | "STRONG"
  | "MODERATE"
  | "WEAK"
  | "INSUFFICIENT";

export type EvidenceItem = {
  title: string;
  statement: string;
  citation_fact_ids: string[];
  strength: EvidenceStrength;
  relevance: string;
};

export type MissingEvidenceItem = {
  item: string;
  reason: string;
  expected_source: string;
  priority: "REQUIRED" | "RECOMMENDED" | "OPTIONAL";
};

export type EvidencePack = {
  schema_version: "1.0";
  payment_id: string;
  case_summary: string;
  case_strength: CaseStrength;

  recommended_action:
    | "MONITOR"
    | "PREPARE_EVIDENCE"
    | "MANUAL_REVIEW"
    | "RECOMMEND_REFUND";

  action_rationale: string;
  evidence_items: EvidenceItem[];
  missing_evidence: MissingEvidenceItem[];

  customer_message: {
    subject: string;
    body: string;
    tone: "NEUTRAL" | "EMPATHETIC" | "URGENT";
    send_status: "DRAFT_ONLY";
  } | null;

  limitations: string[];
  human_approval_required: true;
  action_executed: false;
};

export type GuardrailReport = {
  passed: boolean;
  checks_run: number;
  facts_checked: number;
  evidence_items_checked: number;
  citations_checked: number;

  violations: Array<{
    code: string;
    field_path: string;
    message: string;
  }>;
};

export type EvidenceDeliveryResult = {
  evidence_pack: EvidencePack;

  delivery_mode:
    | "LIVE_OPENAI"
    | "CACHE"
    | "DETERMINISTIC_FALLBACK";

  provider: string;
  model: string | null;
  response_id: string | null;
  latency_ms: number;
  cache_hit: boolean;
  fallback_used: boolean;
  fallback_reason: string | null;
  guardrails: GuardrailReport;
};

export type EvidenceSystemStatus = {
  status: string;
  ai_enabled: boolean;
  api_key_configured: boolean;
  model: string;
  deterministic_fallback_available: boolean;
  cache_enabled: boolean;
  audit_integrity: "SHA256_CHAIN" | "HMAC_SHA256_CHAIN";
  automatic_action_execution: boolean;
};

export type AuditVerification = {
  valid: true;
  records_verified: number;
  last_event_hash: string;
  integrity_mode: "SHA256_CHAIN" | "HMAC_SHA256_CHAIN";
};

async function requestEvidenceJson<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const payload = (await response.json()) as {
        detail?: string;
      };

      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Preserve the safe HTTP status fallback.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export function fetchEvidenceStatus(): Promise<EvidenceSystemStatus> {
  return requestEvidenceJson<EvidenceSystemStatus>(
    "/api/evidence/status",
  );
}

export function generateTransactionEvidence(
  paymentId: string,
  refresh = false,
): Promise<EvidenceDeliveryResult> {
  return requestEvidenceJson<EvidenceDeliveryResult>(
    `/api/evidence/${encodeURIComponent(paymentId)}/generate`,
    {
      method: "POST",
      body: JSON.stringify({ refresh }),
    },
  );
}

export function verifyEvidenceAudit(): Promise<AuditVerification> {
  return requestEvidenceJson<AuditVerification>(
    "/api/evidence/audit/verify",
  );
}