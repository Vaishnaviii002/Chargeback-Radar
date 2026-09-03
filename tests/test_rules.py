from src.rules import evaluate_rules


def make_safe_payment() -> dict:
    return {
        "amount_paise": 100_000,
        "device_is_new": False,
        "ip_country_matches_billing": True,
        "ip_is_proxy_or_vpn": False,
        "cvv_result": "match",
        "threeds_status": "authenticated",
        "threeds_liability_shift": True,
        "txns_last_1h": 0,
        "txns_last_24h": 0,
        "amount_last_24h_paise": 0,
        "account_age_days": 365,
        "prior_disputes_count": 0,
        "descriptor_clarity_score": 0.90,
        "billing_shipping_distance_km": 10,
        "is_duplicate_payment": False,
        "cancelled_subscription_billed": False,
    }


def rule_ids(result: dict) -> set[str]:
    return {
        hit["rule_id"]
        for hit in result["hits"]
    }


def test_safe_payment_has_no_rule_hits():
    result = evaluate_rules(make_safe_payment())

    assert result["triggered"] is False
    assert result["rule_count"] == 0
    assert result["rule_risk_score"] == 0
    assert result["suggested_action"] == "MONITOR"
    assert result["hard_override_action"] is None


def test_duplicate_payment_creates_hard_refund_override():
    payment = make_safe_payment()
    payment["is_duplicate_payment"] = True

    result = evaluate_rules(payment)

    assert "MERCHANT_DUPLICATE_PAYMENT" in rule_ids(result)
    assert result["suggested_action"] == "RECOMMEND_REFUND"
    assert result["hard_override_action"] == "RECOMMEND_REFUND"


def test_cancelled_subscription_creates_hard_override():
    payment = make_safe_payment()
    payment["cancelled_subscription_billed"] = True

    result = evaluate_rules(payment)

    assert (
        "MERCHANT_CANCELLED_SUBSCRIPTION"
        in rule_ids(result)
    )
    assert result["hard_override_action"] == "RECOMMEND_REFUND"


def test_failed_authentication_requires_review():
    payment = make_safe_payment()
    payment["cvv_result"] = "failed"
    payment["threeds_status"] = "failed"
    payment["threeds_liability_shift"] = False

    result = evaluate_rules(payment)

    assert "AUTHENTICATION_GAP" in rule_ids(result)
    assert result["suggested_action"] == "MANUAL_REVIEW"
    assert result["hard_override_action"] is None


def test_new_device_and_proxy_trigger_network_rule():
    payment = make_safe_payment()
    payment["device_is_new"] = True
    payment["ip_is_proxy_or_vpn"] = True

    result = evaluate_rules(payment)

    assert "DEVICE_NETWORK_ANOMALY" in rule_ids(result)
    assert "true_fraud" in result["risk_categories"]


def test_velocity_spike_is_detected():
    payment = make_safe_payment()
    payment["txns_last_1h"] = 4
    payment["txns_last_24h"] = 8

    result = evaluate_rules(payment)

    assert "PAYMENT_VELOCITY_SPIKE" in rule_ids(result)
    assert result["suggested_action"] == "MANUAL_REVIEW"


def test_repeat_disputes_prepare_evidence():
    payment = make_safe_payment()
    payment["prior_disputes_count"] = 3

    result = evaluate_rules(payment)

    assert "REPEAT_DISPUTE_HISTORY" in rule_ids(result)
    assert result["suggested_action"] == "PREPARE_EVIDENCE"


def test_unclear_descriptor_prepares_evidence():
    payment = make_safe_payment()
    payment["descriptor_clarity_score"] = 0.20

    result = evaluate_rules(payment)

    assert "UNCLEAR_BILLING_DESCRIPTOR" in rule_ids(result)
    assert result["suggested_action"] == "PREPARE_EVIDENCE"


def test_new_high_value_account_requires_review():
    payment = make_safe_payment()
    payment["account_age_days"] = 2
    payment["amount_paise"] = 3_000_000

    result = evaluate_rules(payment)

    assert "NEW_ACCOUNT_HIGH_VALUE" in rule_ids(result)
    assert result["suggested_action"] == "MANUAL_REVIEW"


def test_rule_risk_score_is_capped_at_one():
    payment = make_safe_payment()
    payment.update(
        {
            "is_duplicate_payment": True,
            "cancelled_subscription_billed": True,
            "cvv_result": "failed",
            "threeds_status": "failed",
            "threeds_liability_shift": False,
            "device_is_new": True,
            "ip_country_matches_billing": False,
            "ip_is_proxy_or_vpn": True,
            "txns_last_1h": 5,
            "txns_last_24h": 10,
            "prior_disputes_count": 4,
            "descriptor_clarity_score": 0.10,
            "account_age_days": 1,
            "amount_paise": 5_000_000,
            "billing_shipping_distance_km": 2_000,
        }
    )

    result = evaluate_rules(payment)

    assert result["rule_count"] >= 7
    assert result["rule_risk_score"] == 1.0
    assert result["hard_override_action"] == "RECOMMEND_REFUND"


def test_future_outcome_fields_do_not_affect_rules():
    payment = make_safe_payment()
    payment["chargeback_within_120d"] = 1
    payment["chargeback_family"] = "true_fraud"
    payment["dispute_status"] = "lost"

    result = evaluate_rules(payment)

    assert result["triggered"] is False
    assert result["rule_count"] == 0