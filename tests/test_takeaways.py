from src.data_loader import PROV_SYNTHETIC, empty_frame, synthetic_demo
from src.takeaways import DEMO, FINDING, STATUS, generate_takeaways
from tests.helpers import make_df


def build(focal_pos, focal_neg, comp_pos, comp_neg):
    rows = []
    rows += [("Travel.pk", f"Agent was helpful and responsive, great support {i}") for i in range(focal_pos)]
    rows += [("Travel.pk", f"Refund still pending, payment problem and no response {i}") for i in range(focal_neg)]
    rows += [("Rival", f"Great price and cheap deals, easy booking {i}") for i in range(comp_pos)]
    rows += [("Rival", f"Hidden charges and rude staff, worst service {i}") for i in range(comp_neg)]
    return make_df(rows)


def test_exactly_three_takeaways_with_data():
    t = generate_takeaways(build(15, 12, 15, 12), "Travel.pk")
    assert len(t) == 3 and [x["number"] for x in t] == [1, 2, 3]
    assert all(x["type"] == FINDING for x in t)
    assert "customer service" in t[0]["headline"].lower()
    assert "/" in t[0]["statistic"] and t[0]["evidence_basis"]


def test_takeaways_reflect_data_not_hardcoded():
    a = generate_takeaways(build(15, 12, 15, 12), "Travel.pk")
    b = generate_takeaways(make_df([("Travel.pk", f"Cheap price and good deals {i}") for i in range(15)]
                                   + [("Travel.pk", f"Rude agent, worst refund delay {i}") for i in range(12)]
                                   + [("Rival", f"Great easy booking {i}") for i in range(12)]), "Travel.pk")
    assert a[0]["headline"] != b[0]["headline"]


def test_insufficient_data_still_three():
    t = generate_takeaways(build(3, 2, 0, 0), "Travel.pk")
    assert len(t) == 3 and all(x["type"] == STATUS for x in t)
    assert all(x["statistic"] == "n/a" for x in t)


def test_no_competitors_gives_status_for_comparison():
    t = generate_takeaways(build(15, 12, 0, 0), "Travel.pk")
    assert len(t) == 3 and t[2]["type"] == STATUS


def test_empty_dataset_and_missing_company():
    assert len(generate_takeaways(empty_frame(), "Travel.pk")) == 3
    assert len(generate_takeaways(None, "")) == 3
    assert all(x["type"] == STATUS for x in generate_takeaways(build(15, 12, 15, 12), "Nobody"))


def test_no_negative_reviews_is_not_called_satisfaction_proof():
    t = generate_takeaways(build(20, 0, 15, 12), "Travel.pk")
    assert t[1]["type"] == STATUS and "does not" in t[1]["explanation"].lower()


def test_gap_takeaway_uses_counts_and_labels_recommendation():
    t = generate_takeaways(build(15, 12, 15, 12), "Travel.pk")
    assert t[2]["type"] == FINDING and t[2]["recommendation"].startswith("Recommendation (not an observed finding)")
    assert "Travel.pk" in t[2]["companies"] and "Rival" in t[2]["companies"]


def test_synthetic_takeaways_are_labelled_demo():
    demo = synthetic_demo()
    t = generate_takeaways(demo, "Demo Agency A (synthetic)", is_synthetic=True)
    assert len(t) == 3 and all(x["type"] == DEMO for x in t)
    assert (demo["provenance"] == PROV_SYNTHETIC).all()
    assert all("SYNTHETIC" in x["limitation"] for x in t)


def test_takeaways_change_with_filtered_data():
    full = build(15, 12, 15, 12)
    only_focal_positive = full[(full["company"] == "Travel.pk") & (full["sentiment"] == "Positive")]
    t = generate_takeaways(only_focal_positive, "Travel.pk")
    assert len(t) == 3 and t[1]["type"] == STATUS
