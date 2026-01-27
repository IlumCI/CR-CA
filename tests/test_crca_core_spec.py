from crca_core import DraftSpec, lock_spec, validate_spec


def test_draft_spec_validates() -> None:
    draft = DraftSpec()
    report = validate_spec(draft)
    assert report.ok is True
    assert report.errors == []


def test_lock_spec_requires_approvals() -> None:
    draft = DraftSpec()
    try:
        lock_spec(draft, approvals=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_lock_spec_produces_hash() -> None:
    draft = DraftSpec()
    locked = lock_spec(draft, approvals=["human"])
    assert locked.spec_hash
    assert locked.status.value == "locked"

