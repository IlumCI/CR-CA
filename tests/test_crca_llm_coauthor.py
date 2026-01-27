from crca_llm import LLMCoauthor
from crca_core.models.spec import DraftSpec, LockedSpec


def test_llm_coauthor_returns_draft_specs_only() -> None:
    coauthor = LLMCoauthor()
    bundle = coauthor.draft_specs(user_text="Study effect of X on Y", observed_columns=["X", "Y", "Z"])
    assert bundle.drafts
    assert all(isinstance(d, DraftSpec) for d in bundle.drafts)
    # Ensure it never returns a locked spec
    assert not any(isinstance(d, LockedSpec) for d in bundle.drafts)

