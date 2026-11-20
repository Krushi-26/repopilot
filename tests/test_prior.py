import pytest

from repopilot.retrieval.prior import path_prior


@pytest.mark.parametrize(
    "path, expected",
    [
        ("src/requests/sessions.py", 1.0),
        ("lib/app/main.go", 1.0),
        ("LICENSE", 0.1),
        ("HISTORY.md", 0.2),
        ("CHANGELOG.rst", 0.2),
        ("tests/test_requests.py", 0.5),
        ("pkg/foo_test.go", 0.5),
        ("web/button.spec.ts", 0.5),
        ("docs/user/advanced.rst", 0.6),
        ("README.md", 0.6),
        ("pyproject.toml", 0.7),
    ],
)
def test_path_prior(path, expected):
    assert path_prior(path) == expected


def test_source_file_named_like_a_doc_word_is_not_demoted():
    # "testing" or "docs" as part of a module name, not a directory, stays source.
    assert path_prior("src/pkg/docstring_utils.py") == 1.0
    assert path_prior("src/pkg/attestation.py") == 1.0
