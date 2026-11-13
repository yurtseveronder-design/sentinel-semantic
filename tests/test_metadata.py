from sentinel_semantic.metadata import MetadataExtractor

extractor = MetadataExtractor()


def test_extracts_department_year_and_access():
    f = extractor.extract("Show confidential finance reports from 2024")
    assert (f.department, f.access_level, f.year) == ("finance", "confidential", 2024)
    assert f.cleaned_query == "Show reports from 2024"


def test_defaults_when_nothing_found():
    f = extractor.extract("hello there")
    assert (f.department, f.year, f.access_level) == (None, None, "internal")
    assert f.cleaned_query == "hello there"


def test_vector_db_filter_only_includes_present_fields():
    f = extractor.extract("hr policy 2023")
    assert extractor.to_vector_db_filter(f) == {"department": "hr", "access_level": "internal", "year": 2023}
