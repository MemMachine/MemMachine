def setup_nltk():
    """Checks for and downloads required NLTK data packages."""
    import nltk

    print("Checking for required NLTK data...")
    packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, pkg_id in packages:
        try:
            nltk.data.find(path)
            print(f"✅ - NLTK package '{pkg_id}' is already installed.")
        except LookupError:
            print(f"⚠️ - NLTK package '{pkg_id}' not found. Downloading...")
            nltk.download(pkg_id)
    print("\nNLTK data setup is complete. ✨")
