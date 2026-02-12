import pandas as pd
from typing import Dict, Tuple


def parse_screaming_frog_csv(file) -> Tuple[pd.DataFrame, str]:
    """
    Parse Screaming Frog CSV export file.

    Args:
        file: Uploaded file object from Streamlit

    Returns:
        Tuple of (DataFrame, mode) where mode is either 'embeddings' or 'text'
    """
    try:
        # Get file extension
        file_name = file.name.lower()

        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(file, engine='openpyxl')
        elif file_name.endswith('.csv'):
            # Try reading CSV with multiple fallback strategies
            try:
                # First attempt: Auto-detect delimiter with UTF-8
                df = pd.read_csv(
                    file,
                    encoding='utf-8',
                    sep=None,  # Auto-detect delimiter
                    quotechar='"',
                    skipinitialspace=True,
                    on_bad_lines='skip',
                    engine='python'
                )
            except Exception as e1:
                # Second attempt: Try semicolon (European exports)
                file.seek(0)
                try:
                    df = pd.read_csv(
                        file,
                        sep=';',
                        encoding='utf-8',
                        quotechar='"',
                        skipinitialspace=True,
                        on_bad_lines='skip',
                        engine='python'
                    )
                except Exception as e2:
                    # Third attempt: Latin-1 encoding with auto-detect
                    file.seek(0)
                    try:
                        df = pd.read_csv(
                            file,
                            sep=None,
                            encoding='latin-1',
                            quotechar='"',
                            skipinitialspace=True,
                            on_bad_lines='skip',
                            engine='python'
                        )
                    except Exception as e3:
                        # Fourth attempt: Tab-separated
                        file.seek(0)
                        try:
                            df = pd.read_csv(
                                file,
                                sep='\t',
                                encoding='utf-8',
                                on_bad_lines='skip',
                                engine='python'
                            )
                        except Exception as e4:
                            raise ValueError(
                                f"Could not parse CSV file. "
                                f"Tried multiple formats (comma, semicolon, tab). "
                                f"Original error: {str(e1)}"
                            )
        else:
            raise ValueError(
                "Unsupported file format. Please upload .xlsx, .xls, or .csv files."
            )

        # Clean up the dataframe
        # 1. Strip BOM (Byte Order Mark) from column names
        df.columns = df.columns.str.replace('^\ufeff', '', regex=True)

        # 2. Drop completely empty columns
        df = df.dropna(axis=1, how='all')

        # 3. Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # 4. Detect mode: embeddings vs text
        mode = detect_mode(df)

        # 5. Process based on mode
        if mode == 'embeddings':
            df = process_embeddings_mode(df)
        else:
            df = process_text_mode(df)

        return df, mode

    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")


def detect_mode(df: pd.DataFrame) -> str:
    """
    Detect if the CSV contains pre-generated embeddings or text content.

    Returns:
        'embeddings' if embedding columns found, 'text' otherwise
    """
    columns_lower = [col.lower() for col in df.columns]

    # Check for embedding columns (e.g., "Embedding 1", "Embedding_1", etc.)
    embedding_patterns = ['embedding', 'embed', 'vector']

    for pattern in embedding_patterns:
        embedding_cols = [col for col in columns_lower if pattern in col]
        if len(embedding_cols) >= 5:  # At least 5 embedding dimensions
            return 'embeddings'

    return 'text'


def process_embeddings_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process DataFrame in embeddings mode.
    Extract URL and embedding columns.
    """
    # Find URL column
    url_col = detect_url_column(df)

    if url_col is None:
        raise ValueError(
            "Could not find URL column. "
            "Expected columns like 'Address', 'URL', 'Page', etc."
        )

    # Find embedding columns (all columns with 'embedding' in name)
    columns_lower = {col.lower(): col for col in df.columns}
    embedding_cols = [
        columns_lower[col]
        for col in columns_lower
        if 'embedding' in col or 'embed' in col or 'vector' in col
    ]

    # Create standardized DataFrame
    result_df = pd.DataFrame()
    result_df['url'] = df[url_col]

    # Add embedding columns (will be converted to tensor later)
    for i, col in enumerate(sorted(embedding_cols)):
        result_df[f'embedding_{i}'] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing URLs or NaN embeddings
    result_df = result_df.dropna()

    return result_df


def process_text_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process DataFrame in text mode.
    Extract URL and text content columns, combine into single text field.
    """
    # Find URL column
    url_col = detect_url_column(df)

    if url_col is None:
        raise ValueError(
            "Could not find URL column. "
            "Expected columns like 'Address', 'URL', 'Page', etc."
        )

    # Detect text content columns
    text_cols = detect_text_columns(df)

    # Create standardized DataFrame
    result_df = pd.DataFrame()
    result_df['url'] = df[url_col]

    # Extract individual text fields for inspection
    for col_name, original_col in text_cols.items():
        if original_col:
            result_df[col_name] = df[original_col].fillna('')
        else:
            result_df[col_name] = ''

    # Combine text fields into single field for embedding
    result_df['combined_text'] = (
        result_df.get('title', '') + ' ' +
        result_df.get('h1', '') + ' ' +
        result_df.get('meta_description', '') + ' ' +
        result_df.get('body_text', '')
    ).str.strip()

    # Drop rows with missing URLs or empty text
    result_df = result_df[
        (result_df['url'].notna()) &
        (result_df['combined_text'].str.len() > 0)
    ]

    return result_df


def detect_url_column(df: pd.DataFrame) -> str:
    """Detect URL column from Screaming Frog export."""
    columns_lower = {col.lower(): col for col in df.columns}

    # Priority order for URL columns
    url_patterns = ['address', 'url', 'page', 'link', 'website', 'uri']

    for pattern in url_patterns:
        for col_lower, col_original in columns_lower.items():
            if pattern in col_lower:
                return col_original

    return None


def detect_text_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect text content columns from Screaming Frog export.

    Returns:
        Dict mapping standard names to original column names
    """
    columns_lower = {col.lower(): col for col in df.columns}

    detected = {}

    # Title patterns
    title_patterns = ['title 1', 'title', 'page title', 'meta title']
    detected['title'] = find_best_match_col(columns_lower, title_patterns)

    # H1 patterns
    h1_patterns = ['h1-1', 'h1 1', 'h1', 'heading 1', 'main heading']
    detected['h1'] = find_best_match_col(columns_lower, h1_patterns)

    # Meta description patterns
    meta_patterns = [
        'meta description 1',
        'meta description',
        'description',
        'meta_description'
    ]
    detected['meta_description'] = find_best_match_col(columns_lower, meta_patterns)

    # Body text patterns (if available)
    body_patterns = ['body', 'content', 'text', 'page content', 'body text']
    detected['body_text'] = find_best_match_col(columns_lower, body_patterns)

    return detected


def find_best_match_col(columns_dict: Dict[str, str], patterns: list) -> str:
    """
    Find best matching column from patterns.

    Args:
        columns_dict: Dict mapping lowercase column names to original names
        patterns: List of patterns to match (in priority order)

    Returns:
        Original column name or None
    """
    for pattern in patterns:
        # Exact match first
        if pattern in columns_dict:
            return columns_dict[pattern]

    # Partial match
    for pattern in patterns:
        for col_lower, col_original in columns_dict.items():
            if pattern in col_lower:
                return col_original

    return None


def validate_dataframe(df: pd.DataFrame, mode: str) -> Tuple[bool, str]:
    """
    Validate that DataFrame has required columns.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(df) == 0:
        return False, "DataFrame is empty after processing"

    if 'url' not in df.columns:
        return False, "No URL column found"

    if mode == 'embeddings':
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        if len(embedding_cols) < 5:
            return False, f"Not enough embedding columns found (got {len(embedding_cols)}, need at least 5)"

    elif mode == 'text':
        if 'combined_text' not in df.columns:
            return False, "No combined_text column found"

        if df['combined_text'].str.len().max() == 0:
            return False, "All text content is empty"

    return True, ""
