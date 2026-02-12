"""
Export Handler Module

Handles CSV export functionality for similarity analysis results.
"""

import pandas as pd
from datetime import datetime
from io import BytesIO
from typing import Dict


def format_results_for_export(
    results_df: pd.DataFrame,
    stats: Dict = None
) -> pd.DataFrame:
    """
    Format results DataFrame for export.

    Args:
        results_df: DataFrame with URL pairs and similarity scores
        stats: Optional summary statistics dictionary

    Returns:
        Formatted DataFrame ready for export
    """
    # Create a copy to avoid modifying original
    export_df = results_df.copy()

    # Add metadata columns
    export_df['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Reorder columns for better readability
    column_order = [
        'domain_a_url',
        'domain_b_url',
        'similarity_score',
        'rank',
        'analysis_date'
    ]

    # Only include columns that exist
    column_order = [col for col in column_order if col in export_df.columns]

    export_df = export_df[column_order]

    return export_df


def create_csv_download(results_df: pd.DataFrame) -> bytes:
    """
    Create CSV file as bytes for download.

    Args:
        results_df: DataFrame of URL pairs

    Returns:
        CSV file as bytes
    """
    formatted_df = format_results_for_export(results_df)
    return formatted_df.to_csv(index=False).encode('utf-8')


def create_excel_download(
    results_df: pd.DataFrame,
    stats: Dict,
    domain_a_name: str = "Domain A",
    domain_b_name: str = "Domain B"
) -> bytes:
    """
    Create Excel workbook with multiple sheets.

    Sheets:
    1. All URL Pairs (sorted by Domain A URL, then by rank)
    2. Top Matches Only (rank 1 only)
    3. High Similarity (>80%)
    4. Summary Statistics

    Args:
        results_df: DataFrame of URL pairs
        stats: Summary statistics dictionary
        domain_a_name: Name for Domain A
        domain_b_name: Name for Domain B

    Returns:
        Excel file as bytes
    """
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Sheet 1: All URL Pairs
        df_all = format_results_for_export(results_df)
        df_all_sorted = df_all.sort_values(['domain_a_url', 'rank'])
        df_all_sorted.to_excel(writer, sheet_name="All URL Pairs", index=False)

        # Sheet 2: Top Matches Only (rank 1)
        df_top = df_all[df_all['rank'] == 1].copy()
        df_top_sorted = df_top.sort_values('similarity_score', ascending=False)
        df_top_sorted.to_excel(writer, sheet_name="Top Matches", index=False)

        # Sheet 3: High Similarity (>80%)
        df_high = df_all[df_all['similarity_score'] >= 80].copy()
        df_high_sorted = df_high.sort_values('similarity_score', ascending=False)
        df_high_sorted.to_excel(writer, sheet_name="High Similarity", index=False)

        # Sheet 4: Summary Statistics
        if stats:
            stats_df = create_stats_dataframe(stats, domain_a_name, domain_b_name)
            stats_df.to_excel(writer, sheet_name="Summary Statistics", index=False)

    buffer.seek(0)
    return buffer.getvalue()


def create_stats_dataframe(
    stats: Dict,
    domain_a_name: str = "Domain A",
    domain_b_name: str = "Domain B"
) -> pd.DataFrame:
    """
    Create a formatted DataFrame from summary statistics.

    Args:
        stats: Summary statistics dictionary
        domain_a_name: Name for Domain A
        domain_b_name: Name for Domain B

    Returns:
        Formatted statistics DataFrame
    """
    stats_data = [
        {"Metric": "Analysis Date", "Value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
        {"Metric": "Domain A Name", "Value": domain_a_name},
        {"Metric": "Domain B Name", "Value": domain_b_name},
        {"Metric": "", "Value": ""},  # Blank row
        {"Metric": f"Total URLs from {domain_a_name}", "Value": stats.get('total_domain_a_urls', 0)},
        {"Metric": f"Total URLs from {domain_b_name}", "Value": stats.get('total_domain_b_urls', 0)},
        {"Metric": "Total URL Pairs Analyzed", "Value": stats.get('total_pairs', 0)},
        {"Metric": "", "Value": ""},  # Blank row
        {"Metric": "Average Similarity (Top Matches)", "Value": f"{stats.get('avg_similarity', 0)}%"},
        {"Metric": "Median Similarity (Top Matches)", "Value": f"{stats.get('median_similarity', 0)}%"},
        {"Metric": "Min Similarity (Top Matches)", "Value": f"{stats.get('min_similarity', 0)}%"},
        {"Metric": "Max Similarity (Top Matches)", "Value": f"{stats.get('max_similarity', 0)}%"},
        {"Metric": "", "Value": ""},  # Blank row
        {"Metric": "High Similarity Matches (≥80%)", "Value": stats.get('high_similarity_count', 0)},
        {"Metric": "Medium Similarity Matches (50-79%)", "Value": stats.get('medium_similarity_count', 0)},
        {"Metric": "Low Similarity Matches (<50%)", "Value": stats.get('low_similarity_count', 0)}
    ]

    return pd.DataFrame(stats_data)


def create_summary_csv(stats: Dict) -> bytes:
    """
    Create a summary statistics CSV.

    Args:
        stats: Summary statistics dictionary

    Returns:
        CSV file as bytes
    """
    stats_df = create_stats_dataframe(stats)
    return stats_df.to_csv(index=False).encode('utf-8')
