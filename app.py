"""
Semantic Similarity Analysis for URL Migration

Streamlit app for analyzing semantic similarity between URLs from two domains
to inform website migration decisions.
"""

import streamlit as st
import pandas as pd
from services.embeddings_generator import EmbeddingsGenerator
from services.similarity_analyzer import SimilarityAnalyzer
from utils.file_parser import parse_screaming_frog_csv, validate_dataframe
from utils.export_handler import create_csv_download, create_excel_download, create_summary_csv

# Page configuration
st.set_page_config(
    page_title="Semantic Similarity Analysis",
    page_icon="🔗",
    layout="wide"
)

# Title and description
st.title("🔗 Semantic Similarity Analysis for URL Migration")
st.markdown("""
Compare URLs from two domains using semantic embeddings to identify similar content.
Perfect for website migrations, consolidations, and content audits.
""")

# Initialize session state
if 'domain_a_df' not in st.session_state:
    st.session_state.domain_a_df = None
if 'domain_b_df' not in st.session_state:
    st.session_state.domain_b_df = None
if 'domain_a_mode' not in st.session_state:
    st.session_state.domain_a_mode = None
if 'domain_b_mode' not in st.session_state:
    st.session_state.domain_b_mode = None
if 'embeddings_a' not in st.session_state:
    st.session_state.embeddings_a = None
if 'embeddings_b' not in st.session_state:
    st.session_state.embeddings_b = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'stats' not in st.session_state:
    st.session_state.stats = None

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📁 Upload Data Files")

    domain_a_file = st.file_uploader(
        "Domain A (Target Domain)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload Screaming Frog export for the domain you're migrating TO"
    )

    domain_b_file = st.file_uploader(
        "Domain B (Source Domain)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload Screaming Frog export for the domain you're migrating FROM"
    )

    st.divider()

    st.subheader("🎯 Analysis Settings")

    top_k = st.number_input(
        "Top K Matches",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of most similar URLs to show for each Domain A URL"
    )

    st.divider()

    st.subheader("📖 Instructions")
    st.markdown("""
    **Expected CSV Format:**
    - **URL Column**: `Address`, `URL`, or `Page`
    - **Text Columns**: `Title 1`, `H1-1`, `Meta Description 1`, etc.
    - **OR Pre-generated Embeddings**: Columns starting with `Embedding`

    **Screaming Frog Export Settings:**
    1. Crawl both domains separately
    2. Export: `Reports → Export → Bulk Export → All`
    3. Or: Custom export with Address, Title, H1, Meta Description
    """)

# Main content area
if domain_a_file and domain_b_file:
    # Parse uploaded files
    if st.session_state.domain_a_df is None or st.session_state.domain_b_df is None:
        try:
            with st.spinner("📊 Parsing Domain A file..."):
                st.session_state.domain_a_df, st.session_state.domain_a_mode = parse_screaming_frog_csv(domain_a_file)

            with st.spinner("📊 Parsing Domain B file..."):
                st.session_state.domain_b_df, st.session_state.domain_b_mode = parse_screaming_frog_csv(domain_b_file)

            # Validate
            is_valid_a, error_a = validate_dataframe(st.session_state.domain_a_df, st.session_state.domain_a_mode)
            is_valid_b, error_b = validate_dataframe(st.session_state.domain_b_df, st.session_state.domain_b_mode)

            if not is_valid_a:
                st.error(f"❌ Domain A validation error: {error_a}")
                st.stop()

            if not is_valid_b:
                st.error(f"❌ Domain B validation error: {error_b}")
                st.stop()

            st.success("✅ Files parsed successfully")

        except Exception as e:
            st.error(f"❌ Error parsing files: {str(e)}")
            st.stop()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🔍 Similarity Analysis", "📥 Export Results"])

    # Tab 1: Data Preview
    with tab1:
        st.subheader("Domain A (Target Domain)")
        st.info(f"**Mode**: {st.session_state.domain_a_mode.upper()} | **URLs**: {len(st.session_state.domain_a_df)}")

        if st.session_state.domain_a_mode == 'text':
            preview_cols = ['url', 'title', 'h1', 'meta_description']
        else:
            preview_cols = ['url'] + [col for col in st.session_state.domain_a_df.columns if col.startswith('embedding_')][:5]

        st.dataframe(
            st.session_state.domain_a_df[preview_cols].head(10),
            use_container_width=True
        )

        st.divider()

        st.subheader("Domain B (Source Domain)")
        st.info(f"**Mode**: {st.session_state.domain_b_mode.upper()} | **URLs**: {len(st.session_state.domain_b_df)}")

        if st.session_state.domain_b_mode == 'text':
            preview_cols_b = ['url', 'title', 'h1', 'meta_description']
        else:
            preview_cols_b = ['url'] + [col for col in st.session_state.domain_b_df.columns if col.startswith('embedding_')][:5]

        st.dataframe(
            st.session_state.domain_b_df[preview_cols_b].head(10),
            use_container_width=True
        )

    # Tab 2: Similarity Analysis
    with tab2:
        st.subheader("🔍 Run Similarity Analysis")

        st.markdown(f"""
        **Ready to analyze:**
        - Domain A: {len(st.session_state.domain_a_df)} URLs
        - Domain B: {len(st.session_state.domain_b_df)} URLs
        - Total comparisons: {len(st.session_state.domain_a_df) * len(st.session_state.domain_b_df):,} URL pairs
        - Top K matches per URL: {top_k}
        """)

        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            try:
                # Initialize components
                embeddings_gen = EmbeddingsGenerator()
                analyzer = SimilarityAnalyzer(top_k=top_k)

                # Generate or load embeddings for Domain A
                if st.session_state.domain_a_mode == 'text':
                    st.session_state.embeddings_a = embeddings_gen.generate_embeddings(
                        st.session_state.domain_a_df['combined_text'].tolist()
                    )
                else:
                    st.session_state.embeddings_a = embeddings_gen.load_pregenerated_embeddings(
                        st.session_state.domain_a_df
                    )

                # Generate or load embeddings for Domain B
                if st.session_state.domain_b_mode == 'text':
                    st.session_state.embeddings_b = embeddings_gen.generate_embeddings(
                        st.session_state.domain_b_df['combined_text'].tolist()
                    )
                else:
                    st.session_state.embeddings_b = embeddings_gen.load_pregenerated_embeddings(
                        st.session_state.domain_b_df
                    )

                # Calculate similarity matrix
                similarity_matrix = analyzer.calculate_similarity_matrix(
                    st.session_state.embeddings_a,
                    st.session_state.embeddings_b
                )

                # Pair URLs
                st.session_state.results_df = analyzer.pair_urls(
                    st.session_state.domain_a_df['url'].tolist(),
                    st.session_state.domain_b_df['url'].tolist(),
                    similarity_matrix
                )

                # Generate summary stats
                st.session_state.stats = analyzer.generate_summary_stats(
                    st.session_state.results_df
                )

                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.stop()

        # Display results
        if st.session_state.results_df is not None:
            st.divider()
            st.subheader("📊 Summary Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Avg Similarity",
                    f"{st.session_state.stats['avg_similarity']}%"
                )

            with col2:
                st.metric(
                    "High Matches (≥80%)",
                    st.session_state.stats['high_similarity_count']
                )

            with col3:
                st.metric(
                    "Medium Matches (50-79%)",
                    st.session_state.stats['medium_similarity_count']
                )

            with col4:
                st.metric(
                    "Low Matches (<50%)",
                    st.session_state.stats['low_similarity_count']
                )

            st.divider()
            st.subheader("🔗 URL Pairs")

            # Filter controls
            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                min_similarity = st.slider(
                    "Minimum Similarity (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    help="Filter results by minimum similarity score"
                )

            with filter_col2:
                rank_filter = st.multiselect(
                    "Show Ranks",
                    options=list(range(1, top_k + 1)),
                    default=[1],
                    help="Select which match ranks to display"
                )

            # Apply filters
            filtered_results = st.session_state.results_df[
                (st.session_state.results_df['similarity_score'] >= min_similarity) &
                (st.session_state.results_df['rank'].isin(rank_filter))
            ]

            st.info(f"Showing {len(filtered_results)} of {len(st.session_state.results_df)} total pairs")

            # Display results table
            st.dataframe(
                filtered_results,
                use_container_width=True,
                column_config={
                    "domain_a_url": st.column_config.TextColumn("Domain A URL", width="large"),
                    "domain_b_url": st.column_config.TextColumn("Domain B URL", width="large"),
                    "similarity_score": st.column_config.NumberColumn(
                        "Similarity (%)",
                        format="%.2f%%"
                    ),
                    "rank": st.column_config.NumberColumn("Rank")
                }
            )

    # Tab 3: Export Results
    with tab3:
        st.subheader("📥 Export Analysis Results")

        if st.session_state.results_df is not None:
            st.success(f"✅ Results ready for export ({len(st.session_state.results_df)} URL pairs)")

            # Domain names for export
            col1, col2 = st.columns(2)
            with col1:
                domain_a_name = st.text_input(
                    "Domain A Name (for export)",
                    value="Domain A",
                    help="Friendly name for the target domain"
                )
            with col2:
                domain_b_name = st.text_input(
                    "Domain B Name (for export)",
                    value="Domain B",
                    help="Friendly name for the source domain"
                )

            st.divider()

            # Export buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                csv_data = create_csv_download(st.session_state.results_df)
                st.download_button(
                    label="📄 Download CSV (All Pairs)",
                    data=csv_data,
                    file_name=f"similarity_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                excel_data = create_excel_download(
                    st.session_state.results_df,
                    st.session_state.stats,
                    domain_a_name,
                    domain_b_name
                )
                st.download_button(
                    label="📊 Download Excel (Multi-Sheet)",
                    data=excel_data,
                    file_name=f"similarity_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with col3:
                summary_csv = create_summary_csv(st.session_state.stats)
                st.download_button(
                    label="📈 Download Summary Stats",
                    data=summary_csv,
                    file_name=f"summary_stats_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.divider()
            st.subheader("📋 Export Contents")

            with st.expander("CSV Export"):
                st.markdown("""
                **Columns:**
                - `domain_a_url`: URL from Domain A (target)
                - `domain_b_url`: URL from Domain B (source)
                - `similarity_score`: Similarity percentage (0-100%)
                - `rank`: Match rank (1 = most similar)
                - `analysis_date`: Timestamp of analysis
                """)

            with st.expander("Excel Export (Multiple Sheets)"):
                st.markdown("""
                **Sheet 1: All URL Pairs**
                - All URL pairs sorted by Domain A URL and rank

                **Sheet 2: Top Matches**
                - Only rank 1 matches (best match per URL)

                **Sheet 3: High Similarity**
                - Pairs with similarity ≥80%

                **Sheet 4: Summary Statistics**
                - Analysis metadata and statistics
                """)

        else:
            st.warning("⚠️ No results to export yet. Run the similarity analysis first.")

else:
    # Welcome message when no files uploaded
    st.info("👆 Upload CSV/Excel files for both domains using the sidebar to get started.")

    st.divider()

    st.subheader("💡 How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **1️⃣ Upload Data**
        - Export URLs from Screaming Frog
        - Upload Domain A (target)
        - Upload Domain B (source)
        """)

    with col2:
        st.markdown("""
        **2️⃣ Analyze**
        - Generate semantic embeddings
        - Calculate similarity scores
        - Pair similar URLs
        """)

    with col3:
        st.markdown("""
        **3️⃣ Export**
        - Download results as CSV/Excel
        - Review top matches
        - Make migration decisions
        """)

# Footer
st.divider()
st.caption("Built with Streamlit | Powered by sentence-transformers")
