# Semantic Similarity Analysis for URL Migration

A Streamlit application that analyzes semantic similarity between URLs from two different domains to inform website migration decisions. Perfect for website acquisitions, consolidations, and content audits.

## Overview

When acquiring or migrating a website, determining which pages to migrate, remove, or 301 redirect is challenging. This tool uses semantic embeddings to automatically identify similar content across domains, helping you make data-driven migration decisions.

## Features

- **Dual Input Modes**: Supports both pre-generated embeddings from Screaming Frog or generates embeddings from text content
- **Semantic Matching**: Uses sentence-transformers (all-MiniLM-L6-v2) for accurate semantic similarity
- **Flexible Analysis**: Configure top K matches per URL (1-10 matches)
- **Rich Filtering**: Filter results by similarity threshold and match rank
- **Multiple Export Formats**:
  - CSV export with all URL pairs
  - Excel workbook with multiple sheets (All Pairs, Top Matches, High Similarity, Summary Stats)
  - Summary statistics CSV
- **Visual Statistics**: Summary metrics showing high/medium/low similarity match counts
- **User-Friendly UI**: Clean Streamlit interface with progress tracking

## Use Cases

1. **Website Acquisition**: Identify duplicate/similar content before migrating
2. **Content Consolidation**: Find redundant pages to merge or remove
3. **Migration Planning**: Determine which pages from Domain B should map to Domain A
4. **301 Redirect Mapping**: Identify the best redirect targets for old URLs
5. **Content Audit**: Analyze content overlap between two domains

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/SEOptimize-LLC/Semantic-Similarity-Analysis-for-Migrations.git
cd Semantic-Similarity-Analysis-for-Migrations

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select this repository and set `app.py` as the main file
5. Deploy

## Requirements

- Python 3.8+
- streamlit >= 1.30.0
- pandas >= 2.0.0
- sentence-transformers >= 2.2.0
- torch >= 2.0.0
- scikit-learn >= 1.3.0
- openpyxl >= 3.1.0

## Screaming Frog Export Instructions

### Option 1: Full Bulk Export (Recommended)

1. Crawl your target domain with Screaming Frog
2. Go to **Reports → Export → Bulk Export → All**
3. Save as CSV
4. Repeat for the second domain

### Option 2: Custom Export (Minimal)

If you only need basic text content analysis:

1. After crawling, go to **Reports → Export → Custom → Select Columns**
2. Select at minimum:
   - **Address** (URL)
   - **Title 1**
   - **H1-1**
   - **Meta Description 1**
3. Optionally add body text columns if available
4. Export as CSV
5. Repeat for the second domain

### Option 3: Pre-generated Embeddings

If Screaming Frog supports embedding exports (future feature):

1. Generate embeddings in Screaming Frog
2. Export with embedding columns
3. Upload directly to the app

## Usage Guide

### Step 1: Upload Data

1. Launch the app
2. In the sidebar, upload:
   - **Domain A CSV**: The target domain (where you're migrating TO)
   - **Domain B CSV**: The source domain (where you're migrating FROM)

### Step 2: Configure Analysis

- **Top K Matches**: Choose how many similar URLs to find for each Domain A URL (default: 5)

### Step 3: Preview Data

- Check the **Data Preview** tab to verify:
  - URLs loaded correctly
  - Text content extracted properly
  - Row counts match expectations

### Step 4: Run Analysis

1. Go to the **Similarity Analysis** tab
2. Click **🚀 Run Analysis**
3. Wait for:
   - Embedding generation (or loading)
   - Similarity calculation
   - URL pairing
4. Review summary statistics and results table

### Step 5: Filter Results

- Use the **Minimum Similarity** slider to filter low-quality matches
- Use the **Show Ranks** selector to focus on top matches only

### Step 6: Export Results

1. Go to the **Export Results** tab
2. Optionally customize domain names for the export
3. Download:
   - **CSV**: Simple table of all URL pairs
   - **Excel**: Multi-sheet workbook with organized data
   - **Summary Stats**: Just the summary statistics

## Understanding the Results

### Similarity Scores

- **80-100%**: High similarity - likely duplicate or very similar content
  - **Action**: Consider 301 redirect or remove duplicate
- **50-79%**: Medium similarity - related topics, different approach
  - **Action**: Review manually, may need content consolidation
- **0-49%**: Low similarity - different topics
  - **Action**: Likely needs new content or different redirect target

### Rank

- **Rank 1**: Most similar URL from Domain B for this Domain A URL
- **Rank 2-K**: Alternative matches in descending similarity order

## CSV Output Format

The exported CSV contains:

| Column | Description |
|--------|-------------|
| `domain_a_url` | URL from Domain A (target) |
| `domain_b_url` | URL from Domain B (source) |
| `similarity_score` | Similarity percentage (0-100%) |
| `rank` | Match rank (1 = best match) |
| `analysis_date` | Timestamp of analysis |

## Excel Output Format

### Sheet 1: All URL Pairs
Complete dataset sorted by Domain A URL and rank

### Sheet 2: Top Matches
Only the best match (rank 1) for each Domain A URL

### Sheet 3: High Similarity
URL pairs with similarity ≥ 80%

### Sheet 4: Summary Statistics
Analysis metadata:
- Total URLs analyzed
- Average/median/min/max similarity
- High/medium/low similarity match counts

## Technical Details

### Embedding Model

- **Model**: `all-MiniLM-L6-v2` from sentence-transformers
- **Embedding Dimension**: 384
- **Performance**: ~5-10 seconds for 100 URLs on CPU
- **Similarity Metric**: Cosine similarity

### Text Processing

When generating embeddings from text, the app:
1. Extracts: Title, H1, Meta Description, Body Text
2. Combines into single text string
3. Generates embedding vector
4. Calculates cosine similarity between all pairs

### Memory Requirements

- ~1 MB per 1,000 URLs for embeddings
- Similarity matrix: O(n × m) where n = Domain A URLs, m = Domain B URLs
- Recommended max: 10,000 URLs per domain for Streamlit Cloud free tier

## Troubleshooting

### "Could not parse CSV file"

**Solution**:
- Check CSV delimiter (comma, semicolon, tab)
- Try exporting as UTF-8 encoding
- Verify file is not corrupted

### "No URL column found"

**Solution**:
- Ensure CSV has column named: `Address`, `URL`, or `Page`
- Check first row is headers, not data

### "All text content is empty"

**Solution**:
- Verify export includes: Title 1, H1-1, Meta Description 1
- Check crawl completed successfully in Screaming Frog

### Out of memory errors

**Solution**:
- Reduce dataset size (filter URLs before export)
- Use fewer top K matches
- Deploy on machine with more RAM

## Project Structure

```
Semantic Similarity Analysis (Migration)/
├── app.py                          # Main Streamlit application
├── services/
│   ├── __init__.py
│   ├── embeddings_generator.py     # Embedding generation logic
│   └── similarity_analyzer.py      # Similarity calculation
├── utils/
│   ├── __init__.py
│   ├── file_parser.py              # CSV parsing
│   └── export_handler.py           # Export functionality
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use for commercial projects

## Support

For issues or questions:
- Open an issue on GitHub
- Email: support@seoptimize.com

## Credits

- Built with [Streamlit](https://streamlit.io/)
- Embeddings powered by [sentence-transformers](https://www.sbert.net/)
- Developed by SEOptimize LLC

---

**Version**: 1.0.0
**Last Updated**: 2025-02-12
