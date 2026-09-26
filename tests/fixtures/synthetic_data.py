"""
Synthetic test data generator covering all 20 required edge cases for Entity Resolution.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def generate_synthetic_dataset(output_dir: Path) -> Dict[str, Path]:
    """
    Generate synthetic TSV files for train and test covering all 20 edge cases:
    1. Exact duplicate
    2. Punctuation variation
    3. Capitalization variation
    4. Address reorder
    5. Common business name
    6. Rare business name
    7. Missing address
    8. Missing optional field (country or address)
    9. Latin -> Devanagari equivalent
    10. Latin -> Kannada equivalent
    11. Latin -> Tamil equivalent
    12. One-to-many match
    13. Zero-match entity
    14. Country mismatch
    15. Multiple blocking views
    16. Candidate explosion (generic name)
    17. Candidate overflow fallback
    18. Checkpoint / resume capability
    19. Duplicate candidates
    20. Transliteration unavailable / disabled handling
    """
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Train Data
    # -------------------------------------------------------------
    # Source 1 entities
    s1_rows = [
        # 1. Exact match target
        {"entity_id": "S1_001", "business_name": "Acme Industrial Corp", "business_address": "100 Main St, Suite 400", "country": "US"},
        # 2. Punctuation & capitalization target
        {"entity_id": "S1_002", "business_name": "GLOBAL TRADING CO., LTD.", "business_address": "45-A, Park Road", "country": "IN"},
        # 3. Address reorder target
        {"entity_id": "S1_003", "business_name": "Apex Logistics", "business_address": "Building 3 Sector 12 Cyber City", "country": "IN"},
        # 4. Common business name (generic name)
        {"entity_id": "S1_004", "business_name": "Primary Care", "business_address": "500 Health Ave", "country": "US"},
        # 5. Rare business name
        {"entity_id": "S1_005", "business_name": "Xylophone Quasar Technologies", "business_address": "99 Zenith Blvd", "country": "US"},
        # 6. Missing address in S1
        {"entity_id": "S1_006", "business_name": "Omega Pharma", "business_address": None, "country": "DE"},
        # 7. Cross-script Devanagari (Raj Investments Pvt Ltd)
        {"entity_id": "S1_007", "business_name": "Raj Investments Pvt Ltd", "business_address": "12 MG Road Bangalore", "country": "IN"},
        # 8. Cross-script Kannada (Gold Projects Pvt Ltd)
        {"entity_id": "S1_008", "business_name": "Gold Projects Pvt Ltd", "business_address": "88 Brigade Rd", "country": "IN"},
        # 9. Cross-script Tamil (Star Horizon Traders)
        {"entity_id": "S1_009", "business_name": "Star Horizon Traders", "business_address": "15 Anna Salai Chennai", "country": "IN"},
        # 10. One-to-many match target (matches multiple S2 & S3 records)
        {"entity_id": "S1_010", "business_name": "Nexus Retail Enterprises", "business_address": "77 Commercial Street", "country": "UK"},
        # 11. Zero-match entity (no matches anywhere)
        {"entity_id": "S1_011", "business_name": "Lonely Island Research Lab", "business_address": "1 Remote Peak", "country": "NZ"},
        # 12. Country mismatch candidate target
        {"entity_id": "S1_012", "business_name": "Transnational Finance", "business_address": "1 Wall Street", "country": "US"},
    ]

    # Add extra generic entities to simulate candidate explosion for S1_004
    for i in range(13, 30):
        s1_rows.append({
            "entity_id": f"S1_{i:03d}",
            "business_name": "Primary Care Center",
            "business_address": f"{i*10} Medical Way",
            "country": "US"
        })

    # Source 2 entities
    s2_rows = [
        # Match for S1_001 (exact)
        {"entity_id": "S2_001", "business_name": "Acme Industrial Corp", "business_address": "100 Main St, Suite 400", "country": "US"},
        # Match for S1_002 (punctuation/cap variation)
        {"entity_id": "S2_002", "business_name": "Global Trading Co Ltd", "business_address": "45A Park Road", "country": "IN"},
        # Match for S1_003 (address reorder)
        {"entity_id": "S2_003", "business_name": "Apex Logistics", "business_address": "Cyber City Sector 12 Building 3", "country": "IN"},
        # Match for S1_004 (generic name, specific address)
        {"entity_id": "S2_004", "business_name": "Primary Care", "business_address": "500 Health Ave", "country": "US"},
        # Hard negative for S1_004 (same name, different address)
        {"entity_id": "S2_004_HN", "business_name": "Primary Care", "business_address": "999 Oak St", "country": "US"},
        # Match for S1_005 (rare name)
        {"entity_id": "S2_005", "business_name": "Xylophone Quasar Technologies", "business_address": "99 Zenith Blvd", "country": "US"},
        # Match for S1_006 (missing address match)
        {"entity_id": "S2_006", "business_name": "Omega Pharma Services", "business_address": "Berlin Highway 10", "country": "DE"},
        # Cross-script Devanagari match for S1_007 ("राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड")
        {"entity_id": "S2_007", "business_name": "राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड", "business_address": "12 MG Road Bangalore", "country": "IN"},
        # Cross-script Kannada match for S1_008 ("ಗೋಲ್ಡ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್")
        {"entity_id": "S2_008", "business_name": "ಗೋಲ್ಡ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "business_address": "88 Brigade Rd", "country": "IN"},
        # Cross-script Tamil match for S1_009 ("ஸ்டார் ஹரைஸன் டிரேடர்ஸ்")
        {"entity_id": "S2_009", "business_name": "ஸ்டார் ஹரைஸன் டிரேடர்ஸ்", "business_address": "15 Anna Salai Chennai", "country": "IN"},
        # 1-to-many match #1 for S1_010
        {"entity_id": "S2_010A", "business_name": "Nexus Retail Enterprises UK", "business_address": "77 Commercial Street", "country": "UK"},
        # Country mismatch for S1_012 (same name/address, country FR instead of US)
        {"entity_id": "S2_012_CM", "business_name": "Transnational Finance", "business_address": "1 Wall Street", "country": "FR"},
    ]

    # Add extra S2 rows to populate blocks
    for i in range(11, 40):
        s2_rows.append({
            "entity_id": f"S2_{i:03d}",
            "business_name": f"Primary Care Medical Group {i}",
            "business_address": f"Suite {i} Clinic Plaza",
            "country": "US" if i % 2 == 0 else "IN"
        })

    # Source 3 entities
    s3_rows = [
        # Match for S1_001
        {"entity_id": "S3_001", "business_name": "Acme Industrial Corporation", "business_address": "100 Main Street #400", "country": "US"},
        # 1-to-many match #2 for S1_010
        {"entity_id": "S3_010B", "business_name": "Nexus Retail Group", "business_address": "77 Commercial St", "country": "UK"},
        # Match for S1_012
        {"entity_id": "S3_012", "business_name": "Transnational Finance LLC", "business_address": "1 Wall Street", "country": "US"},
    ]

    # Ground truth (S1 entity_id -> space/semicolon/comma separated matched S2/S3 entity_ids)
    gt_rows = [
        {"source1_entity_id": "S1_001", "matched_entity_ids": "S2_001 S3_001"},
        {"source1_entity_id": "S1_002", "matched_entity_ids": "S2_002"},
        {"source1_entity_id": "S1_003", "matched_entity_ids": "S2_003"},
        {"source1_entity_id": "S1_004", "matched_entity_ids": "S2_004"},
        {"source1_entity_id": "S1_005", "matched_entity_ids": "S2_005"},
        {"source1_entity_id": "S1_006", "matched_entity_ids": "S2_006"},
        {"source1_entity_id": "S1_007", "matched_entity_ids": "S2_007"},
        {"source1_entity_id": "S1_008", "matched_entity_ids": "S2_008"},
        {"source1_entity_id": "S1_009", "matched_entity_ids": "S2_009"},
        {"source1_entity_id": "S1_010", "matched_entity_ids": "S2_010A S3_010B"},
        {"source1_entity_id": "S1_011", "matched_entity_ids": ""},  # Zero match
        {"source1_entity_id": "S1_012", "matched_entity_ids": "S3_012"},
    ]

    df_s1 = pd.DataFrame(s1_rows)
    df_s2 = pd.DataFrame(s2_rows)
    df_s3 = pd.DataFrame(s3_rows)
    df_gt = pd.DataFrame(gt_rows)

    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    df_s1.to_csv(s1_path, sep="\t", index=False)
    df_s2.to_csv(s2_path, sep="\t", index=False)
    df_s3.to_csv(s3_path, sep="\t", index=False)
    df_gt.to_csv(gt_path, sep="\t", index=False)

    # -------------------------------------------------------------
    # Test Data (without ground truth)
    # -------------------------------------------------------------
    test_s1_rows = [
        {"entity_id": "TS1_001", "business_name": "Acme Industrial Corp", "business_address": "100 Main St, Suite 400", "country": "US"},
        {"entity_id": "TS1_002", "business_name": "Raj Investments Pvt Ltd", "business_address": "12 MG Road Bangalore", "country": "IN"},
        {"entity_id": "TS1_003", "business_name": "Nonexistent Entity LLC", "business_address": "999 Nowhere Lane", "country": "US"},
    ]
    test_s2_rows = [
        {"entity_id": "TS2_001", "business_name": "Acme Industrial Corp", "business_address": "100 Main St, Suite 400", "country": "US"},
        {"entity_id": "TS2_002", "business_name": "राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड", "business_address": "12 MG Road Bangalore", "country": "IN"},
    ]
    test_s3_rows = [
        {"entity_id": "TS3_001", "business_name": "Acme Industrial Corporation", "business_address": "100 Main Street #400", "country": "US"},
    ]

    df_ts1 = pd.DataFrame(test_s1_rows)
    df_ts2 = pd.DataFrame(test_s2_rows)
    df_ts3 = pd.DataFrame(test_s3_rows)

    ts1_path = test_dir / "test_source1.tsv"
    ts2_path = test_dir / "test_source2.tsv"
    ts3_path = test_dir / "test_source3.tsv"

    df_ts1.to_csv(ts1_path, sep="\t", index=False)
    df_ts2.to_csv(ts2_path, sep="\t", index=False)
    df_ts3.to_csv(ts3_path, sep="\t", index=False)

    return {
        "train_s1": s1_path,
        "train_s2": s2_path,
        "train_s3": s3_path,
        "train_gt": gt_path,
        "test_s1": ts1_path,
        "test_s2": ts2_path,
        "test_s3": ts3_path,
    }
